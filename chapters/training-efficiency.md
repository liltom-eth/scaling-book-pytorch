# Training Efficiency

A model that fits on your GPUs and trains correctly is just the beginning. This chapter covers techniques to make training faster and more memory-efficient without changing the model architecture or the data.

The three levers are:
1. **Mixed precision** — use lower-precision numerics to reduce memory and increase compute throughput
2. **Gradient checkpointing** — trade recompute for activation memory
3. **Gradient accumulation and micro-batching** — decouple batch size from per-step memory

Each technique targets a different bottleneck. Understanding which one you're hitting is the key to applying the right fix.

## Memory Budget

Before optimizing, you need to know where memory is going. For a transformer with P parameters trained with Adam:

| Component | Memory (BF16 params, FP32 optimizer) |
|:---|:---|
| Parameters | 2P bytes (BF16) |
| Optimizer state (Adam m, v) | 8P bytes (FP32 m and v) |
| Gradients | 2P bytes (BF16) or 4P bytes (FP32) |
| Activations | depends on batch size and checkpointing |
| **Total (no checkpointing)** | **~18P bytes + activations** |

For LLaMA 3 8B (P = 8e9): parameters + optimizer + gradients ≈ 144 GB. An H100 has 80 GB. This is why FSDP (Chapter 6) is necessary — but even after sharding, activations can be the next bottleneck.

```python
import torch

# Check memory before and after model loading
print(f"Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
model = MyTransformer().cuda()
print(f"After model: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

## Mixed Precision Training

### BF16 Training

BF16 (bfloat16) has the same exponent range as FP32 (8 exponent bits) with reduced mantissa precision (7 bits vs 23 bits). On H100 and A100, BF16 matrix operations use Tensor Cores and run at up to 8× the throughput of FP32.

The standard PyTorch approach uses `torch.autocast` with a BF16 context for the forward pass:

```python
import torch

for batch in dataloader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output, target)

    # For BF16, no gradient scaler needed — same exponent range as FP32
    loss.backward()
    optimizer.step()
```

For BF16, gradient scaling is unnecessary because BF16 cannot underflow to zero the way FP16 can. This simplifies the training loop vs. FP16 mixed precision.

**What runs in BF16 vs FP32:**
- BF16: matmuls, convolutions, attention — the compute-intensive parts
- FP32 (by default): optimizer states, loss accumulation
- FP32: layer norm, softmax (PyTorch's autocast keeps these in FP32 for stability)

### When BF16 Is Not Enough: FP8

On H100, FP8 (8-bit floating point) doubles throughput again over BF16 — up to 1979 TFLOP/s with sparsity. FP8 is useful for very large models where memory and compute are extreme. For training, FP8 requires per-tensor scaling to avoid underflow/overflow.

PyTorch's `torchao` library provides FP8 training:

```python
from torchao.float8 import convert_to_float8_training

# Convert model to use FP8 for matmuls
convert_to_float8_training(
    model,
    module_filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear)
)
```

Start with BF16 and move to FP8 only if memory or throughput are the bottleneck.

### TF32 (Default on Ampere+)

Even without explicit autocast, PyTorch uses TF32 for FP32 matmuls on Ampere+ GPUs by default. TF32 uses 10-bit mantissa (vs 23-bit FP32) but runs through Tensor Cores at full throughput. Precision loss is negligible for training.

```python
# TF32 is enabled by default
print(torch.backends.cuda.matmul.allow_tf32)  # True
```

## Gradient Checkpointing

### The Problem

During the backward pass, PyTorch needs the activations from the forward pass to compute gradients. Without checkpointing, all intermediate activations are kept in memory. For a transformer with L layers, this is roughly:

$$\text{Activation memory} \approx 20 \cdot B \cdot T \cdot D \cdot L \cdot 2 \text{ bytes}$$

For LLaMA 3 8B with B·T = 1M tokens, L=32, D=4096: ≈ 5.2 TB. This is enormous compared to the 80 GB HBM on an H100.

### How Checkpointing Works

Instead of saving all activations, save only the inputs at checkpointing boundaries. During the backward pass, re-run the forward pass to recompute the activations on the fly.

- **Memory:** saves only 1 activation per checkpoint instead of ~20
- **Cost:** re-runs the forward pass, increasing total FLOPs by ~33% (from 6P×T to ~8P×T)

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # use_reentrant=False is strongly preferred for modern PyTorch
        return checkpoint(self.layer, x, use_reentrant=False)
```

**`use_reentrant=False` is strongly preferred** — it avoids gotchas of the older reentrant implementation (such as requiring all inputs to have `requires_grad=True`).

### Selective Checkpointing

Checkpointing every layer maximizes memory savings but always pays the full 33% compute overhead. You can checkpoint selectively:

```python
# Checkpoint every other layer — halves both memory savings and compute overhead
for i, layer in enumerate(model.layers):
    if i % 2 == 0:
        model.layers[i] = CheckpointedLayer(layer)
```

A common heuristic: checkpoint the attention blocks but not the MLP blocks, since the $QK^T$ attention matrix is large while MLP activations are smaller.

### Activation Checkpointing with FSDP

When using FSDP, PyTorch provides a convenience wrapper:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointWrapper,
    CheckpointImpl,
)
from functools import partial

non_reentrant_wrapper = partial(
    CheckpointWrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=lambda m: isinstance(m, TransformerLayer),
)
```

## Gradient Accumulation

### The Problem

Large effective batch sizes improve training stability. But large batch sizes require large activation memory. With a fixed GPU memory budget, you hit an OOM at some batch size.

### How Accumulation Works

Split the macro-batch into M micro-batches. Run forward + backward for each micro-batch, accumulating gradients in `.grad` buffers. Apply the optimizer update once per macro-batch.

```python
accumulation_steps = 8  # effective batch is 8× the per-GPU micro-batch
optimizer.zero_grad()

for i, micro_batch in enumerate(micro_batches):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(micro_batch) / accumulation_steps  # normalize loss
    loss.backward()  # gradients accumulate in .grad buffers

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Loss normalization is critical.** Dividing loss by `accumulation_steps` ensures gradients are equivalent to a single forward pass on the full batch.

### Gradient Accumulation with DDP

By default, DDP launches an AllReduce after every `.backward()` call. With accumulation, this wastes bandwidth — you're doing M AllReduces per optimizer step instead of 1. Use `model.no_sync()` to disable gradient syncing during accumulation steps:

```python
import contextlib

for i, micro_batch in enumerate(micro_batches):
    sync = (i + 1) % accumulation_steps == 0
    ctx = contextlib.nullcontext() if sync else model.no_sync()
    with ctx:
        loss = model(micro_batch) / accumulation_steps
        loss.backward()

if sync:
    optimizer.step()
    optimizer.zero_grad()
```

FSDP handles this automatically — the ReduceScatter fires only when `require_backward_grad_sync=True` (set automatically on the last micro-batch).

## Activation Offloading

When GPU memory is the bottleneck but you have fast CPU RAM available, activations can be offloaded to CPU and paged back during the backward pass:

```python
from torch.utils.checkpoint import checkpoint

def checkpoint_with_offload(fn, *args):
    return checkpoint(fn, *args, use_reentrant=False, offload_to_cpu=True)
```

Offloading to CPU (via PCIe at ~64 GB/s) is slower than recomputation (which uses GPU compute) and should only be used when the model is too large even with full checkpointing. Measure both before choosing.

## Compile for Throughput

`torch.compile` provides free throughput improvements by fusing adjacent operations and reducing kernel launch overhead. For a training loop:

```python
model = torch.compile(model, mode="max-autotune")
```

In training, compilation fuses operations like layer norm + matmul, attention + dropout, etc. into fewer kernels. For H100, this typically provides 5–20% throughput improvement with no changes beyond the `compile()` call.

**Caveats:**
- First-step compilation time can be 5–30 minutes for large models
- Dynamic shapes (varying sequence lengths) trigger recompilation — use `dynamic=True` or pad to fixed shapes
- Some custom ops not registered with `torch.library` bypass compilation

## Worked Example: LLaMA 3 8B Memory Budget

Training LLaMA 3 8B (P=8B, D=4096, L=32) with BF16 and Adam FP32:

**Baseline (no techniques):**
- Parameters: 8B × 2 bytes = 16 GB
- Optimizer state: 8B × 8 bytes = 64 GB
- Gradients: 8B × 2 bytes = 16 GB
- Activations (no checkpoint, B·T=1M): ≈ 5.2 TB
- **Total: >> 80 GB** — impossible on a single H100

**With FSDP across 8 GPUs + gradient checkpointing:**
- Params: 16/8 = 2 GB
- Optimizer: 64/8 = 8 GB
- Gradients: 16/8 = 2 GB
- Activations (checkpoint per layer, 1M/8 = 125K tokens): 125K × 4096 × 32 × 2 bytes ≈ 32.5 GB
- **Total per GPU: ~44.5 GB** — fits on one H100 (80 GB) ✓

Check memory distribution during training:

```python
print(torch.cuda.memory_summary(device=0, abbreviated=True))
```

## Key Takeaways

- **BF16 autocast is the default starting point.** It halves parameter/gradient memory vs FP32 and unlocks Tensor Core throughput. Start every training run in BF16.

- **Gradient checkpointing trades 33% compute for ~20× activation memory reduction.** Use it whenever activations are the bottleneck; the compute overhead is usually worth it.

- **Gradient accumulation decouples effective batch size from per-GPU memory.** Use `model.no_sync()` in DDP to avoid redundant AllReduces during accumulation.

- **`torch.compile` is free throughput** on H100. Add it to your training loop by default; only remove if it causes correctness issues.

- **Profile before you optimize.** `torch.cuda.memory_summary()` shows where memory goes. The PyTorch profiler (covered in [Chapter 16](profiling.md)) shows where time goes. Fix the actual bottleneck, not the assumed one.
