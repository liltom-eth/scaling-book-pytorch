# torch.compile and torch.export

`torch.compile` is PyTorch's JIT compiler, introduced in PyTorch 2.0. It takes PyTorch code written for eager mode and compiles it to optimized CUDA kernels — typically providing 20–50% speedup on training and inference with a single line of code.

This chapter covers how the compiler works, how to use it effectively, and how to debug it when things go wrong. It also covers `torch.export`, PyTorch's mechanism for freezing and deploying compiled models.

## What torch.compile Does

When you call `torch.compile(model)`, three subsystems activate:

```
Python function
      ↓
TorchDynamo  — captures the compute graph by tracing Python bytecode
      ↓
AOTAutograd  — generates forward + backward graph together (for training)
      ↓
TorchInductor  — optimizes the graph and generates fused CUDA/Triton kernels
```

**TorchDynamo** is a Python bytecode tracing system. It intercepts Python execution, records all PyTorch operations, and handles Python control flow (if/else, loops) via "graph breaks" — points where the compiler stops tracing and falls back to Python.

**AOTAutograd** (Ahead-of-Time Autograd) traces both the forward pass and the backward pass graph together before execution, enabling the Inductor backend to optimize across both.

**TorchInductor** is the default backend. It takes the compute graph and:
1. Fuses pointwise operations (relu + add + layer_norm → single kernel)
2. Generates Triton kernels for fused operations
3. Applies CUDA Graphs to eliminate kernel launch overhead
4. Optimizes memory access patterns for Tensor Core utilization

## Basic Usage

```python
import torch

model = MyTransformer().cuda()

# Single line — that's it
compiled_model = torch.compile(model)

# First forward pass triggers compilation (slow, 30s–5min for large models)
output = compiled_model(input)

# Subsequent calls use the compiled graph (fast)
for batch in dataloader:
    output = compiled_model(batch)
```

For **inference only** (no backward pass), wrap the forward call instead:

```python
@torch.compile
def forward(model, x):
    return model(x)

# Or for a module
model = torch.compile(model, fullgraph=True)
with torch.no_grad():
    output = model(input)
```

## Compilation Modes

```python
# Default — good balance of compile time and runtime performance
model = torch.compile(model)

# Maximize runtime performance at the cost of longer compile time
model = torch.compile(model, mode="max-autotune")

# Minimize kernel launch overhead via CUDA Graphs (best for fixed shapes)
model = torch.compile(model, mode="reduce-overhead")
```

| Mode | Compile time | Runtime speedup | When to use |
|:---|:---:|:---:|:---|
| `default` | ~30s | ~20% | Starting point |
| `max-autotune` | 5–30 min | ~30–50% | Production training/inference |
| `reduce-overhead` | ~60s | ~15% on compute + lower overhead | Many small ops (decode) |

`max-autotune` runs TorchInductor's autotuning loop, testing multiple Triton kernel configurations for each op and picking the fastest. The compiled kernel is hardware-specific — cache compiled artifacts to avoid repeated autotuning.

## What Gets Fused

TorchInductor fuses:
- **Pointwise chains:** consecutive elementwise ops on the same tensor (relu, add, mul, layer_norm scale/bias application)
- **Reduction + pointwise:** e.g., layer_norm (reduction) immediately followed by an activation
- **Attention + dropout:** the softmax + dropout + value matmul path

It does **not** fuse:
- Cross-tensor dependencies requiring synchronization
- cuBLAS GEMM calls (these are already optimal; Triton won't beat cuBLAS for standard matmul)
- Operations that require dynamic shape analysis

Check what's being fused by looking at the generated code:

```python
import torch._inductor.config as inductor_config
inductor_config.debug = True

model = torch.compile(model, mode="max-autotune")
output = model(input)  # prints generated Triton code to stdout
```

## Dynamic Shapes

By default, `torch.compile` assumes input shapes are static — it recompiles for each new shape. This is correct but expensive for variable-length inputs.

```python
# Option 1: mark inputs as dynamic
model = torch.compile(model, dynamic=True)

# Option 2: use torch.export with dynamic dimensions explicitly declared
from torch.export import export, Dim

batch_dim = Dim("batch", min=1, max=32)
seq_dim = Dim("seq", min=1, max=8192)

program = export(
    model,
    args=(example_input,),
    dynamic_shapes={"x": {0: batch_dim, 1: seq_dim}},
)
```

With `dynamic=True`, the compiler generates kernels that work for a range of shapes using symbolic shapes. Dynamic kernels are ~5–15% slower than static kernels.

**Practical advice:** use static shapes during training (pad sequences to a fixed length per batch), and use `dynamic=True` or shape bucketing for inference with variable prompt lengths.

## Graph Breaks

Graph breaks occur when TorchDynamo encounters Python code it can't trace:
- Arbitrary Python objects that aren't tensors (custom classes with non-standard `__call__`)
- Python data structures that change shape based on runtime values
- Some third-party C extensions
- `print()` statements on tensor values (forces a GPU sync, breaks the graph)

When a graph break occurs, the region between breaks runs in Python eager mode — nullifying the compilation benefit for that region.

**To investigate graph breaks:**

```python
import torch._dynamo
torch._dynamo.explain(model)(input)
```

This prints a summary of all graph breaks and their causes.

**To minimize graph breaks:**
- Avoid `if` conditions that depend on tensor values (use `torch.where` instead)
- Replace custom Python classes in the forward path with standard `nn.Module`
- Use `@torch.jit.script` for utility functions called in the forward pass

## CUDA Graphs via torch.compile

`torch.compile(mode="reduce-overhead")` uses CUDA Graphs internally to reduce kernel launch overhead. Each kernel launch has ~2–10 µs of CPU overhead. For a model with thousands of small operations, this adds up to milliseconds per step.

CUDA Graphs capture the entire forward pass as a single GPU work submission:

- **Without CUDA Graphs:** CPU submits N kernel launches, GPU processes them in sequence
- **With CUDA Graphs:** CPU submits 1 graph replay command, GPU executes N kernels without CPU involvement

The performance gain is most significant when:
- The GPU is fast (H100, A100) and step times are short
- The model has many small operations (attention + residuals + layer norms)
- Shapes are fixed

For training with FSDP2, CUDA Graphs have constraints — use `torch.compile` with FSDP2 (not FSDP1) for best compatibility.

## Caching Compiled Artifacts

Recompiling on every run wastes time. Cache compiled kernels:

```python
import torch._inductor.config as config

# Cache FX graphs between runs
config.fx_graph_cache = True

# For torch.export: save and load a compiled program
from torch._export import aot_compile, aot_load

program = export(model, args=(example_input,))
path = aot_compile(program, example_input)  # compiles and saves a .so file
model_so = aot_load(path)                   # load without recompiling
```

Set the `TORCHINDUCTOR_CACHE_DIR` environment variable to a persistent directory to reuse compiled artifacts across runs:

```bash
export TORCHINDUCTOR_CACHE_DIR=/shared/torch_cache
python train.py
```

## torch.export

`torch.export` creates a portable, fully-traced representation of a PyTorch model — no Python interpreter required at inference time. This is used for:
- **Deployment to non-Python environments** (mobile, embedded, C++ servers)
- **AOT compilation** with TensorRT or other backends
- **ExecuTorch** for edge deployment (iOS, Android)

```python
from torch.export import export

# Export with example inputs
exported = export(
    model,
    args=(example_input,),
)

# Serialize
torch.export.save(exported, "model.pt2")

# Load and run
loaded = torch.export.load("model.pt2")
output = loaded.module()(input)
```

The exported program is a graph of ATen operations without Python control flow. It can be compiled by TorchInductor, TensorRT, or other backends:

```python
# Compile with TensorRT for maximum inference throughput on H100
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    exported.module(),
    inputs=[torch_tensorrt.Input(example_input.shape, dtype=torch.bfloat16)],
    enabled_precisions={torch.bfloat16},
)
output = trt_model(input)
```

## Practical Benchmarking

Always benchmark `torch.compile` on your actual workload:

```python
import torch
import time

model = MyTransformer().cuda()
compiled = torch.compile(model, mode="max-autotune")
x = torch.randn(batch_size, seq_len, d_model, device='cuda', dtype=torch.bfloat16)

# Warmup — includes compilation on first call
for _ in range(5):
    compiled(x)
torch.cuda.synchronize()

# Benchmark compiled
N = 100
start = time.perf_counter()
for _ in range(N):
    compiled(x)
torch.cuda.synchronize()
t_compiled = (time.perf_counter() - start) / N * 1000

# Benchmark eager
for _ in range(5):
    model(x)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(N):
    model(x)
torch.cuda.synchronize()
t_eager = (time.perf_counter() - start) / N * 1000

print(f"Compiled: {t_compiled:.2f} ms/iter")
print(f"Eager:    {t_eager:.2f} ms/iter")
print(f"Speedup:  {t_eager / t_compiled:.2f}x")
```

Expected speedup ranges:
- Inference (no grad): 1.3–2.0×
- Training forward+backward: 1.1–1.4×
- Training with FSDP: 1.05–1.2× (communication is not compiled)

## Key Takeaways

- **`torch.compile` is one line of code** that typically gives 20–50% speedup. Add it to every model by default; only remove it if it causes correctness issues or compilation failures.

- **TorchDynamo captures Python bytecode** and handles Python control flow via graph breaks. Minimize graph breaks by keeping the forward pass in PyTorch-native code.

- **TorchInductor fuses pointwise operations** into single Triton kernels, eliminating redundant HBM reads/writes. This is most impactful for chains of small elementwise ops after large matmuls.

- **`mode="max-autotune"` for production.** Pay the 5–30 minute compile time once; cache the artifacts with `TORCHINDUCTOR_CACHE_DIR`.

- **Use static shapes whenever possible.** Dynamic shapes reduce optimization opportunities. Pad training sequences to fixed lengths; use shape bucketing for inference.

- **`torch.export` for deployment.** Export captures the model as a portable graph without Python dependencies. Use it for TensorRT compilation, mobile deployment, or C++ serving.
