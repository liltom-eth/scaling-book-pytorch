# Training Parallelism

## What Do We Mean By Scaling?

The goal of training parallelism is to use more GPUs while achieving a proportional, linear increase in throughput (*strong scaling*). Performance on a single GPU depends on arithmetic intensity — are we compute-bound or memory-bound? Performance at the cluster level depends on hiding inter-GPU communication behind useful compute. The two constraints interact: adding GPUs reduces per-GPU work while increasing communication volume.

This chapter covers four parallelism strategies used in practice: **data parallelism (DDP)**, **fully sharded data parallelism (FSDP/ZeRO)**, **tensor parallelism (TP)**, and **pipeline parallelism (PP)**. For each, we derive when communication starts to bottleneck compute.

We use the following notation throughout.

| Symbol | Meaning |
|:---|:---|
| D | d\_model (hidden dimension) |
| F | d\_ff (feed-forward dimension) |
| B | Total batch size in tokens |
| L | Number of layers |
| C | GPU FLOPs/s (BF16 without sparsity) |
| W | Bidirectional link bandwidth |
| X, Y | Number of GPUs along a mesh axis |

For H100 SXM5: C = 989 TFLOPs/s, HBM bandwidth = 3.35 TB/s. Critical arithmetic intensity = C / HBM ≈ 295 FLOPs/byte — we need ~295 BF16 tokens processed per weight byte loaded to be compute-bound. For a single linear layer with weight $$W[D, F]$$ in BF16 ($$2DF$$ bytes), this requires batch size $$B > 295 / 2 \approx 148$$ tokens.

We model a Transformer as a stack of two-matrix MLP blocks (attention is typically a small fraction of FLOPs for D ≥ 4096). Each layer:

$$\text{Out}[B, D] = \text{In}[B, D] \cdot W_\text{in}[D, F] \cdot W_\text{out}[F, D]$$

Training FLOPs per layer: $$6BTDF$$ ($$2BTDF$$ forward, $$4BTDF$$ backward for dW and dIn).

## Data Parallelism (DDP)

**Idea:** each GPU holds a complete copy of the model. Activations are split along the batch dimension. After each backward pass, gradients are summed across all GPUs with an AllReduce.

```
GPU 0: In[B/X, D] · W_in[D, F] · W_out[F, D]  →  Out[B/X, D]
GPU 1: In[B/X, D] · W_in[D, F] · W_out[F, D]  →  Out[B/X, D]
...      ← AllReduce(dW_in, dW_out) at end of backward →
```

In PyTorch, `DistributedDataParallel` (DDP) implements this:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = MyTransformer().cuda()
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)

# DDP overlaps AllReduce of gradients with backward pass
for batch in dataloader:
    loss = model(batch)
    loss.backward()   # gradient AllReduce happens here
    optimizer.step()
    optimizer.zero_grad()
```

**Communication cost:** gradient AllReduce after each backward pass. An AllReduce of $$V$$ bytes costs $$2V / W$$ on a ring. Per layer, we AllReduce $$W_\text{in}$$ and $$W_\text{out}$$, totaling $$2 \times 2DF$$ bytes (BF16). With $$L$$ layers, total comms:

$$T_\text{comms} = \frac{2 \times 4DFL}{W}$$

**Compute time per step** (forward + backward):

$$T_\text{math} = \frac{6BTDFL}{C}$$

**Compute-bound condition** ($$T_\text{math} > T_\text{comms}$$):

$$\frac{6BTDFL}{C} > \frac{8DFL}{W} \implies \frac{B}{X} > \frac{4C}{3W}$$

Note: DDP overlaps the AllReduce with the backward pass via gradient bucketing (`bucket_cap_mb=25`). The effective bottleneck is whichever of compute or comms is larger — when overlap is perfect, you pay only max(T\_math, T\_comms).

**H100 numbers:**

- **Within a node (NVLink, W = 900 GB/s):** minimum $$B/X > 4 \times 989\text{e12} / (3 \times 900\text{e9}) \approx 1,465$$ tokens/GPU
- **Cross-node (InfiniBand NDR, W = 50 GB/s):** minimum $$B/X > 4 \times 989\text{e12} / (3 \times 50\text{e9}) \approx 26,373$$ tokens/GPU

Cross-node DDP requires very large per-GPU batch sizes. This is why DDP gradient AllReduces are bucketed — DDP groups small gradient tensors into 25 MB buckets to maximize InfiniBand bandwidth utilization.

**When to use DDP:** your model fits on a single GPU (parameters + optimizer state). With Adam in FP32 optimizer state and BF16 parameters, you need roughly 18 bytes/parameter, so a 7B model needs ~126 GB — more than one H100's 80 GB of HBM. DDP alone cannot handle models larger than ~4B parameters on H100.

## Fully Sharded Data Parallelism (FSDP / ZeRO)

**Idea:** instead of replicating parameters everywhere, shard them across GPUs. Each GPU holds only 1/X of each parameter tensor. Before a layer's forward pass, parameters are reconstructed via AllGather. After the backward pass, gradients are ReduceScattered — each GPU accumulates only its parameter shard's gradient.

This is called ZeRO-3 (Zero Redundancy Optimizer stage 3) in DeepSpeed terminology, and FSDP in PyTorch's native implementation.

```
GPU 0: holds W_in[D/X, F]   GPU 1: holds W_in[D/X, F]   ...
                 ↓ AllGather W_in                ↓
GPU 0: W_in[D, F]            GPU 1: W_in[D, F]            ...
                 → forward pass →
                 ↓ ReduceScatter dW_in
GPU 0: dW_in[D/X, F]         GPU 1: dW_in[D/X, F]         ...
```

In PyTorch:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
import torch

mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    MyTransformer(),
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    device_id=local_rank,
)
```

**FSDP2** (the newer implementation in `torch.distributed._composable.fsdp`) wraps individual modules and composes cleanly with tensor parallelism:

```python
from torch.distributed._composable.fsdp import fully_shard

for layer in model.layers:
    fully_shard(layer)
fully_shard(model)
```

**Communication cost:** per layer, FSDP does one AllGather (forward) and one ReduceScatter (backward). Each costs $$V / W$$ for $$V$$ bytes on a ring. Per layer, we transfer $$W_\text{in}[D, F]$$ and $$W_\text{out}[F, D]$$, each of $$2DF$$ bytes:

$$T_\text{comms} = \frac{4DFL}{W}$$

This is identical to DDP's comms cost — the AllGather and ReduceScatter overlap with the compute of the preceding/following layer in practice. The key difference is memory:

**Memory savings:** DDP stores 12+ bytes per parameter per GPU (2 BF16 params + 8 FP32 Adam state). FSDP stores only 12/X bytes per parameter per GPU. For X=128 GPUs, a 70B model requires: DDP needs ~~84 TB~~ (impossible); FSDP needs ~10 GB for params (70e9 × 2 / 128) plus proportional optimizer state.

**Compute-bound condition** (same as DDP):

$$\frac{B}{X} > \frac{4C}{3W}$$

**Takeaway:** FSDP does not increase communication vs. DDP but dramatically reduces memory per GPU. Use it whenever the model doesn't fit on a single GPU.

### ZeRO Stages

PyTorch FSDP corresponds to ZeRO-3 in the DeepSpeed taxonomy:

| Stage | What is sharded | Memory reduction |
|:---|:---|:---|
| **ZeRO-1** | Optimizer states (Adam m/v) only | ~4× |
| **ZeRO-2** | Gradients + optimizer states | ~8× |
| **ZeRO-3** | Parameters + gradients + optimizer states | ~X× (linear in GPU count) |

ZeRO-1 and ZeRO-2 keep parameters replicated, avoiding the AllGather during the forward pass. They are useful when the model fits on one GPU but optimizer state is the bottleneck.

## Tensor Parallelism (TP)

**Idea:** shard individual weight matrices across GPUs along the non-batch dimension. Each GPU computes a partial result, then an AllReduce combines them. This is the Megatron-LM style of parallelism.

For the two-matrix MLP block, split $$W_\text{in}$$ column-wise and $$W_\text{out}$$ row-wise:

```
GPU 0: W_in[D, F/Y]   GPU 1: W_in[D, F/Y]
       ↓                      ↓
GPU 0: tmp[B, F/Y]    GPU 1: tmp[B, F/Y]
       ↓  W_out[F/Y, D]       ↓  W_out[F/Y, D]
GPU 0: partial[B, D]  GPU 1: partial[B, D]
              ↓   AllReduce   ↓
              Out[B, D]
```

All GPUs see the full input activations, each computes a partial output over $$F/Y$$ features, and an AllReduce sums the partial results.

In PyTorch with DTensor (see [Chapter 4](sharding.md)):

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel
)

tp_mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))

parallelize_module(
    model,
    tp_mesh,
    {
        "mlp.w_in":  ColwiseParallel(),   # each GPU holds W_in[D, F/Y]
        "mlp.w_out": RowwiseParallel(),   # each GPU holds W_out[F/Y, D]
    },
)
```

**Communication cost per layer:** one AllReduce of $$Out[B, D]$$, which is $$2BD$$ bytes in BF16:

$$T_\text{comms} = \frac{2BD}{W}$$

Compute per GPU per layer: $$\frac{6BTDF}{Y}$$ (weight is sharded by Y).

**Compute-bound condition** ($$T_\text{math} > T_\text{comms}$$):

$$\frac{6BTDF}{YC} > \frac{2BD}{W} \implies T \cdot F > \frac{YC}{3W}$$

This is **independent of batch size** — TP is limited by the product $$T \cdot F$$ relative to the operational intensity $$C/W$$.

**H100 numbers:**

- **NVLink (W = 900 GB/s):** $$C/W \approx 1,099$$. For 8-way TP (Y=8): need $$T \cdot F > 2,931$$. At T=2048 this is F > 1.4 — always satisfied. TP within a node is essentially free for any realistic transformer.
- **InfiniBand NDR (W = 50 GB/s):** $$C/W \approx 19,780$$. For Y=8: need $$T \cdot F > 52,613$$. At T=2048 this requires F > 26. Still fine numerically, but AllReduce latency (~5–20 µs per call × many layers) accumulates and degrades efficiency badly. **Keep TP within-node.**

**When to use TP:**
- Activation memory is a bottleneck (activations are also sharded in sequence-parallel TP variants)
- The model doesn't fit on a single GPU even with FSDP (e.g., very large layer widths)
- You have NVLink between GPUs (8 GPUs per H100 node)

## Mixed FSDP + Tensor Parallelism

The standard recipe for large model training (e.g., LLaMA 3 70B on 512+ GPUs) combines tensor parallelism within a node (NVLink) and FSDP across nodes (InfiniBand).

Let $$Y$$ = TP degree (typically 8, within-node), $$X$$ = FSDP degree (across nodes).

**Per-layer communication:**
- TP AllReduce (NVLink): $$2BD / W_\text{NVLink}$$
- FSDP AllGather + ReduceScatter (InfiniBand): $$4DF / (X \cdot W_\text{IB})$$

**Compute-bound conditions:**

$$\text{(TP bound)}: \quad T \cdot F > \frac{YC}{3W_\text{NVLink}}$$

$$\text{(FSDP bound)}: \quad \frac{B}{X} > \frac{2C}{3W_\text{IB}}$$

With H100:
- **TP bound** (Y=8, NVLink 900 GB/s): $$T \cdot F > 8 \times 989\text{e12} / (3 \times 900\text{e9}) \approx 2{,}931$$. At T=2048, need F > 1.4. Always satisfied.
- **FSDP bound** (InfiniBand 50 GB/s): $$B/X > 2 \times 989\text{e12} / (3 \times 50\text{e9}) \approx 13{,}187$$ tokens/GPU.

For LLaMA 3 70B training: with 16M tokens/step and X=512 FSDP shards, B/X ≈ 31,250 > 13,187. Compute-bound. ✓

PyTorch 2.0+ supports this via FSDP2 + DTensor TP:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# 2D mesh: TP within each 8-GPU node, FSDP across nodes
mesh = init_device_mesh("cuda", (num_nodes, 8), mesh_dim_names=("dp", "tp"))
tp_mesh = mesh["tp"]
dp_mesh = mesh["dp"]

# Apply tensor parallelism first
for layer in model.layers:
    parallelize_module(layer, tp_mesh, {
        "mlp.w_in":  ColwiseParallel(),
        "mlp.w_out": RowwiseParallel(),
    })

# Then wrap with FSDP on the DP mesh
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

## Pipeline Parallelism (PP)

**Idea:** split model layers across GPUs along the depth dimension. GPU 0 runs layers 1–L/Z, GPU 1 runs layers L/Z+1–2L/Z, etc. Activations flow forward through the pipeline; gradients flow backward.

```
GPU 0: layers 0–7    GPU 1: layers 8–15    GPU 2: layers 16–23   GPU 3: layers 24–31
    → activations → activations → activations →
    ← gradients  ← gradients  ← gradients  ←
```

The key challenge is **pipeline bubbles** — GPUs idle while waiting for previous stages. A naive schedule (one micro-batch at a time) has bubble fraction $$(Z-1)/Z$$. Interleaved schedules (Megatron's 1F1B) achieve bubble fraction $$(Z-1)/(Z + M - 1)$$ where M is the number of micro-batches, approaching 0 as M → ∞.

**Communication per PP step:** only the activation tensor $$[B_\text{micro}, D]$$ passes between adjacent stages — much smaller than TP's AllReduce or FSDP's AllGather.

```python
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

stage = PipelineStage(
    model_chunk,       # this GPU's layers
    stage_index,
    num_stages,
    device,
    input_args=example_input,
)
schedule = ScheduleGPipe(stage, n_microbatches=8)
schedule.step(batch)
```

**When to use PP:**
- Very deep models that can't fit on one GPU even with layer-level FSDP
- Reducing peak activation memory by overlapping forward computation with backward passes

PP adds significant implementation complexity (scheduling, bubble management, checkpoint coordination). For most runs below 1K GPUs with FSDP + TP, PP is unnecessary.

## Putting It Together: 3D Parallelism

Large models in practice combine all three strategies:

| Model size | Typical strategy |
|:---|:---|
| ≤ 4B | DDP only (model fits on 1 GPU) |
| 4B–70B | FSDP across all GPUs |
| 70B–200B | TP=8 (intra-node NVLink) + FSDP across nodes |
| 200B+ | TP=8 + PP across node groups + FSDP within PP groups |

For LLaMA 3 70B (D=8192, F=28672, L=80) on 512 H100s (80 GB each):
- **Memory:** 70B × 18 bytes/param = 1.26 TB total. FSDP shards to ~2.5 GB params/GPU + ~7.5 GB optimizer state/GPU = ~10 GB, leaving ~70 GB for activations.
- **Compute-bound check (FSDP):** B/X = 16M/512 = 31,250 tokens/GPU >> 13,187 threshold. ✓
- **Compute-bound check (TP, Y=8):** T·F = 8192 × 28672 ≈ 235M >> 2,931 threshold. ✓

## Key Takeaways

- **DDP**: simple, effective, works when model fits on one GPU. AllReduce gradients with bucketing. Limited to ~4B params on H100.

- **FSDP (ZeRO-3)**: same communication cost as DDP, dramatically lower per-GPU memory. The standard baseline for models that don't fit on one GPU.

- **Tensor parallelism**: communication scales with activation size ($$BD$$), not weight size. Keep TP within-node (≤ 8 GPUs, NVLink). TP reduces activation memory and allows larger per-GPU batch sizes.

- **Pipeline parallelism**: useful for very deep models or extreme GPU counts. Adds scheduling complexity; use interleaved 1F1B to minimize bubble overhead.

- **The hardware hierarchy dictates strategy**: NVLink (900 GB/s) is 18× faster than InfiniBand NDR (50 GB/s). High-bandwidth operations (TP AllReduce) must stay on NVLink; lower-frequency operations (FSDP AllGather across layers) go on InfiniBand.

[^1]: We model the transformer as a two-matrix MLP and ignore attention FLOPs for this analysis. Attention is typically 10–20% of total FLOPs for D=8192, T=4096, and its communication patterns follow the same logic as TP.
[^2]: For FSDP2 composability with TP, use `fully_shard` from `torch.distributed._composable.fsdp` rather than the legacy `FullyShardedDataParallel` wrapper class. The two APIs are not directly compatible.
