# CUDA Under the Hood

When you call `loss.backward()` in PyTorch, dozens of CUDA kernels launch on your GPU within milliseconds. You don't write any of them. But knowing *what* they are, *who* calls them, and *why* they perform the way they do is essential for reasoning about training speed and debugging bottlenecks.

This chapter builds a mental model of the software stack between PyTorch and the GPU. We won't write CUDA code. The goal is to understand the system well enough to interpret profiler traces, make sense of memory errors, and know which library to blame when something is slow.

## The Stack

When PyTorch runs a tensor operation on GPU, it travels through several layers:

```
Your Python code
      ↓
PyTorch Python API  (torch.matmul, nn.Linear, F.softmax, …)
      ↓
ATen  (C++ tensor library — selects the right kernel)
      ↓
Dispatch layer  (picks CUDA vs CPU vs MPS vs …)
      ↓
┌──────────────────────────────────────────────────────┐
│  CUDA libraries                                      │
│                                                      │
│  cuBLAS   cuDNN   NCCL   cuSPARSE   custom kernels  │
└──────────────────────────────────────────────────────┘
      ↓
CUDA driver + hardware
```

Each layer hands work to the next. Your Python code sees a clean API. The hardware sees a sequence of GPU kernel launches. The layers in between translate one into the other.

## CUDA Kernels: the Basic Model

A **CUDA kernel** is a function that runs in parallel on the GPU. When PyTorch calls cuBLAS to do a matmul, cuBLAS launches one or more kernels. Understanding the launch model helps interpret profiler output.

Every kernel is launched with a **grid** of **thread blocks**, each block containing up to 1024 **threads**. Threads within a block share L1/shared memory and can synchronize with each other. Threads in different blocks cannot directly communicate.

The GPU schedules blocks onto SMs. Each SM runs one or more blocks concurrently until done, then picks up the next. The programmer specifies grid and block dimensions; the GPU decides which SM runs which block.

In practice you never choose these dimensions manually for matmuls — cuBLAS does. For custom kernels (covered in [Chapter 14](triton-kernels)), choosing good tile sizes is critical for performance.

## ATen and the Dispatch System

**ATen** (A Tensor library) is PyTorch's C++ core. It defines every PyTorch operator — `aten::mm`, `aten::add`, `aten::softmax` — and implements them for each device and dtype combination.

When you call `torch.matmul(a, b)` on CUDA tensors, ATen:

1. Determines the operation: `aten::mm` or `aten::bmm` depending on shape
2. Dispatches to the CUDA implementation
3. Calls into cuBLAS (for dense matmuls) or a fused custom kernel

The dispatch happens at C++ speed and adds negligible overhead for large tensors. For very small tensors, Python overhead and kernel launch latency can dominate runtime — this is why batching small ops matters.

You can inspect kernel launches by profiling:

```python
import torch

a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()

print(p.key_averages().table(sort_by='cuda_time_total', row_limit=5))
```

## cuBLAS: Matrix Multiplication

**cuBLAS** is NVIDIA's library for dense linear algebra. It implements GEMM (General Matrix Multiply) and related operations. When PyTorch calls `torch.matmul`, `torch.mm`, `nn.Linear`, or any other operation that bottoms out in a matmul, it almost always calls cuBLAS.

cuBLAS selects the best kernel for your problem size, dtype, and GPU automatically. It picks among dozens of hand-tuned GEMM implementations — different tile sizes, memory access patterns, Tensor Core configurations. On H100 and A100, cuBLAS uses 4th/3rd-generation Tensor Cores for BF16 and FP16.

**Key behaviors to know:**

- **Warm-up:** cuBLAS runs a heuristic search on the first few calls to select the best algorithm. This is why the first training iteration is always slower.
- **TF32:** On Ampere and later GPUs, FP32 matmuls use TF32 precision (10-bit mantissa instead of 23-bit) through Tensor Cores by default. This gives ~8× speedup with minimal precision loss. Disable only if you see numerical instability.
- **Alignment:** cuBLAS achieves peak Tensor Core throughput when matrix dimensions are multiples of 16 (BF16/FP16) or 32 (FP8). PyTorch's vocabulary sizes, hidden dimensions, and batch sizes are usually designed with this in mind.

```python
# TF32 is enabled by default on Ampere+ (PyTorch >= 1.7)
print(torch.backends.cuda.matmul.allow_tf32)  # True

# Disable for strict FP32 precision
torch.backends.cuda.matmul.allow_tf32 = False
```

## cuDNN: Convolutions and Fused Ops

**cuDNN** is NVIDIA's library for deep neural network primitives. It provides highly optimized implementations of:

- **Convolutions** — forward, backward weight, backward input
- **Batch normalization** — fused with relu
- **Attention** — Flash Attention via cuDNN's fused multi-head attention kernel (since cuDNN 8.9)
- **Layer normalization** — fused forward+backward in recent PyTorch builds
- **Softmax** — fused implementations

Like cuBLAS, cuDNN auto-tunes on first use:

```python
# Auto-tune: benchmarks multiple algorithms, picks fastest.
# Good when input shapes are fixed each step.
torch.backends.cudnn.benchmark = True

# Deterministic mode — reproducible results at the cost of speed
torch.backends.cudnn.deterministic = True
```

`benchmark = True` is worthwhile when shapes are fixed (common in LLM training with fixed sequence length and batch size). It runs a brief search on first call and caches the result. If shapes vary every step, the search overhead outweighs the gains.

**Flash Attention** in PyTorch (`F.scaled_dot_product_attention` since PyTorch 2.0) dispatches to cuDNN's fused attention kernel on supported hardware automatically:

```python
# Automatically uses Flash Attention on CUDA (Ampere+)
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

PyTorch selects among backends — cuDNN Flash Attention, a Triton kernel, or a math fallback — depending on head size, dtype, and hardware. You can inspect which backend was chosen:

```python
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    out = F.scaled_dot_product_attention(q, k, v)
```

## NCCL: Collective Communication

**NCCL** (NVIDIA Collective Communications Library) implements the communication operations used in distributed training:

| Operation      | What it does                                              | Use case                          |
|:-------------- |:--------------------------------------------------------- |:--------------------------------- |
| `AllReduce`    | Sum a tensor across all GPUs, give everyone the result    | Gradient sync in DDP              |
| `AllGather`    | Concatenate a shard from each GPU into one full tensor    | FSDP weight reconstruction        |
| `ReduceScatter`| Reduce across GPUs, scatter result shards to each GPU     | FSDP gradient reduction           |
| `Broadcast`    | Send one GPU's tensor to all others                       | Model init, param sync            |
| `Send`/`Recv`  | Point-to-point transfer between two GPUs                  | Pipeline parallelism              |

When you use `torch.distributed` — DDP, FSDP, or manually — it calls NCCL for CUDA tensors.

NCCL automatically uses NVLink when GPUs share a node with NVSwitch. For cross-node traffic it uses InfiniBand or RoCE via GPUDirect RDMA, which lets the NIC read directly from GPU memory without passing through CPU RAM.

**Key behaviors:**

- NCCL operations are **asynchronous** — they return immediately and run on a separate CUDA stream. The actual transfer happens in the background.
- NCCL performance depends heavily on **message size**. Small tensors (< 1 MB) are latency-bound; large tensors saturate the link bandwidth. This is why gradient bucketing in DDP matters — many small per-parameter gradients are grouped into one large all-reduce.

```python
# DDP gradient bucketing — gradients are grouped before all-reduce
model = torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=25  # 25 MB per bucket; larger = better bandwidth utilization
)
```

## Memory Management: the Caching Allocator

CUDA memory allocation (`cudaMalloc`) is slow — it can take milliseconds. PyTorch avoids calling it repeatedly by using a **caching allocator**.

The caching allocator maintains a pool of free GPU memory blocks. When you create a tensor, PyTorch finds a free block of the right size in the pool. When the tensor is freed, the block returns to the pool rather than being returned to the OS.

```python
# Memory held by live tensors
print(torch.cuda.memory_allocated())   # bytes

# Memory held by the caching allocator (including free blocks)
print(torch.cuda.memory_reserved())    # bytes

# Release unused cached blocks back to the OS
torch.cuda.empty_cache()
```

**Fragmentation** is the main failure mode. If you allocate a 10 GB tensor, free it, then try to allocate a 12 GB tensor, the allocator may fail even though `memory_allocated()` shows only 0 GB in use — because the 10 GB block is the wrong size and there is no room for a fresh 12 GB allocation. OOM errors despite low `memory_allocated()` are almost always fragmentation.

**Practical tips:**

- Call `torch.cuda.empty_cache()` after deleting large tensors if you see fragmentation-induced OOMs.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (default in PyTorch 2.2+) reduces fragmentation by growing allocations incrementally.
- Use `torch.cuda.memory_summary()` to get a breakdown of what is using memory before reaching for gradient checkpointing or a smaller batch size.

```python
print(torch.cuda.memory_summary(device=0, abbreviated=True))
```

## Streams and Asynchronous Execution

**CUDA streams** are ordered queues of operations that execute in order on the GPU. Operations on *different* streams can run concurrently if the GPU has capacity.

PyTorch uses multiple streams internally. The default stream runs your forward and backward pass. A separate stream runs NCCL all-reduces. This is how DDP overlaps gradient communication with the backward pass — gradients are reduced on the NCCL stream while the backward pass continues on the compute stream.

```python
# Create a non-default stream
s = torch.cuda.Stream()

with torch.cuda.stream(s):
    y = expensive_op(x)  # runs on stream s, overlaps with default stream

# Ensure default stream waits for s before using y
torch.cuda.current_stream().wait_stream(s)
result = y + 1
```

**CPU-GPU synchronization** is the key gotcha. CUDA operations are asynchronous — the CPU submits work and continues immediately. A sync point blocks the CPU until the GPU catches up. These happen whenever you:

- Call `.item()` on a CUDA tensor
- `print()` a CUDA tensor
- Call `torch.cuda.synchronize()`
- Save a checkpoint or copy a tensor to CPU

Frequent syncs stall the training loop. The most common offender is logging loss every step:

```python
# Slow: forces a GPU sync every step
for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    print(f"step {step}: loss={loss.item()}")  # GPU sync here

# Fast: accumulate on GPU, sync only at log interval
running = torch.zeros(1, device='cuda')
for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    running += loss.detach()
    if step % 100 == 0:
        avg = running.item() / 100   # one sync per 100 steps
        print(f"step {step}: loss={avg:.4f}")
        running.zero_()
```

## CUDA Graphs

**CUDA Graphs** capture a sequence of kernel launches and replay them with a single API call, eliminating the CPU dispatch overhead of submitting each kernel individually. For a typical transformer training step, this can save 5–15% of step time when the GPU is fast enough that CPU overhead becomes significant.

```python
# Warm-up: let cuBLAS/cuDNN select algorithms
for _ in range(3):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Capture the graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()

# Training loop: copy new data into static tensors and replay
for batch_input, batch_target in dataloader:
    static_input.copy_(batch_input)
    static_target.copy_(batch_target)
    g.replay()           # fast: no Python/CPU dispatch
    optimizer.step()     # optimizer step runs outside the graph
    optimizer.zero_grad()
```

`torch.compile` (covered in [Chapter 15](torch-compile)) applies CUDA Graphs automatically via `mode="reduce-overhead"` without any manual graph capture code. This is the recommended approach for most use cases.

## What PyTorch Does Not Control

Some behaviors live in the driver or hardware layer and PyTorch cannot change them:

- **Clock throttling:** The GPU boosts its clock based on thermal headroom. A hot GPU throttles by 10–30%, causing step time to drift upward over long runs. Monitor GPU temperature (`nvidia-smi dmon`) if you see unexplained slowdowns.
- **PCIe data loading:** If the dataloader is CPU-bound or the CPU→GPU transfer is a bottleneck, the GPU sits idle waiting. Use `pin_memory=True` in the DataLoader and `non_blocking=True` on `.to(device)` to overlap transfer with compute.
- **Driver kernel launch overhead:** Each CUDA kernel launch costs ~2–10 µs on the CPU side. Models with thousands of tiny ops accumulate this overhead. `torch.compile` and CUDA Graphs both address this.

```python
# Overlap data loading with GPU compute
loader = DataLoader(dataset, pin_memory=True, num_workers=4)

for batch in loader:
    # Async transfer — CPU continues while GPU receives data
    x = batch.to('cuda', non_blocking=True)
    out = model(x)
```

## Key Takeaways

- **PyTorch → ATen → cuBLAS / cuDNN / NCCL → CUDA driver.** Each layer is an abstraction. Understanding what lives where tells you what to profile and what to tune.

- **cuBLAS runs your matmuls.** It auto-selects algorithms and uses Tensor Cores for BF16/FP16. TF32 is enabled by default. Matrix dimensions should be multiples of 16 for peak throughput.

- **cuDNN runs your attention.** `F.scaled_dot_product_attention` dispatches to Flash Attention automatically on Ampere+. No manual implementation needed.

- **NCCL runs your collectives.** All-reduces are async on a separate stream. DDP buckets gradients to maximize bandwidth utilization. Cross-node traffic over InfiniBand is the dominant bottleneck for large-scale training.

- **The caching allocator hides `cudaMalloc`.** OOM errors with low `memory_allocated()` are fragmentation. `empty_cache()` and `expandable_segments` help.

- **CUDA is asynchronous.** Minimize sync points (`.item()`, `print`) in the hot path. Accumulate metrics as tensors and sync only when logging.

- **`torch.compile` eliminates CPU overhead** via automatic CUDA Graphs and kernel fusion. Use it.

[^1]: The full PyTorch dispatch path also includes autograd (which wraps ops in backward functions) and TorchDynamo (when `torch.compile` is active, which rewrites Python bytecode before dispatch). The path shown here is eager mode without compilation.
[^2]: NCCL also respects `NCCL_DEBUG=INFO` for verbose logging, useful when debugging slow collectives or InfiniBand configuration. `NCCL_SOCKET_IFNAME` controls which network interface NCCL uses for out-of-band coordination.
[^3]: `pin_memory` allocates CPU tensors in page-locked memory, which the GPU DMA engine can access directly without staging. This roughly doubles CPU→GPU transfer bandwidth for large tensors at the cost of higher CPU memory usage.
[^4]: GPUDirect RDMA allows the InfiniBand NIC to read directly from GPU memory, skipping a copy through CPU RAM. This requires both GPUDirect-capable hardware and the `nv_peer_mem` or `nvidia_peermem` kernel module. Many cloud providers configure this automatically for HPC instances.
