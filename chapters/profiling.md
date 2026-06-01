# Profiling GPU Code

You can't optimize what you can't measure. This chapter covers the two main profiling tools for PyTorch GPU code: the **PyTorch Profiler** (built-in, good for Python-level analysis) and **NVIDIA NSight Systems** (nsys, good for low-level CUDA timeline analysis). The goal is to be able to:

1. Find where training time is actually being spent
2. Identify GPU idle periods and their causes
3. Understand memory allocation patterns
4. Confirm that communication overlaps with compute

## The PyTorch Profiler

The PyTorch Profiler captures CPU and CUDA events, including kernel durations, memory operations, and communication calls. It integrates with TensorBoard and outputs Chrome-compatible traces.

### Basic Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters())

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,      # skip first 1 steps (warmup)
        warmup=1,    # profile warmup steps but discard
        active=3,    # capture 3 steps
        repeat=1,
    ),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # advance the profiler schedule

# Print summary table
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

The `schedule` is important — always skip at least 1 step to avoid profiling the CUDA graph compilation or cuBLAS warm-up. The `active=3` captures 3 steps for averaging.

### Reading the Summary Table

```
-------------------------------------------------------  ------------  ------------  ------------  
                                                   Name    Self CUDA    CUDA total  # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  
                                       aten::mm         4.521s        4.521s            640
                    aten::_scaled_dot_product_flash_attn 892.000ms     892.000ms          80
                                aten::linear_backward    834.000ms     834.000ms         320
                                      ncclAllReduce      421.000ms     421.000ms          32
                                    aten::layer_norm     187.000ms     187.000ms         320
-------------------------------------------------------  ------------  ------------  ------------  
```

Key columns:
- **Self CUDA:** time in this op, not counting called ops
- **CUDA total:** total time including called ops
- **# of Calls:** how many times this op was called

From this table: matmuls take 4.5s, Flash Attention 892ms, communication 421ms. The ratio of communication to compute is 421/(4521+892) ≈ 7.7% — good, should be overlapping.

### TensorBoard Trace Viewer

Open the trace in TensorBoard:

```bash
pip install torch_tb_profiler
tensorboard --logdir=./log/profiler
```

Navigate to the **Trace** tab. The trace view shows:

- **CPU timeline (top rows):** Python code, kernel launches
- **CUDA timeline (bottom rows):** actual GPU kernel execution
- **NCCL communication:** AllReduce/AllGather operations

What to look for:

1. **GPU idle gaps:** white spaces in the CUDA timeline where the GPU is waiting. Common causes:
   - CPU bottleneck (Python loop overhead, data loading)
   - GPU-CPU sync point (`.item()` call, checkpoint save)
   - NCCL collective that isn't overlapping with compute

2. **Kernel durations:** hover over a kernel to see its name, duration, and memory addresses. cuBLAS GEMM kernels should be the largest kernels. If you see many tiny kernels, you have a fusion opportunity.

3. **Memory events:** the memory tab shows allocation/deallocation timelines. OOM errors often show up as a spike in allocation followed by a crash — look for the last successful allocation before the OOM.

### Annotating Code

Add custom labels to understand which parts of your code correspond to which kernels:

```python
with torch.profiler.record_function("attention"):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

with torch.profiler.record_function("mlp_up"):
    x = F.linear(x, self.w_up)
```

These labels appear in the trace viewer and summary table, making it much easier to map kernel durations back to your code.

### Memory Profiling

```python
with profile(profile_memory=True, record_shapes=True) as prof:
    model(input)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
```

For a more detailed view of peak memory:

```python
torch.cuda.reset_peak_memory_stats()
model(input)
print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Peak reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

# Full breakdown
print(torch.cuda.memory_summary(abbreviated=False))
```

If `max_memory_reserved` >> `max_memory_allocated`, you have memory fragmentation — see Chapter 3 for remediation.

## NSight Systems (nsys)

The PyTorch Profiler is great for high-level analysis. For low-level CUDA timeline analysis — understanding kernel occupancy, SM utilization, memory bandwidth, and instruction-level bottlenecks — use NVIDIA NSight Systems.

### Capturing a Trace

```bash
# Profile a training script
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=profile.nsys-rep \
    python train.py

# Open in NSight Systems GUI
nsys-ui profile.nsys-rep
```

For a minimal capture (avoids overhead from tracing every Python call):

```bash
# Only capture the middle of training (skip warmup)
nsys profile \
    --capture-range=cudaProfilerApi \
    --trace=cuda,nvtx \
    python train.py
```

In your script, bracket the region to profile:

```python
import torch.cuda.nvtx as nvtx

# Warmup
for _ in range(5):
    model(input)

# Start capture
torch.cuda.cudaProfilerStart()
nvtx.range_push("training_step")

output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

nvtx.range_pop()
torch.cuda.cudaProfilerStop()
```

### Reading the NSight Systems Timeline

The NSight Systems GUI shows:

- **CUDA HW row:** actual hardware execution — SMs busy, memory transfers
- **CUDA API row:** kernel launches from the CPU side
- **NVTX row:** your custom labels (nvtx.range_push annotations)
- **NVLink row:** NVLink traffic over time (if traced)

Key things to look for:

**SM Utilization:** is the GPU busy? Gaps in the SM Utilization row mean the GPU is idle. Find the kernel immediately before the gap and look at what's causing the stall.

**Memory bandwidth:** the HBM bandwidth gauge shows whether kernels are memory-bound. A kernel at 100% memory bandwidth and 10% compute utilization is memory-bound — optimization should focus on reducing data movement, not adding compute.

**PCIe / NVLink traffic:** the NVLink row shows AllGather/ReduceScatter/AllReduce operations. If they appear on the critical path (GPU idle during the collective), your communication and compute are not overlapping.

### NSight Compute for Kernel Deep-Dive

For deep analysis of a specific kernel, use NSight Compute (ncu):

```bash
ncu --target-processes all \
    --kernel-name "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x3_tn" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
               l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
               dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python script.py
```

This shows for the cuBLAS GEMM kernel:
- SM throughput (compute utilization)
- L1 cache hit rate
- DRAM bandwidth utilization

A GEMM at 80% SM throughput and 80% DRAM bandwidth is well-tuned. A GEMM at 10% SM throughput and 100% DRAM bandwidth is memory-bound — likely needs larger tiles or quantization.

## Common Bottlenecks and Fixes

### GPU Idle Due to CPU Bottleneck

**Symptom:** long gap in CUDA timeline, then many kernels launch in a burst.
**Cause:** Python loop, data loader, or pre-processing is slower than the GPU.
**Fix:** use `num_workers` in DataLoader, pin memory, use `non_blocking=True` for tensor copies.

```python
loader = DataLoader(dataset, num_workers=8, pin_memory=True)
for batch in loader:
    x = batch.to('cuda', non_blocking=True)  # async copy, GPU continues
    output = model(x)
```

### GPU Sync Points (`.item()`, `print`)

**Symptom:** regular short idle periods every N steps.
**Cause:** `.item()` or `print(tensor)` blocks until GPU finishes the computation.
**Fix:** accumulate metrics as CUDA tensors; call `.item()` only at logging intervals.

```python
# Bad: sync every step
for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    print(f"Loss: {loss.item()}")  # GPU sync every step

# Good: sync only every 100 steps
running_loss = torch.zeros(1, device='cuda')
for step, batch in enumerate(loader):
    loss = compute_loss(batch)
    running_loss += loss.detach()
    if step % 100 == 0:
        print(f"Loss: {running_loss.item() / 100}")  # sync once per 100 steps
        running_loss.zero_()
```

### Communication Not Overlapping with Compute

**Symptom:** in the trace, NCCL operations appear after compute finishes (sequential, not overlapping).
**Cause:** DDP bucket size too small, gradient accumulation with no_sync not used, or FSDP communication misconfiguration.
**Fix:** increase DDP bucket size; verify FSDP overlap settings.

```python
# DDP: larger buckets → fewer, larger AllReduces → better overlap
model = DDP(model, bucket_cap_mb=50)  # default is 25 MB

# FSDP2: overlap is automatic, but verify by checking the trace
```

### Memory Fragmentation

**Symptom:** OOM error despite low `memory_allocated()`.
**Cause:** allocated and freed tensors of varying sizes leave unusable gaps.
**Fix:** `torch.cuda.empty_cache()` after large tensor deletion; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # default in PyTorch 2.2+
```

## Quick Profiling Workflow

1. **Coarse profiling** with PyTorch Profiler + TensorBoard: find where time goes (matmuls, attention, communication, idle)
2. **Identify the bottleneck:** is it compute, communication, or CPU?
3. **If compute:** check arithmetic intensity, use `torch.compile`, consider quantization
4. **If communication:** increase batch size (more tokens/step), check bucket sizes, verify overlap
5. **If CPU:** check DataLoader workers, remove sync points, use CUDA Graphs
6. **Fine-grained kernel analysis** with NSight Systems + NSight Compute for the specific bottleneck kernel

## Key Takeaways

- **PyTorch Profiler gives a Python-level view.** Use it first to find which ops consume the most time and whether communication overlaps with compute.

- **NSight Systems gives a CUDA-level view.** Use it when you need to see SM utilization, NVLink traffic, and exact kernel timings beyond what the PyTorch Profiler exposes.

- **GPU idle periods are the most actionable finding.** Find what's causing the GPU to stall — CPU bottleneck, sync point, or non-overlapping communication — and fix that first.

- **`record_function` annotations** make it dramatically easier to map kernel timings back to your code. Add them around every major component.

- **Memory issues** (OOM, fragmentation) are best diagnosed with `torch.cuda.memory_summary()` and the profiler's memory mode, not by guessing at what's large.
