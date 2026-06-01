# How to Think About GPUs

**An NVIDIA GPU is a massively parallel processor built around a large number of identical compute units called Streaming Multiprocessors (SMs), each backed by a stack of fast memory (HBM) and a hierarchy of on-chip caches.** Understanding how these pieces fit together — and where the bottlenecks are — is the key to reasoning about GPU performance.

This chapter builds on the roofline framework from [Chapter 1](roofline). There, we asked whether a computation was compute-bound or memory-bound. Here we make that precise: what is the compute, and what is the memory bandwidth?

## The Streaming Multiprocessor

If you zoom into an H100 GPU, you'll find 132 **Streaming Multiprocessors (SMs)**. You can think of each SM as a mini-GPU in its own right: it has its own scheduler, its own arithmetic units, its own fast local memory, and its own Tensor Cores. The full GPU is essentially 132 of these running in parallel.

### CUDA cores and warps

Each SM runs threads in groups of 32 called **warps**. All 32 threads in a warp execute the same instruction simultaneously — this is the GPU's SIMD model (Single Instruction, Multiple Data). If threads in a warp take different branches (e.g., an if/else), the warp must execute both branches and mask off the inactive threads. This is called **warp divergence** and wastes cycles.

Each SM on an H100 has:
- **4 warp schedulers**, each capable of issuing one instruction per cycle to one warp
- **128 FP32 CUDA cores** (32 per scheduler group), for scalar floating-point and integer ops
- **4 Tensor Core units** (one per scheduler group), for matrix multiply-accumulate
- **256 KB of shared memory / L1 cache** per SM, configurable
- **256 KB of register file** per SM

When PyTorch launches a CUDA kernel, it specifies a grid of thread blocks. Each thread block is assigned to one SM for its lifetime. The SM runs multiple warps from that block concurrently, hiding memory latency by switching to a ready warp while another waits for data.

:::{note}
**Key mental model:** An SM is not like a CPU core with deep out-of-order execution. It hides latency by *breadth* — running many warps at once — not by predicting what comes next. This is why GPUs need many parallel threads to achieve high utilization.
:::

### Tensor Cores

**Tensor Cores are the real reason GPUs dominate LLM training.** They perform a small matrix multiply-accumulate in a single operation: `D = A @ B + C` where A, B are small matrices in a low-precision format (BF16, FP16, FP8) and C, D are in FP32. This is exactly the inner loop of a larger matmul.

Each 4th-generation Tensor Core in an H100 operates on a `16×8×16` (M×K×N) matrix fragment per cycle, delivering roughly 2× the throughput of A100's 3rd-generation Tensor Cores. The practical implication: if your computation is a big matmul with enough parallelism, you can use almost all of the GPU's 989 BF16 TFLOPS. If it's something else — a reduction, a softmax, an elementwise add — the Tensor Cores sit idle and you're limited to the 67 TFLOPS of the CUDA cores.

This asymmetry is why transformer architectures dominate: they are almost entirely matmuls.

**Alignment requirements:** Tensor Cores require matrix dimensions to be multiples of 16 (for BF16/FP16) or 32 (for FP8) to achieve peak throughput. Padding to these sizes is handled automatically by libraries like cuBLAS, but it's worth knowing when debugging performance.

## The Memory Hierarchy

The GPU memory system has four levels, each trading capacity for bandwidth:

```
Registers      →  fastest, ~19 TB/s, ~256 KB per SM, private to each thread
L1/Shared Mem  →  ~19 TB/s, 256 KB per SM, shared within a thread block
L2 Cache       →  ~10 TB/s, 50 MB total, shared across all SMs
HBM            →  3.35 TB/s, 80 GB total, shared across all SMs
```

*H100 SXM5 numbers. Throughput figures are aggregate across the full chip.*

### HBM (High Bandwidth Memory)

HBM is the GPU's main memory — the equivalent of RAM. It stores model weights, activations, gradients, and optimizer states. The H100 SXM5 has **80 GB of HBM3** with **3.35 TB/s** of bandwidth.

Contrast this with the Tensor Core peak: 989 TFLOPS of BF16 compute. If every operation reads two BF16 inputs (4 bytes) and produces one BF16 output, a compute-bound operation does at minimum `989e12 / (4 bytes) = 247 TB/s` of "effective compute throughput." The HBM bandwidth of 3.35 TB/s is about **74× slower** than this. This is why the roofline arithmetic intensity number matters so much: only operations with high arithmetic intensity (like big matmuls) can keep the Tensor Cores fed from HBM.

This translates to the same critical intensity calculation from [Chapter 1](roofline): for a BF16 matmul on H100, the crossover point is roughly:

$$\text{Intensity}_\text{H100} = \frac{\text{TFLOPS}}{\text{HBM BW}} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ FLOPs/byte}$$

A BF16 weight is 2 bytes, so this means roughly **B > 148 tokens** per-replica batch size to be compute-bound — similar to, but a bit tighter than, the TPU's ~240 number.

### L1 cache and shared memory

Each SM has **256 KB** that can be split between L1 cache (automatically managed by the hardware) and shared memory (manually managed by the programmer). Shared memory is addressable from CUDA kernels — it's similar to VMEM on a TPU in that the programmer explicitly controls what goes in and out.

Shared memory is critical for algorithms that reuse data many times. **Flash Attention** (covered in [Chapter 14](triton-kernels)) is the canonical example: by keeping the query, key, and value tiles in shared memory across the softmax computation, it avoids writing intermediate attention scores back to HBM — a 10× reduction in memory bandwidth for attention.

The aggregate shared memory bandwidth across all 132 SMs is enormous (order of 19 TB/s) compared to HBM. The challenge is that shared memory is small (256 KB per SM, ~34 MB total), so algorithms must carefully tile their computations to fit.

### L2 cache

The L2 cache is 50 MB on H100 (up from 40 MB on A100) and is shared across all SMs. For small models or repeated access patterns, the L2 can meaningfully reduce effective HBM traffic. For LLM training workloads with large weight matrices, it is mostly a miss.

### Registers

Register files are the fastest storage on the GPU — on-chip, close to the ALUs, with no latency. Each thread gets its own private registers. The SM has 256 KB of register file, shared across all concurrently resident threads. If a kernel uses too many registers per thread, fewer warps can run simultaneously (this is called **register pressure**), hurting the SM's ability to hide latency.

## GPU Networking

A single GPU is fast but not enough for frontier model training. The challenge is connecting many GPUs efficiently.

### NVLink: intra-node

**NVLink** is NVIDIA's high-bandwidth, low-latency interconnect for connecting GPUs within a single node. The H100 SXM5 has **18 NVLink 4.0 lanes**, delivering **900 GB/s bidirectional** total bandwidth between GPUs. Compare this to the **3.35 TB/s** HBM bandwidth: NVLink is about 4× slower than HBM but still fast enough to sustain large all-reduce operations for gradient synchronization.

In a standard **DGX H100** server, 8 GPUs are connected through an **NVSwitch** — a dedicated switching chip that gives each GPU a fully non-blocking, point-to-point 900 GB/s path to every other GPU. This is significantly better than a ring topology, which would limit each GPU to sharing bandwidth with its two neighbors.

:::{note}
**NVLink vs PCIe:** GPUs without NVLink (like the PCIe variant of H100) communicate through the CPU's PCIe bus at 64–128 GB/s — about 7–14× slower than NVLink. For multi-GPU training, the PCIe bottleneck is severe enough that NVLink nodes are strongly preferred.
:::

Here are the bandwidth numbers for H100 and A100:

| Link          | H100 SXM5      | A100 SXM4      |
|:------------- |:-------------- |:-------------- |
| HBM BW        | 3.35 TB/s      | 2.0 TB/s       |
| NVLink BW     | 900 GB/s bidi  | 600 GB/s bidi  |
| PCIe BW       | 128 GB/s       | 64 GB/s        |

### InfiniBand and Ethernet: inter-node

Once you go beyond a single node, you need a network fabric. The two options are:

**InfiniBand (IB):** The standard for HPC and large-scale ML training. NDR InfiniBand provides **400 Gbps (50 GB/s)** per port. A DGX H100 node typically has 8 InfiniBand ports (one per GPU), giving **400 GB/s per node** of inter-node bandwidth. HDR InfiniBand (the previous generation) is 200 Gbps per port.

**RDMA over Converged Ethernet (RoCE):** Cheaper than IB but similar bandwidth, used when cost is a priority over latency. Used by many cloud providers.

The key ratio to keep in mind for multi-node training:

```
HBM bandwidth:   3.35 TB/s per GPU
NVLink BW:         900 GB/s per GPU    (~4× slower than HBM)
InfiniBand BW:      50 GB/s per GPU    (~67× slower than HBM)
```

This hierarchy directly shapes parallelism strategy: operations requiring frequent all-reduce (like gradient synchronization) should either be done within a node over NVLink, or kept infrequent enough that InfiniBand latency doesn't dominate.

## Full Spec Tables

Here are the numbers we'll use throughout the book. Commit the H100 SXM5 column to memory — it's our primary reference hardware.

### Compute and memory

| Spec                    | H100 SXM5      | H100 PCIe      | A100 SXM4      |
|:----------------------- |:-------------- |:-------------- |:-------------- |
| Architecture            | Hopper         | Hopper         | Ampere         |
| Streaming Multiprocessors | 132           | 114            | 108            |
| BF16 TFLOPS (Tensor Core) | 989          | 756            | 312            |
| BF16 TFLOPS (w/ sparsity) | 1,979        | 1,513          | 624            |
| FP8 TFLOPS (Tensor Core)  | 3,958        | 3,026          | —              |
| FP32 TFLOPS (CUDA cores)  | 67           | 51             | 19.5           |
| HBM capacity            | 80 GB HBM3     | 80 GB HBM2e    | 80 GB HBM2e    |
| HBM bandwidth           | 3.35 TB/s      | 2.0 TB/s       | 2.0 TB/s       |
| L2 cache                | 50 MB          | 50 MB          | 40 MB          |
| Shared mem / SM         | 256 KB         | 256 KB         | 164 KB         |
| TDP                     | 700 W          | 350 W          | 400 W          |

### Interconnect

| Spec                    | H100 SXM5      | H100 PCIe      | A100 SXM4      |
|:----------------------- |:-------------- |:-------------- |:-------------- |
| NVLink version          | NVLink 4.0     | NVLink 4.0     | NVLink 3.0     |
| NVLink BW (bidi)        | 900 GB/s       | 600 GB/s       | 600 GB/s       |
| PCIe version            | PCIe 5.0       | PCIe 5.0       | PCIe 4.0       |
| PCIe BW (bidi)          | 128 GB/s       | 128 GB/s       | 64 GB/s        |

### Multi-GPU node configs

| System               | GPUs | GPU type    | Intra-node fabric    | Per-GPU NVLink BW |
|:-------------------- |:---- |:----------- |:-------------------- |:----------------- |
| DGX H100             | 8    | H100 SXM5   | NVSwitch 3.0         | 900 GB/s bidi     |
| DGX A100             | 8    | A100 SXM4   | NVSwitch 2.0         | 600 GB/s bidi     |
| HGX H100             | 8    | H100 SXM5   | NVSwitch 3.0         | 900 GB/s bidi     |
| GB200 NVL72          | 72   | B200 + GB200 | NVLink 5.0 Switch   | 1,800 GB/s bidi   |

## Worked Problems

These numbers come alive when you use them to estimate real workloads. Let's work through a few.

**Question 1 [bounding inference latency]:** You want to run inference on a LLaMA 3 70B model in BF16, loaded across 8× H100 SXM5 GPUs. Each decode step generates one new token. How long must each decode step take, at minimum?

:::{dropdown} Answer
Each decode step loads all model weights once (ignoring KV cache for now). Total weight size: `70e9 params × 2 bytes/param = 140 GB`. Split across 8 GPUs: `17.5 GB per GPU`. Each GPU has 3.35 TB/s HBM bandwidth, so loading takes:

$$T_\text{min} = \frac{17.5 \times 10^9}{3.35 \times 10^{12}} \approx 5.2 \text{ ms per GPU}$$

Since the GPUs work in parallel, the minimum decode latency is about **5 ms per token**. In practice you'll see 8–15 ms depending on KV cache size and batch size. Smaller models or INT8 quantization roughly halve this.
:::

**Question 2 [NVLink vs InfiniBand for all-reduce]:** After a gradient step, you need to all-reduce 70B BF16 gradients across 8 GPUs. (a) How long does this take over NVLink within a DGX H100 node? (b) How long if those 8 GPUs were spread across 8 nodes connected by NDR InfiniBand at 50 GB/s per GPU?

:::{dropdown} Answer
A ring all-reduce transfers `2 × (N-1)/N × data` bytes per GPU, which approaches `2 × data` for large N. For 70B params in BF16: `2 × 70e9 × 2 = 280 GB` total bytes per GPU.

**(a) NVLink (900 GB/s bidi per GPU):**
$$T_\text{NVLink} = \frac{280 \times 10^9}{900 \times 10^9} \approx 0.31 \text{ s}$$

**(b) InfiniBand (50 GB/s per GPU):**
$$T_\text{IB} = \frac{280 \times 10^9}{50 \times 10^9} \approx 5.6 \text{ s}$$

This is an 18× difference. In practice, gradient synchronization is overlapped with the backward pass computation, but the InfiniBand case is still a serious bottleneck for large models. This is why gradient compression, ZeRO (covered in [Chapter 8](deepspeed-megatron)), and careful parallelism strategies matter.
:::

**Question 3 [matmul roofline]:** You're running a single BF16 linear layer: `y = x @ W` where `x` has shape `[B, 4096]` and `W` has shape `[4096, 16384]`, on one H100 SXM5. For what batch sizes B is this compute-bound vs memory-bound?

:::{dropdown} Answer
**FLOPs:** `2 × B × 4096 × 16384 = 1.34e8 × B`

**Bytes loaded from HBM:**
- W: `4096 × 16384 × 2 = 1.34e8` bytes (read once, independent of B)
- x: `B × 4096 × 2 = 8.19e3 × B` bytes
- y: `B × 16384 × 2 = 3.28e4 × B` bytes (write)

Total bytes ≈ `1.34e8 + 4.1e4 × B`

**Compute time:** `1.34e8 × B / 989e12 = 1.36e-13 × B` seconds

**Memory time:** `(1.34e8 + 4.1e4 × B) / 3.35e12` seconds

Setting compute time > memory time:
$$1.36 \times 10^{-13} \times B > \frac{1.34 \times 10^8 + 4.1 \times 10^4 \times B}{3.35 \times 10^{12}}$$

$$B \times (1.36 \times 10^{-13} \times 3.35 \times 10^{12} - 4.1 \times 10^{4}) > 1.34 \times 10^8$$

$$B \times (4.56 \times 10^{-1} - 4.1 \times 10^4) \approx B \times 4.1 \times 10^4 > ... $$

Let's simplify by noting at large B the weight term dominates: $1.34 \times 10^8 / (1.36 \times 10^{-13} \times 3.35 \times 10^{12} - 4.1 \times 10^4) \approx 1.34 \times 10^8 / 4.16 \times 10^2 \approx 320$.

So roughly **B > 320 tokens** to be compute-bound. At B = 1 (typical for single-user inference), you're deeply memory-bound and using less than 0.3% of the GPU's peak TFLOPS. This is the core challenge of inference: the GPU's Tensor Cores sit mostly idle.
:::

**Question 4 [node bandwidth budget]:** A training job runs data parallel across 2 DGX H100 nodes (16 GPUs total) with NDR InfiniBand. The forward+backward pass for a batch takes 800 ms on 16 GPUs. The gradient all-reduce requires transferring 7 GB of BF16 gradients per GPU (a small model). Is the gradient sync a bottleneck?

:::{dropdown} Answer
The all-reduce traffic per GPU: `2 × 7 GB = 14 GB` (ring all-reduce approximation).

InfiniBand per GPU: 50 GB/s.

$$T_\text{sync} = \frac{14 \times 10^9}{50 \times 10^9} = 0.28 \text{ s}$$

The compute step takes 800 ms and the sync takes 280 ms. If they can be overlapped (gradient sync during the backward pass), the effective overhead is the non-overlappable tail — typically 10–20% of the sync time. If not overlapped, the sync adds 35% to the step time. This is why PyTorch DDP overlaps gradient all-reduces with the backward pass by default.
:::

**Question 5 [arithmetic intensity of attention]:** Consider the attention score computation for one head: `S = Q @ K^T` where `Q` has shape `[B, T, D_h]` and `K` has shape `[B, T, D_h]`, with `B=1`, `T=2048`, `D_h=128`, in BF16.

1. What is the arithmetic intensity of this operation (FLOPs / bytes loaded from HBM)?
2. Is it compute-bound on H100?

:::{dropdown} Answer
**FLOPs:** `2 × T × T × D_h = 2 × 2048 × 2048 × 128 = 1.07e9`

**Bytes loaded:**
- Q: `T × D_h × 2 = 2048 × 128 × 2 = 524 KB`
- K: same, `524 KB`
- S (output): `T × T × 2 = 2048 × 2048 × 2 = 8 MB`

Total HBM bytes: `0.524 + 0.524 + 8 ≈ 9.05 MB = 9.05e6 bytes`

**Arithmetic intensity:**
$$\text{Intensity} = \frac{1.07 \times 10^9}{9.05 \times 10^6} \approx 118 \text{ FLOPs/byte}$$

H100 critical intensity = 989e12 / 3.35e12 ≈ **295 FLOPs/byte**.

At 118 FLOPs/byte, attention score computation is **memory-bound** on H100. The output matrix `S` dominates the byte count. Flash Attention avoids materializing `S` in HBM by computing attention tile-by-tile in shared memory, dramatically improving effective arithmetic intensity. This is why flash attention yields such large speedups for long sequences.
:::

## Key Takeaways

- **The SM is the unit of parallelism.** H100 has 132 SMs; each has 4 warp schedulers, 128 CUDA cores, and 4 Tensor Cores. Peak throughput requires saturating all of them.

- **Tensor Cores are everything.** The ratio between BF16 Tensor Core TFLOPS (989) and FP32 CUDA core TFLOPS (67) is ~15×. Nearly all training performance comes from Tensor Cores; nearly all Tensor Core performance comes from matmuls.

- **The memory hierarchy sets the roofline.** HBM at 3.35 TB/s and 295 FLOPs/byte critical intensity means only large-batch matmuls are compute-bound. Attention, elementwise ops, and small-batch inference are memory-bound.

- **NVLink is 4× slower than HBM; InfiniBand is 67× slower.** Parallelism strategies that require frequent inter-GPU communication over slow links will be bottlenecked. Tensor parallelism (within-node) works because NVLink is fast; pipeline parallelism (across nodes) works because it minimizes cross-node traffic.

- **Shared memory is the programmer's lever.** Algorithms like Flash Attention improve performance by keeping data in shared memory instead of spilling to HBM. This is the core idea behind Triton kernel optimization ([Chapter 14](triton-kernels)).

## Appendix: Warp Scheduling and Occupancy

**Warp scheduling** is how the SM hides latency. When a warp stalls waiting for HBM data (which can take hundreds of cycles), the warp scheduler switches to another ready warp with no penalty. This is called **latency hiding**. To hide latency effectively, you need enough resident warps per SM to keep the schedulers busy.

The number of warps that can reside on an SM at once is limited by:
- **Register file:** Each SM has 256 KB. If a warp uses 64 registers/thread × 32 threads = 2048 registers, you can fit at most `256×1024/4 / 2048 = 32` warps on one SM.
- **Shared memory:** If your kernel uses 64 KB of shared memory per block and each block has 2 warps, you can fit `256 KB / 64 KB = 4` blocks, or 8 warps.
- **Maximum warps per SM:** 64 on H100 (hardware limit).

**Occupancy** is the ratio of actual resident warps to the hardware maximum: `actual / 64`. 100% occupancy is rarely necessary for good performance — usually 25–50% is enough to hide latency if the warps issue memory requests at a reasonable rate.

The CUDA profiler (NSight Systems) reports occupancy and the limiting factor. Low occupancy is often a symptom, not a root cause — the root cause is usually high register pressure or large shared memory usage.

**Warp divergence** occurs when threads within a warp take different code paths (e.g., `if (threadIdx.x < 16)`). The SM executes both paths sequentially, masking off inactive threads each time. For ML kernels, divergence is rare because the same operation is applied to all elements, but it can appear in custom reduction kernels or sparse operations.

## Appendix: The Tensor Core Operation

A 4th-generation Tensor Core (H100) performs a matrix multiply-accumulate (MMA) of the form:

```
D[M, N] = A[M, K] @ B[K, N] + C[M, N]
```

where the tile sizes depend on the precision:
- BF16/FP16: `M=16, N=8, K=16` per Tensor Core per cycle
- FP8: `M=16, N=8, K=32` per Tensor Core per cycle (2× throughput vs BF16)
- INT8: same tile as FP8

A and B are in low precision; C and D accumulate in FP32. In PyTorch, `torch.matmul` on BF16 tensors uses Tensor Cores automatically via cuBLAS. The `torch.backends.cuda.matmul.allow_tf32 = True` flag (on by default since PyTorch 1.7) further allows FP32 matmuls to use Tensor Cores with TF32 precision, giving ~8× speedup at the cost of slightly reduced precision.

The practical takeaway: **large square-ish matmuls with dimensions that are multiples of 16 will hit peak Tensor Core throughput.** Odd-shaped matmuls, small K dimensions, or non-multiple dimensions will underutilize the Tensor Cores.

[^1]: The H100 SXM5 achieves 989 TFLOPS BF16 at a boost clock of ~1.98 GHz. The PCIe variant clocks lower and has fewer SMs, giving 756 TFLOPS. All TFLOPS figures in this book are for the SXM5 variant unless otherwise noted.
[^2]: NVSwitch 3.0 in the DGX H100 provides 57.6 TB/s of total switching capacity for 8 GPUs. Each GPU sees 900 GB/s of bidirectional bandwidth to every other GPU simultaneously — it is fully non-blocking.
[^3]: InfiniBand NDR bandwidth per port is 400 Gbps (50 GB/s). A DGX H100 has 8 InfiniBand ports (one per GPU), so the total inter-node bandwidth per node is 400 GB/s bidirectional. In practice, effective bandwidth after protocol overhead is 80–90% of theoretical.
[^4]: The H100 PCIe variant does not have NVSwitch; its 8-GPU configs use NVLink directly between GPUs in a ring/mesh, giving lower effective bandwidth per pair than the SXM5 NVSwitch configuration.
