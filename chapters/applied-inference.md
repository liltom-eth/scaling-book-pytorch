# Applied Inference: Serving LLaMA 3 70B

This chapter works through the hardware math for serving LLaMA 3 70B on H100 GPUs. Try each question before looking at the answer. The goal is to be able to design a serving configuration and estimate throughput and latency before provisioning any hardware.

## Model Specs

| Hyperparameter | LLaMA 3 70B |
|:---|:---:|
| D (d\_model) | 8,192 |
| F (d\_ff) | 28,672 |
| L (layers) | 80 |
| N (Q heads) | 64 |
| K (KV heads, GQA) | 8 |
| H (head dim) | 128 |
| V (vocab size) | 128,256 |

Hardware reference: **H100 SXM5**
- BF16: 989 TFLOP/s, FP8: 1,979 TFLOP/s
- HBM: 3.35 TB/s bandwidth, 80 GB capacity
- NVLink: 900 GB/s bidirectional (within-node)

---

## Part 1: Memory and Hardware Sizing

### Question 1: KV cache size

How large is the KV cache per token? Per sequence at 8k context? At what batch size does the KV cache consume more memory than the model weights (in BF16)?

:::dropdown Answer

KV cache per token (BF16, both keys and values):

$$2 \times K \times H \times L \times 2 \text{ bytes} = 2 \times 8 \times 128 \times 80 \times 2 = 327,680 \text{ bytes} \approx 320 \text{ KB/token}$$

Per sequence at S=8192 tokens:

$$320 \text{ KB} \times 8192 = 2.62 \text{ GB/sequence}$$

Model weights in BF16: $$70 \times 10^9 \times 2 = 140 \text{ GB}$$

KV cache exceeds model weight memory when:

$$B \times 2.62 \text{ GB} > 140 \text{ GB} \implies B > 53 \text{ sequences}$$

At B=53 concurrent sequences at 8k context, the KV cache equals model memory. For B=100, the KV cache is 262 GB — nearly 2× model weight memory. **KV cache management is as important as model weight memory for inference.**

:::

### Question 2: Minimum GPU count

What is the minimum number of H100s to serve LLaMA 3 70B with:
(a) BF16 weights
(b) FP8 weights
(c) INT4 weights (AWQ/GPTQ)

*Assume a small batch size (B=4) with 8k context. Count only parameter memory.*

:::dropdown Answer

| Config | Weight memory | GPUs needed (80 GB each) | Min topology |
|:---|:---:|:---:|:---|
| BF16 | 140 GB | 2 | 2× H100 |
| FP8 | 70 GB | 1 | 1× H100 |
| INT4 (W4A16) | 35 GB | 1 | 1× H100 (plenty of headroom) |

FP8 is the sweet spot for H100: same number of parameters, half the memory, and 2× the FLOP/s vs BF16. A single 80 GB H100 can serve LLaMA 3 70B with 45 GB remaining for KV cache headroom (≈17 concurrent sequences at 8k).

INT4 leaves 45 GB for KV cache after weights + some overhead — similar to FP8 despite lower precision.

:::

### Question 3: Serving configuration for B=64 at 8k context

What is the minimum number of H100s to serve LLaMA 3 70B at B=64 concurrent sequences with 8k context, in BF16? In INT8?

:::dropdown Answer

**BF16:**
- Model: 140 GB
- KV cache: 64 × 2.62 GB = 167.7 GB
- Total: 307.7 GB
- GPUs needed: $$\lceil 307.7 / 80 \rceil = 4$$ H100s

**INT8 weights + INT8 KV cache:**
- Model: 70 GB
- KV cache: 64 × 1.31 GB (INT8 halves KV size) = 83.8 GB
- Total: 153.8 GB
- GPUs needed: $$\lceil 153.8 / 80 \rceil = 2$$ H100s

Quantization reduces GPU count from 4 to 2 for this serving configuration — halving hardware costs.

:::

---

## Part 2: Decode Latency and Throughput

### Question 4: Decode step time (single GPU, memory-bound)

For LLaMA 3 70B in FP8 on a single H100 (70 GB weights), serving B=8 concurrent sequences at 4k context, estimate the decode step time. Is this compute-bound or memory-bound?

:::dropdown Answer

**Critical batch size** (FP8 weights, BF16 activations):

$$B_\text{crit} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \times \frac{8}{16} = 295 \times 0.5 = 147.5 \approx 148 \text{ tokens}$$

B=8 << 148: **memory-bound**.

Bytes to load per step (weights + KV cache):
- FP8 weights: 70 GB
- KV cache (BF16, 4k context, 8 sequences): 8 × 320 KB × 4096 = 10.5 GB
- Total: 80.5 GB

Step time:

$$T = \frac{80.5 \text{ GB}}{3.35 \text{ TB/s}} \approx 24 \text{ ms}$$

**Throughput:**
$$\frac{8 \text{ tokens}}{0.024 \text{ s}} \approx 333 \text{ tok/s}$$

:::

### Question 5: Throughput scaling with batch size

Using the memory-bound model from Question 4, how does throughput scale with batch size B (up to B_crit)? Fill in the table.

:::dropdown Answer

At B concurrent requests, bytes loaded per step (FP8 weights constant, KV cache grows):

$$\text{Bytes} = 70 \text{ GB} + B \times 320 \text{ KB} \times 4096 \text{ tokens}$$

| B | KV cache | Total bytes | Step time | Throughput |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.3 GB | 71.3 GB | 21.3 ms | 47 tok/s |
| 8 | 10.5 GB | 80.5 GB | 24 ms | 333 tok/s |
| 32 | 41.9 GB | 111.9 GB | 33 ms | 970 tok/s |
| 64 | 83.9 GB | 153.9 GB | 46 ms | 1,391 tok/s |
| 100 | 131 GB | 201 GB | 60 ms | 1,667 tok/s |
| 148 (B_crit) | 193.5 GB | 263.5 GB | 79 ms | 1,873 tok/s |

The single H100 only has 80 GB HBM — KV cache at B=32 already requires 41.9 GB, leaving only 38 GB for models and other overhead. In practice, FP8 LLaMA 3 70B on a single H100 can serve B≈10-15 at 4k context before running out of KV cache space.

For B=100, you need 4 H100s total (or INT8 on 2 H100s). More GPUs allow larger batch sizes → higher throughput per GPU.

:::

### Question 6: Compute-bound serving

At what batch size does single-H100 serving (FP8 LLaMA 3 70B, 4k context) become compute-bound?

:::dropdown Answer

At B_crit = 148 tokens, compute and memory times are equal:

$$T_\text{compute} = T_\text{memory} = 79 \text{ ms}$$

$$T_\text{compute} = \frac{6 \times 148 \times 70 \times 10^9}{1979 \times 10^{12}} \approx 31 \text{ ms}$$

Wait — that's 31 ms compute vs 79 ms memory at B=148. Let me recalculate B_crit for FP8:

With FP8 FLOPs (1,979 TFLOP/s) and FP8 weights (1 byte/param):

$$B_\text{crit} = \frac{1979 \times 10^{12}}{3.35 \times 10^{12}} = 591 \text{ tokens}$$

For compute-bound serving with FP8 weights AND FP8 FLOPs, you'd need B>591 — requiring multi-GPU setup just to fit the KV cache.

The practical limit: **with FP8, compute-bound serving on H100 requires B>591 concurrent requests**, which would need ~197 GB of KV cache at 4k context (far beyond one GPU). This confirms that **decode is almost always memory-bound** even on H100 with FP8.

:::

---

## Part 3: Prefill Analysis

### Question 7: Prefill latency at 2k prompt

Estimate prefill latency for a single 2k-token prompt on 2 H100s (TP=2, BF16 LLaMA 3 70B).

:::dropdown Answer

**Is prefill compute-bound?** B_crit (BF16) = 295 tokens. A 2k prompt has B=2000 >> 295. Yes, compute-bound.

**Compute time** (with TP=2, compute is split across 2 GPUs):

$$T_\text{prefill} = \frac{6 \times 2000 \times 70 \times 10^9}{2 \times 989 \times 10^{12}} \approx 424 \text{ ms}$$

Plus TP AllReduce overhead (1 per layer, 2000 × 8192 × 2 bytes = 32.8 MB per AllReduce at 900 GB/s NVLink):
$$80 \text{ layers} \times \frac{32.8 \text{ MB}}{900 \text{ GB/s}} \approx 2.9 \text{ ms}$$

Total prefill latency ≈ **427 ms** (dominated by compute).

TTFT for this request: ~427 ms. This is the time-to-first-token for a 2k prompt. For real-time chat, this is borderline (users notice latencies > 200 ms). Splitting across more GPUs or using FP8 would halve the prefill time.

:::

### Question 8: Optimal serving configuration

You want to serve LLaMA 3 70B to 1,000 concurrent users. Each user sends requests with average prompt length 512 tokens, output length 256 tokens, and expects < 50ms TPOT (time per output token). You have a budget of 8 H100s. Recommend a configuration.

:::dropdown Answer

**Constraints:**
1. TPOT < 50ms → decode step time < 50ms at the batch size we're serving
2. 8 H100s available
3. High concurrency (1,000 users)

**Configuration options:**

Option A: TP=8 (single model instance across all 8 H100s)
- Memory: 140 GB across 8 GPUs = 17.5 GB per GPU (well under 80 GB)
- KV cache per GPU: ~62 GB
- Max concurrent at 1k context: ≈ $$62 \text{ GB} / (320 \text{ KB} \times 1024) \approx 190$$ sequences
- But 1,000 users → need queuing/batching

Option B: 2× replicas with TP=4 each
- Each replica has 4 H100s, 320 GB total memory
- Model weights (BF16): 140 GB → 35 GB per GPU
- KV cache headroom: 180 GB per replica → ≈100 concurrent sequences per replica
- Total: ≈200 concurrent active sequences
- Step time at B=100: ≈46 ms (from Question 5 math, adjusted for 4 GPUs)... well within 50ms ✓

Option B is better: higher concurrency (2 replicas can handle burst traffic independently), and TPOT stays under 50ms.

**Recommended:** 2 replicas of TP=4, BF16, with continuous batching (vLLM or SGLang). For lower cost: INT8 weights with TP=2 per replica (4 H100s total per replica with 4 replicas, or 2 H100s per replica with more replicas).

:::

---

## Summary

| Metric | LLaMA 3 70B BF16 | LLaMA 3 70B FP8 | LLaMA 3 70B INT4 |
|:---|:---:|:---:|:---:|
| Weight memory | 140 GB | 70 GB | 35 GB |
| Min GPUs (weights only) | 2 | 1 | 1 |
| KV per sequence (8k) | 2.62 GB | 2.62 GB | 2.62 GB |
| B\_crit (decode) | 295 | 591 | 148 |
| Practical decode regime | Memory-bound | Memory-bound | Memory-bound |
| Peak throughput (1 GPU) | N/A | ~1,900 tok/s | ~1,200 tok/s |

**Key insight:** decode is always memory-bound in practice. The bottleneck is HBM bandwidth, not compute FLOPs. Quantization helps by reducing weight bytes loaded per step, effectively increasing the batch size at which you hit B_crit.

For H100 production serving:
- **FP8 + vLLM** for maximum throughput on NVIDIA hardware
- **AWQ-4bit + vLLM** for cost-optimized deployment on fewer GPUs
- **TP within-node (NVLink)** and multiple replica instances for high-concurrency serving
