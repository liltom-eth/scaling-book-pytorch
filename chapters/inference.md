# Transformer Inference

## The Basics

Inference is conceptually simple: feed tokens in, get log-probabilities out, sample the next token, append it, repeat. But naively, this re-processes the entire prefix at every step — $O(n^2)$ for the FFW and $O(n^3)$ for attention to generate $n$ tokens.

**The KV cache** fixes this. During the forward pass, each token's key and value projections are computed once and saved. Future tokens only compute their own $q_i \cdot k_j$ products against the cached keys/values, never re-processing prior tokens. With a KV cache:

- Time to generate $n$ tokens: $O(n)$ on FFW, $O(n^2)$ on attention
- Each generation step is a single forward pass with batch size = 1 per sequence (plus the cache read)

This splits inference into two fundamentally different regimes:

**Prefill:** process the full prompt in parallel, populate the KV cache. Like training — batch size is the prompt length. Usually compute-bound.

**Decode (generation):** generate one token at a time, reading the full KV cache at each step. Batch size per request is 1 token. Almost always memory-bound.

Understanding this split is the foundation of all inference optimization.

## What Are We Optimizing?

Training has one metric: throughput (tokens/sec/GPU). Inference has two:

- **TTFT (Time To First Token):** latency from prompt submission to first generated token. Driven by prefill speed.
- **TPOT (Time Per Output Token):** latency per generated token. Driven by decode speed.
- **Throughput:** total tokens generated per second across all concurrent requests. Driven by batching efficiency.

These trade off against each other. Batching more requests together improves throughput but increases TTFT (the batched prefill takes longer). Choosing operating points requires understanding the hardware math.

## Linear Operations: Compute-Bound or Memory-Bound?

All transformer matmuls — MLP ($W_\text{in}$, $W_\text{out}$) and attention projections ($W_Q$, $W_K$, $W_V$, $W_O$) — follow the same roofline analysis. For a BF16 matmul of $[B, D] \times [D, F]$:

$$T_\text{math} = \frac{2BDF}{C}$$

$$T_\text{mem} = \frac{2BD + 2DF + 2BF}{W_\text{HBM}}$$

For large $D$ and $F$ ($D, F \gg B$, which holds for any reasonable model with small batch size):

$$T_\text{mem} \approx \frac{2DF}{W_\text{HBM}} \quad \text{(weight loading dominates)}$$

Compute-bound when $T_\text{math} > T_\text{mem}$:

$$\frac{2BDF}{C} > \frac{2DF}{W_\text{HBM}} \implies B > \frac{C}{W_\text{HBM}} = B_\text{crit}$$

For **H100 SXM5** (BF16 params and activations):

$$B_\text{crit} = \frac{989 \times 10^{12}}{3.35 \times 10^{12}} \approx 295 \text{ tokens}$$

With quantization, the critical batch size changes:

$$B_\text{crit} = \frac{C}{W_\text{HBM}} \cdot \frac{\text{bits per param}}{\text{bits per activation}}$$

| Config | $B_\text{crit}$ on H100 |
|:---|:---:|
| BF16 weights, BF16 activations | 295 |
| INT8 weights, BF16 activations | 148 |
| FP8 weights, FP8 activations | 295 |
| INT4 weights, BF16 activations | 74 |

:::note
**Takeaway — prefill:** Prompts are typically 100–10,000 tokens long. With $B > 295$, prefill is compute-bound on H100 BF16. Maximize GPU utilization (MFU) and you maximize both throughput and TTFT.
:::

:::note
**Takeaway — decode:** Each decode step has B = 1 token per request. Even with 100 concurrent requests (B=100 total), this is below $B_\text{crit} = 295$. **Decode is almost always memory-bound.** The bottleneck is HBM bandwidth, not compute.
:::

The implication: to optimize decode throughput, maximize the weight bytes read per second ($W_\text{HBM} = 3.35$ TB/s on H100) by batching more requests, using quantization to reduce weight size, and avoiding memory fragmentation.

## Attention: The Quadratic Term

Dot-product attention has a different arithmetic intensity than linear ops because the cost scales with sequence length, not model width.

For the attention computation with query $Q[B, T, N, H]$, keys $K[B, S, K, H]$, values $V[B, S, K, H]$:

$$\text{FLOPs} \approx 4BTSKGH = 4BTSNH$$

$$\text{Bytes} \approx 4BTNH + 4BSKH = 4BHK(TG + S)$$

Arithmetic intensity $\approx \frac{4BTSNH}{4BHK(TG + S)}$.

During prefill ($S = T$): intensity $\approx O(T)$ — attention becomes compute-bound at long context lengths (T > 295 tokens on H100 BF16).

During decode ($T = 1$, $S = $ context length): intensity $= O(G)$ where $G = N/K$ is the GQA group size. For LLaMA 3 with G=4: intensity = 4, far below $B_\text{crit}$. **Attention decode is memory-bound regardless of context length.**

The KV cache must be read from HBM at every decode step. For a sequence of length S with K KV heads and head dim H:

$$\text{KV cache bytes read per step} = 4SLKH \cdot \frac{1}{B} \text{ per request}$$

For LLaMA 3 8B (L=32, K=8, H=128, S=4096): 4 × 4096 × 32 × 8 × 128 = 536 MB per request per step.

At 3.35 TB/s HBM bandwidth, time to read one request's KV cache: ≈ 0.16 ms. With 64 concurrent requests: ≈ 10 ms/token. 

## Memory Budget for Inference

Unlike training, inference doesn't need optimizer state or gradients. The memory budget is:

| Component | Memory |
|:---|:---|
| Model weights | 2P bytes (BF16) or P bytes (INT8) |
| KV cache | 4SLKH bytes per sequence (BF16) |
| Activations | Small (one layer at a time) |

For LLaMA 3 70B (P ≈ 70B, L=80, K=8, H=128) in BF16:
- Weights: 140 GB — needs **2 H100s** (or INT8/INT4 quantization for 1 GPU)
- KV cache per sequence at S=8192: 4 × 8192 × 80 × 8 × 128 bytes = 2.15 GB
- Concurrent requests on 2 H100s (160 GB total, 140 GB for weights): 20 GB remaining → ~9 concurrent requests

This is why inference systems prioritize KV cache management so aggressively. Serving 100 concurrent requests on 2 H100s at 8k context would require 215 GB for KV caches alone — impossible.

Solutions: quantize KV caches to INT8 (half the memory), use shorter context, use more GPUs, or use paged memory management (vLLM's PagedAttention).

## Throughput and Latency Estimates

For a decode step on a single H100 with LLaMA 3 8B (140 GB weights... wait, 8B fits — 8B × 2 bytes = 16 GB):

**Single-request decode (memory-bound):**

Time to load weights: $16\text{ GB} / 3.35\text{ TB/s} \approx 4.8\text{ ms}$

At 4.8 ms/token: $1/0.0048 \approx 208 \text{ tokens/s}$ — but this is for a single request.

**Batch of 50 requests (still memory-bound, weight load amortized):**

Time to load weights: still ~4.8 ms (weights are the same regardless of batch size in the memory-bound regime).

Plus KV cache: 50 × 536 MB × 2 (K and V for 4096 tokens) / 3.35 TB/s ≈ 0.016 ms — negligible.

Throughput: 50 requests × 1 token / 4.8 ms ≈ **10,400 tokens/s** at 50 concurrent users.

More concurrent requests → more throughput (up to the memory-bound limit). The ceiling is when batching enough requests makes us compute-bound (B > 295 tokens), at which point adding more requests doesn't help throughput.

**Compute-bound ceiling:**

At B = 295 (batch of 295 requests): compute time ≈ memory time:

$$T = \frac{2 \times 295 \times D \times F \times L}{C} \approx \frac{2 \times 295 \times 4096 \times 14336 \times 32}{989\text{e12}} \approx 112\text{ ms}$$

Throughput at saturation: 295 / 0.112 ≈ **2,634 tokens/s** per H100 — worse than 50 concurrent requests at 208 tokens/s each? Wait, let me recalculate:

At 50 requests, 4.8 ms/step: throughput = 50/0.0048 ≈ 10,400 tok/s. 
At 295 requests, 112 ms/step: throughput = 295/0.112 ≈ 2,634 tok/s.

This seems to go down. That's because with 295 tokens, we've crossed into compute-bound territory — compute time (112 ms) >> memory time (4.8 ms). The optimal operating point is near B_crit = 295, but only if the compute time matches the memory time.

Corrected: at B = 295, if still memory-bound:
- Memory time: 4.8 ms (same weight load)
- Throughput: 295 / 4.8 ms ≈ **61,500 tokens/s**

This is why batching matters so much: throughput scales linearly with batch size all the way up to B_crit.

## Prefill vs Decode: Disaggregation

In production systems, prefill (compute-bound) and decode (memory-bound) have very different hardware needs. Running them on the same GPU means a sub-optimal compromise for both.

**Disaggregated prefill** (used by vLLM v0.6+, SGLang) separates prefill and decode into different GPU pools:

```
[User request]
       ↓
[Prefill GPU pool]  ← compute-optimized, large batch, high MFU
       ↓ (KV cache transfer)
[Decode GPU pool]   ← memory-bandwidth-optimized, large concurrent requests
       ↓
[Response stream]
```

The KV cache is transferred between pools via NVLink or InfiniBand after prefill completes. This adds transfer latency but allows each pool to be tuned independently.

## Multi-GPU Inference

For models that don't fit on one GPU, inference uses the same parallelism strategies as training, with slightly different trade-offs.

### Tensor Parallelism for Inference

The most common strategy: shard weight matrices across GPUs (same as Megatron-LM's TP). Each GPU holds 1/Y of the weight matrices; each decode step requires one AllReduce of the output activations.

```python
# Using HuggingFace + tensor parallel (Accelerate)
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    device_map="auto",          # auto tensor-parallels across GPUs
    torch_dtype=torch.bfloat16,
)
```

**Cost:** one AllReduce per layer per decode step. For output $Out[B, D]$ in BF16: $2BD$ bytes.

At B=50 requests, D=8192: AllReduce ≈ 50 × 8192 × 2 bytes = 820 KB. At NVLink 900 GB/s: ≈ 0.9 µs — negligible.

Across InfiniBand (50 GB/s): ≈ 16 µs per layer × 80 layers = 1.3 ms. Non-trivial — keep TP within-node (NVLink) for inference.

### Pipeline Parallelism for Inference

PP splits layers across GPUs. Each GPU processes its stages and passes activations to the next. With a single request (no micro-batching), only one GPU is active at a time — GPU utilization is 1/Z.

To hide this, batch multiple requests and use **micro-batch pipelining**: while GPU 0 processes request 1's stage, GPU 1 processes request 2's stage. This works well for large batch sizes but adds latency.

For decode (sequential token generation), PP is usually less efficient than TP at small batch sizes.

### KV Cache Sharding

The KV cache must be accessible to all layers that need it. With TP, the KV cache is already naturally sharded: each GPU holds the KV for its attention heads (GQA groups are distributed). No additional communication is needed for the KV cache with TP.

With PP, the KV cache for each layer lives on the corresponding pipeline stage's GPU — no inter-stage KV cache traffic.

## Speculative Decoding

**Speculative decoding** uses a small "draft" model to propose K tokens at once, then verifies them in parallel with the large model. If the draft is usually correct, this provides a speedup proportional to the average acceptance rate.

```python
from transformers import pipeline

# Target model: LLaMA 3 70B (slow, accurate)
# Draft model: LLaMA 3 8B (fast, approximate)

pipe = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-70B",
    assistant_model="meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

For draft acceptance rate $\alpha$ and draft length K:
$$\text{Speedup} \approx \frac{1 + K\alpha}{1 + K \cdot (\text{draft cost} / \text{target cost})}$$

For K=5, α=0.7 with a 10× cheaper draft model: speedup ≈ (1 + 3.5)/(1 + 0.5) ≈ 3×.

Speculative decoding is most effective when the draft and target agree frequently (e.g., the target model has already seen similar distributions to the draft's training set) and when decode is the bottleneck (always true for memory-bound generation).

## Key Takeaways

- **Prefill is compute-bound** ($B_\text{crit} \approx 295$ tokens on H100 BF16). Maximize GPU utilization for fast TTFT.

- **Decode is memory-bound.** The bottleneck is HBM bandwidth (3.35 TB/s on H100). Throughput scales linearly with batch size up to $B_\text{crit}$; beyond that, you're compute-bound.

- **KV cache is the dominant memory consumer** at inference time. At 8k context, LLaMA 3 70B needs 2.15 GB per concurrent request. Quantizing KV cache to INT8 halves this.

- **Batch more to improve throughput.** Each additional concurrent request costs only the marginal KV cache memory and improves tokens/second proportionally up to $B_\text{crit}$.

- **Quantize weights to reduce $B_\text{crit}$** and the minimum memory bandwidth needed per step. INT8 weights halve $B_\text{crit}$ to 148, meaning you reach full throughput at smaller batch sizes.

- **Keep tensor parallelism within-node** for inference (NVLink). Cross-node AllReduce overhead (~16 µs/layer) becomes significant at small batch sizes.

[^1]: In production, "inference" covers several different workloads: streaming chat (latency-sensitive), batch processing (throughput-sensitive), and edge inference (memory-constrained). The optimal strategy differs for each.
[^2]: This chapter models GQA as N Q heads and K KV heads, with group size G = N/K. LLaMA 3 uses N=32/64, K=8, G=4/8 for 8B/70B.
