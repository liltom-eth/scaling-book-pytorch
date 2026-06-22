# Applied Training: LLaMA 3 on GPUs

This chapter works through the hardware math for training LLaMA 3 8B and 70B on H100 clusters. The goal is to be able to estimate GPU count, step time, training time, and cost before writing a single line of training code. These estimates also serve as sanity checks: if your actual run is 3× slower than the estimate, something is wrong.

Work through each question with a pen and paper before looking at the answer.

## Model Specs

| Hyperparameter | LLaMA 3 8B | LLaMA 3 70B |
|:---|:---:|:---:|
| D (d\_model) | 4096 | 8192 |
| F (d\_ff, SwiGLU) | 14336 | 28672 |
| L (layers) | 32 | 80 |
| N (Q heads) | 32 | 64 |
| K (KV heads, GQA) | 8 | 8 |
| H (head dim) | 128 | 128 |
| V (vocab size) | 128,256 | 128,256 |
| Training tokens | 15T | 15T |
| Context length | 8192 | 8192 |

Hardware reference: **H100 SXM5**
- BF16 peak: 989 TFLOP/s
- HBM bandwidth: 3.35 TB/s
- NVLink bandwidth: 900 GB/s (bidirectional)
- InfiniBand NDR: ~50 GB/s per GPU
- HBM capacity: 80 GB

---

## Part 1: LLaMA 3 8B

### Question 1: Parameter count

How many parameters does LLaMA 3 8B have? Break it down by component.

:::dropdown Answer

$$\text{MLP params per layer} = 3DF = 3 \times 4096 \times 14336 = 176{,}160{,}768$$

$$\text{Attention params per layer} = 2D(N+K)H = 2 \times 4096 \times 40 \times 128 = 41{,}943{,}040$$

$$\text{Embedding} = DV = 4096 \times 128256 = 525{,}336{,}576$$

$$\text{Total} = L \times (\text{MLP} + \text{Attn}) + \text{Embedding}$$

$$= 32 \times (176{,}160{,}768 + 41{,}943{,}040) + 525{,}336{,}576$$

$$= 32 \times 218{,}103{,}808 + 525{,}336{,}576 \approx 7.5\text{B}$$

Including layer norm parameters and rounding to the usual convention: **~8B parameters**.

MLP accounts for about 80% of parameters; attention about 18%; embeddings about 7%.

:::

### Question 2: Training FLOPs

How many total FLOPs are required to train LLaMA 3 8B on 15 trillion tokens?

:::dropdown Answer

Using the $6 \times \text{params} \times \text{tokens}$ rule:

$$\text{FLOPs} = 6 \times 8 \times 10^9 \times 15 \times 10^{12} = 7.2 \times 10^{23}$$

For comparison, GPT-4 is estimated at $\sim 2 \times 10^{25}$ FLOPs; LLaMA 3 8B is about 36× cheaper.

:::

### Question 3: Minimum GPU-hours

What is the minimum number of H100 GPU-hours to train LLaMA 3 8B, assuming peak BF16 throughput?

:::dropdown Answer

$$\text{Time per GPU} = \frac{7.2 \times 10^{23}}{989 \times 10^{12} \text{ FLOPs/s}} \approx 7.28 \times 10^{8} \text{ s}$$

$$= \frac{7.28 \times 10^8}{3600} \approx 202{,}200 \text{ GPU-hours}$$

At 100% hardware utilization (physically impossible but useful as a lower bound). Typical hardware utilization (MFU — model FLOPs utilization) for LLM training is 35–50%. At 40% MFU:

$$\text{Actual GPU-hours} \approx \frac{202{,}200}{0.4} \approx 505{,}000 \text{ GPU-hours}$$

Meta's LLaMA 3 8B paper reports training on 1.46M GPU-hours total (including cooling, overhead, multiple runs). Order of magnitude consistent.

:::

### Question 4: Memory fit check

Can LLaMA 3 8B fit on a single H100 for training with BF16 parameters and FP32 Adam? If not, how many GPUs are needed minimum?

:::dropdown Answer

From Chapter 7:
- Parameters: 8B × 2 bytes (BF16) = 16 GB
- Adam optimizer state (FP32 m + v): 8B × 8 bytes = 64 GB
- Gradients (BF16): 8B × 2 bytes = 16 GB
- Total (no activations): 96 GB

This **does not fit** on a single H100 (80 GB). With FSDP across 2 GPUs: 96/2 = 48 GB for params/optimizer/grads, leaving 32 GB for activations. That's tight but feasible with gradient checkpointing.

In practice, LLaMA 3 8B is typically trained with FSDP across 16–64 H100s to have headroom for activations and the optimizer warmup.

:::

### Question 5: Step time with DDP

You're using DDP with 64 H100s, global batch size of 8M tokens, T=8192 sequence length. Estimate the step time. Is training compute-bound or communication-bound?

:::dropdown Answer

**Per-GPU batch size:** 8M / 64 = 125,000 tokens/GPU.

**Compute time per step per GPU:**

$$T_\text{math} = \frac{6 \times 125{,}000 \times 8 \times 10^9}{989 \times 10^{12}} \approx 6.08 \text{ s}$$

**DDP AllReduce communication (cross-node, InfiniBand 50 GB/s):**

Weight matrices per layer: $2 \times 2DF \approx 2 \times 2 \times 4096 \times 14336 \approx 235\text{ MB}$ per layer. For L=32 layers: ~7.5 GB total.

Ring AllReduce on 64 GPUs (8 nodes): total bytes × 2 / bandwidth:
$$T_\text{comms} = \frac{2 \times 7.5\text{ GB}}{50\text{ GB/s}} \approx 0.3 \text{ s}$$

**Compute-bound check:** B/X = 125,000 >> 26,373 threshold from Chapter 6. Strongly compute-bound. ✓

**Step time ≈ 6 s** (dominated by compute, not comms, since DDP overlaps AllReduce with backward).

:::

---

## Part 2: LLaMA 3 70B

### Question 6: Parameter count and memory

How many parameters does LLaMA 3 70B have? What is the minimum GPU memory for training parameters + optimizer state?

:::dropdown Answer

$$\text{MLP per layer} = 3 \times 8192 \times 28672 = 704{,}643{,}072$$

$$\text{Attention per layer} = 2 \times 8192 \times (64 + 8) \times 128 = 150{,}994{,}944$$

$$\text{Embedding} = 8192 \times 128256 = 1{,}050{,}673{,}152$$

$$\text{Total} = 80 \times (704{,}643{,}072 + 150{,}994{,}944) + 1{,}050{,}673{,}152 \approx 69.4\text{B}$$

Rounding to the conventional 70B.

**Memory for training (BF16 params + FP32 Adam):**

$$70\text{B} \times 18 \text{ bytes/param} = 1.26\text{ TB}$$

Minimum GPUs at 80 GB HBM: $\lceil 1260/80 \rceil = 16$ GPUs, though in practice you need more headroom for activations.

:::

### Question 7: FSDP scaling

With FSDP across 512 H100s (64 nodes × 8 GPUs) and 16M tokens/step batch size, is training compute-bound?

:::dropdown Answer

**Per-GPU batch size:** 16M / 512 = 31,250 tokens/GPU.

**FSDP compute-bound threshold (InfiniBand, W = 50 GB/s):**

$$\frac{B}{X} > \frac{4C}{3W} = \frac{4 \times 989\text{e12}}{3 \times 50\text{e9}} \approx 26{,}373$$

B/X = 31,250 > 26,373. Compute-bound. ✓ (barely, with ~18% margin)

**Step time:**

$$T_\text{math} = \frac{6 \times 31{,}250 \times 70 \times 10^9}{989 \times 10^{12}} \approx 13.2 \text{ s/GPU}$$

Wall-clock step time ≈ 13.2 s (all GPUs run in parallel).

:::

### Question 8: TP + FSDP configuration

You have 512 H100s in 64 nodes of 8 GPUs each. What's the natural TP + FSDP split?

:::dropdown Answer

**TP = 8** (within each node, over NVLink).

**FSDP = 512 / 8 = 64** (one per node, across InfiniBand).

This gives:
- TP AllReduce (NVLink, 900 GB/s): activation tensor $Out[B_\text{micro}, T, D] = [4, 8192, 8192]$ in BF16 ≈ 537 MB. Comms time: $2 \times 537\text{ MB} / 900\text{ GB/s} \approx 1.2\text{ ms}$ per layer.
- FSDP AllGather (InfiniBand, 50 GB/s): weight $W_\text{in}[D, F/8] = [8192, 3584]$ in BF16 ≈ 59 MB per layer. Comms time: $59\text{ MB} / 50\text{ GB/s} \approx 1.2\text{ ms}$ per layer.
- Compute time per layer per GPU: $6 \times 31250 \times 8192 \times 3584 / (989 \times 10^{12} \times 8) \approx 7\text{ ms}$

Both communication costs are smaller than compute. The TP + FSDP configuration is compute-bound with 512 H100s and 16M tokens/step.

:::

### Question 9: Total training time and cost

Using 512 H100s with the TP + FSDP configuration from Question 8, how long does training on 15T tokens take? What does it cost at $2/GPU-hour?

:::dropdown Answer

**Total training FLOPs:**

$$6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24}$$

**Step time:** ~13.2 s (from Question 7, adjusted slightly for TP overhead)

**Steps:** 15T tokens / 16M tokens per step = 937,500 steps

**Total wall-clock time:**

$$937{,}500 \times 13.2 \text{ s} \approx 1.24 \times 10^7 \text{ s} \approx 144 \text{ days}$$

**GPU-hours:** 144 days × 24 h × 512 GPUs ≈ **1.77M GPU-hours**

**Cost:** 1.77M × $2/GPU-hr ≈ **$3.5M**

This is a lower bound assuming perfect hardware utilization. Real costs are 2–3× higher due to node failures, restarts, suboptimal MFU (35–45%), and multiple trial runs. Meta's LLaMA 3 70B paper reports ~6.4M GPU-hours, consistent with 3–4× overhead.

:::

### Question 10: MFU estimate

During training you observe a step time of 18 s/step with the 512-GPU TP+FSDP setup and 16M tokens/step batch. What is your MFU?

:::dropdown Answer

**Theoretical FLOPs per step at peak throughput:**

$$\text{Peak FLOPs} = 512 \times 989 \times 10^{12} \times 18 \text{ s} = 9.1 \times 10^{18}$$

**Actual model FLOPs per step:**

$$6 \times 70 \times 10^9 \times 16 \times 10^6 = 6.72 \times 10^{18}$$

$$\text{MFU} = \frac{6.72 \times 10^{18}}{9.1 \times 10^{18}} \approx 73.8\%$$

Wait — this seems too high. Check: at 13.2 s theoretical step time, 18 s actual step time, MFU = 13.2/18 ≈ **73%**. This is actually excellent (most runs achieve 35–50%). Either this is an optimistic scenario or sequence parallelism and CUDA graphs are helping significantly. Real-world 70B training typically achieves 40–55% MFU.

If your MFU is < 35%, common culprits are: insufficient gradient checkpointing, data loading bottleneck, untuned bucket sizes in DDP, or non-overlapping communication.

:::

---

## Summary

| | LLaMA 3 8B | LLaMA 3 70B |
|:---|:---:|:---:|
| Parameters | ~8B | ~70B |
| Training FLOPs (15T tokens) | 7.2 × 10²³ | 6.3 × 10²⁴ |
| Minimum GPU memory | 16 GPUs | 128 GPUs |
| Recommended cluster | 64 H100s | 512 H100s |
| Parallelism strategy | FSDP only | TP=8 + FSDP=64 |
| Estimated step time | ~6 s | ~13 s |
| Estimated GPU-hours | 505K | 1.77M |
| Estimated cost (@$2/hr) | ~$1M | ~$3.5M |

These estimates use 40% MFU for the 8B model and assume ideal step time for the 70B (adjust upward ~2–3× for real-world overhead).

**The key skill this chapter demonstrates:** before touching a cluster, you can estimate whether a run is feasible, how long it will take, and whether your parallelism choices are compute-bound. The frameworks (FSDP, Megatron, DeepSpeed) handle the implementation; the math tells you if you're making the right choices.
