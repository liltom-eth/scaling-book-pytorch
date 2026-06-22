# Quantization

Quantization reduces the number of bits used to represent model weights (and optionally activations). The primary motivations:

1. **Reduce memory:** 4-bit quantization stores 2× more parameters per byte vs INT8, 4× vs BF16
2. **Reduce $B_\text{crit}$:** fewer bytes per weight → compute-bound at smaller batch sizes
3. **Increase throughput:** smaller weights → faster HBM load → more tokens/sec in memory-bound decode

The cost: lower precision introduces quantization error. How much error, and where, depends heavily on the method.

## The Math of Quantization

**Uniform quantization** maps FP values to integers in $[-2^{b-1}, 2^{b-1}-1]$:

$$x_q = \text{round}\left(\frac{x}{s}\right), \quad x \approx x_q \cdot s$$

where $s$ (scale) is chosen to cover the range of $x$. The quantization error per element is bounded by $s/2$ (half the step size).

**Key insight:** a scale factor is required per quantized tensor to recover approximate floating-point values. How you choose the scale granularity and what you quantize determines most of the quality/performance trade-off.

| Granularity | Scale per | Notes |
|:---|:---|:---|
| Per-tensor | Entire weight matrix | Lowest overhead, highest error |
| Per-channel | Each output channel (row) | Standard for most INT8 schemes |
| Per-group | Group of G consecutive elements | Used in GPTQ, AWQ (G=128 typical) |
| Per-token (activations) | Each token's activation vector | Needed for INT8 activations |

## Weight-Only Quantization

The simplest and most common case: quantize weights to lower precision, keep activations in BF16. No accuracy loss from activation quantization; weight loading from HBM is halved (INT8) or quartered (INT4).

**Impact on $B_\text{crit}$:**

$$B_\text{crit} = \frac{C}{W_\text{HBM}} \cdot \frac{\text{bits per param}}{\text{bits per activation}}$$

| Config | $B_\text{crit}$ on H100 |
|:---|:---:|
| BF16 weights, BF16 activations | 295 |
| INT8 weights, BF16 activations | 148 |
| INT4 weights, BF16 activations | 74 |

With INT4 weights, you're compute-bound at just 74 concurrent tokens — much easier to saturate the GPU in production.

### Quantizing with bitsandbytes

`bitsandbytes` provides simple weight-only quantization for HuggingFace models:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# INT8 weight quantization (LLM.int8())
config_8bit = BitsAndBytesConfig(load_in_8bit=True)

# INT4 weight quantization (NF4 or FP4)
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — better for normally distributed weights
    bnb_4bit_use_double_quant=True,      # quantize the scale factors too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct",
    quantization_config=config_4bit,
    device_map="auto",
)
```

LLM.int8() uses mixed-precision: outlier features in FP16, remaining features in INT8. This preserves accuracy at the cost of slightly lower throughput vs. pure INT8.

## Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without fine-tuning. The main challenge: transformers have activation outliers — a small fraction of features have very large magnitudes that dominate the quantization range, causing large errors for the majority of values.

### GPTQ

GPTQ (Frantar et al. 2022) is the dominant PTQ method for weight-only quantization. It minimizes the layer-wise quantization error using second-order information (Hessian of the layer output w.r.t. weights):

$$\min_{W_q} \|W X - W_q X\|^2$$

GPTQ quantizes weights sequentially, using the Hessian to compensate for errors in already-quantized weights. This is done offline using a small calibration dataset (~128 samples).

```python
# Using AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,       # per-group quantization (128 elements per scale)
    desc_act=False,       # act-order: reorder columns by Hessian diagonal
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantize_config=quantize_config,
)

# Calibration data
examples = tokenizer(calibration_texts, return_tensors="pt", padding=True)
model.quantize(examples)
model.save_quantized("llama3-8b-gptq-4bit")
```

GPTQ-4bit with group\_size=128 typically loses < 1 perplexity point on standard benchmarks vs the full-precision model, with 4× memory reduction.

### AWQ (Activation-Aware Weight Quantization)

AWQ (Lin et al. 2023) addresses the outlier problem differently: instead of using second-order optimization, it finds which weight channels are most sensitive by looking at activation magnitudes. It then applies a per-channel scale to "protect" these channels before quantization.

$$W_q = \text{Quant}(W \cdot \text{diag}(s)), \quad x' = x \cdot \text{diag}(s)^{-1}$$

This is equivalent in computation (scales cancel in the matmul) but concentrates quantization error on channels that are less sensitive.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("llama3-8b-awq-4bit")
```

AWQ is faster than GPTQ to quantize (seconds vs. minutes) and achieves comparable accuracy. It's the default in vLLM's quantization pipeline.

**GPTQ vs AWQ:**
- GPTQ: slower to quantize, slightly better accuracy in some benchmarks, more mature tooling
- AWQ: faster to quantize, comparable accuracy, first-class vLLM support

For most use cases, **AWQ-4bit with group_size=128** is the recommended starting point.

## FP8 Quantization

FP8 (E4M3) is an 8-bit floating-point format supported natively on H100. Unlike INT8, FP8 uses a floating-point representation with 4 exponent bits and 3 mantissa bits, which handles the outlier problem more gracefully than integer formats.

On H100, FP8 matmuls run at 1979 TFLOP/s (with sparsity) — 2× the BF16 throughput. This matters for **compute-bound** workloads (prefill, training) as well as inference.

**FP8 with PyTorch:**

```python
# Using torchao for FP8 training and inference
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

# For inference: quantize weights + activations to FP8
from torchao.quantization import quantize_, float8_weight_only

quantize_(model, float8_weight_only())
```

**FP8 with vLLM:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --quantization fp8 \
    --tensor-parallel-size 8
```

vLLM supports FP8 quantization with per-channel static or dynamic scales, calibrated from a representative dataset.

**FP8 accuracy:** generally within 0.1–0.5% of BF16 on standard benchmarks for weight-only FP8. Dynamic activation quantization can cause larger degradation on tasks with outlier-heavy distributions.

## KV Cache Quantization

The KV cache is a major memory consumer at inference time (see Chapter 10). Quantizing it to INT8 halves memory usage with minimal accuracy loss:

```python
# vLLM KV cache quantization
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B \
    --kv-cache-dtype fp8 \          # FP8 KV cache
    --tensor-parallel-size 8
```

KV cache quantization is almost free in terms of accuracy — the attention softmax is relatively robust to small precision errors in keys/values.

## Accuracy vs Efficiency Trade-offs

A rough guide to expected accuracy degradation and memory savings:

| Method | Bits | Memory vs BF16 | Perplexity increase (approx) |
|:---|:---:|:---:|:---:|
| BF16 (baseline) | 16 | 1× | 0 |
| FP8 W8A8 | 8 | 0.5× | ~0.1% |
| INT8 W8A16 (LLM.int8()) | 8 | 0.5× | ~0.2% |
| GPTQ W4A16 (group=128) | 4 | 0.25× | ~0.5% |
| AWQ W4A16 (group=128) | 4 | 0.25× | ~0.5% |
| GPTQ W4A16 (group=32) | 4 | 0.27× | ~0.3% |
| 2-bit (extreme) | 2 | 0.125× | >2% |

Perplexity increases translate non-linearly to benchmark degradation — tasks with tight answer distributions (multiple choice, math) degrade faster than open-ended generation.

**Rule of thumb:** 4-bit quantization is safe for most production deployments. 2-bit quantization is generally too lossy for instruction-following models.

## Worked Example: LLaMA 3 70B Quantization Options

LLaMA 3 70B in BF16 requires 140 GB — more than 1 H100 (80 GB). Options:

| Config | Memory | GPUs | Throughput notes |
|:---|:---:|:---:|:---|
| BF16 | 140 GB | 2 | TP=2, NVLink |
| FP8 W8A8 | 70 GB | 1 | Single GPU! 2× BF16 TFLOP/s |
| INT8 W8A16 | 70 GB | 1 | Single GPU, BF16 compute |
| AWQ W4A16 | 35 GB | 1 | Single GPU, plenty of KV cache headroom |

FP8 on a single H100 is compelling: same number of parameters fit in half the memory, and H100 FP8 matmuls are 2× faster than BF16. The accuracy cost is minimal for most workloads.

For production serving where quality matters most: **FP8 on H100** is the sweet spot — maximum performance with minimal accuracy loss. For constrained hardware (single consumer GPU): **AWQ 4-bit** fits 70B in 35 GB with good accuracy.

## Key Takeaways

- **Weight-only quantization** (W4A16, W8A16) is safe and widely deployed. INT8 halves memory with near-zero accuracy loss; INT4 quarters it with ~0.5% degradation.

- **GPTQ and AWQ** are the standard PTQ methods for 4-bit weight quantization. AWQ is faster and has better vLLM integration; GPTQ is slightly more accurate on some benchmarks.

- **FP8 (H100)** is the best option when you can afford to stay on NVIDIA hardware: 2× throughput vs BF16 with minimal accuracy loss. Use `torchao` or vLLM's built-in FP8 support.

- **KV cache quantization** halves KV memory with virtually no accuracy impact. Enable it whenever serving long contexts.

- **Group size matters:** smaller groups (32 vs 128) improve accuracy at the cost of slightly more overhead from storing more scale factors.
