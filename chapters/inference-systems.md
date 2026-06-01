# Inference Systems

Chapter 10 covered the hardware math of transformer inference. This chapter covers the systems that implement that math in production: **vLLM**, **SGLang**, and **TensorRT-LLM**. Each makes different trade-offs between generality, performance, and ease of use.

The core challenge they all solve: serve dozens to thousands of concurrent requests on a fixed set of GPUs, maximizing throughput while keeping latency bounded. The naive approach — one request at a time, static batching — leaves 99% of GPU capacity unused.

## The Core Problem: KV Cache Management

From Chapter 10, the KV cache for a single request at 8k context in BF16 LLaMA 3 70B is ~2.15 GB. For 100 concurrent requests, that's 215 GB — far exceeding GPU memory.

The naive solution: reserve a fixed KV cache slot per request at arrival time. This wastes memory because:
1. You reserve for the maximum possible length even if the request is short
2. Slots sit idle during prefill and between generation steps
3. Internal fragmentation from variable-length prefixes

**PagedAttention** (vLLM's key innovation) solves this by managing KV cache like virtual memory in an OS:

- KV cache is divided into fixed-size "pages" (e.g., 16 tokens each)
- Pages are allocated on demand as generation proceeds
- Non-contiguous pages are collected via a block table — attention is modified to look up pages via indirection
- Pages can be shared across requests (for identical prefixes — prefix caching)

This eliminates nearly all fragmentation and allows near-100% KV cache utilization.

## Continuous Batching

Static batching — waiting for a batch to fill up, then processing all requests to completion — wastes GPU time. Requests finish at different times, leaving partial batches.

**Continuous batching** (also called "in-flight batching") maintains a running batch. As each request generates an EOS token, it's removed from the batch and a new request is inserted immediately. The batch size stays roughly constant throughout.

```
Time →
Req A: [prefill] [gen] [gen] [gen] [EOS]
Req B:           [prefill] [gen] [gen] [gen] [EOS]
Req C:                     [prefill] [gen] [gen] [gen] [EOS]
          ↑ Req B starts as soon as there's space
```

Continuous batching + PagedAttention together are the foundation of modern LLM serving. They're implemented in vLLM and SGLang.

## vLLM

vLLM is the most widely used open-source LLM serving framework. It implements:
- PagedAttention for KV cache management
- Continuous batching
- Tensor parallelism (via `--tensor-parallel-size`)
- Speculative decoding
- Chunked prefill (interleave prefill tokens with decode tokens to bound TTFT)

**Quick start:**

```bash
pip install vllm

# Start an OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

In Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    dtype="bfloat16",
    max_model_len=8192,
)

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(["Tell me about GPUs.", "What is CUDA?"], params)
```

**KV cache utilization:** vLLM reports KV cache hit rates and utilization in logs. Monitor these in production — low utilization (< 80%) indicates either under-batching or prefix caching opportunities.

**Chunked prefill:** by default, long prefills can block decode. `--enable-chunked-prefill` breaks prefill into chunks interleaved with decode steps, bounding TTFT jitter:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048  # max tokens per step (prefill + decode)
```

### vLLM Architecture

```
[Request queue]
      ↓
[Scheduler]  ─── decides which requests to run each step
      ↓           ─── manages KV cache block allocation (PagedAttention)
[Worker pool]  ─── GPU workers with tensor-parallel model
      ↓
[Output processor]  ─── decodes token IDs, checks stop conditions
      ↓
[Response stream]
```

The scheduler is the bottleneck in high-throughput scenarios. It runs on CPU and must decide each step (4–10 ms) which requests to include. For thousands of concurrent requests, scheduler latency can add up.

## SGLang

SGLang (Structured Generation Language) from the researchers at UC Berkeley extends the continuous batching model with:

- **RadixAttention:** a tree-based prefix caching system that shares KV cache across requests with common prefixes (e.g., a shared system prompt)
- **Constrained decoding:** efficient structured output generation (JSON, regex) with compiled FSM
- **Disaggregated prefill/decode:** separate GPU pools for prefill and decode
- **CUDA graph replay:** captures decode steps as CUDA graphs, eliminating kernel launch overhead

**Quick start:**

```bash
pip install sglang[all]

python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tensor-parallel-size 2 \
    --enable-torch-compile
```

**RadixAttention prefix caching** is particularly valuable for chat applications where many requests share the same system prompt (often hundreds of tokens). Instead of recomputing the KV cache for the system prompt for every request, SGLang stores it once and reuses it:

```python
import sglang as sgl

@sgl.function
def chat_with_system_prompt(s, question):
    s += sgl.system("You are a helpful AI assistant specializing in GPU programming.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512))

# System prompt is prefill-cached after the first request
results = chat_with_system_prompt.run_batch([
    {"question": "What is a CUDA kernel?"},
    {"question": "Explain memory coalescing."},
    {"question": "What is warp divergence?"},
])
```

With a 500-token system prompt shared across all requests, RadixAttention saves 500 × (decode throughput / cache hit) tokens of prefill work per request.

**When to choose SGLang over vLLM:**
- High prefix sharing (shared system prompts, RAG pipelines with shared context)
- Structured output generation (JSON schemas, function calling)
- You want CUDA graph replay without manual graph capture
- Multi-modal or speculative decoding workloads that benefit from SGLang's pipeline

## TensorRT-LLM

TensorRT-LLM is NVIDIA's production-grade inference framework. It provides:
- **TensorRT-optimized engines**: compiles model weights + graph into a static NVIDIA TRT engine with op fusion, quantization, and kernel autotuning
- **FP8 inference** (H100): native support with per-tensor scaling
- **In-flight batching** with multi-head attention kernel optimizations
- **Multi-GPU and multi-node** inference support

**TensorRT-LLM workflow:**

```python
import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams

# Build a TRT engine from a HuggingFace model
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    dtype="bfloat16",
)

# Engines are compiled and cached — first run is slow
params = SamplingParams(max_tokens=256)
outputs = llm.generate(["Explain Flash Attention."], params)
```

The first run compiles the TRT engine (can take 10–30 minutes for large models). Subsequent runs use the cached engine and run significantly faster than PyTorch eager mode — typically 20–40% faster for the same hardware.

**When to choose TensorRT-LLM:**
- Maximum single-GPU throughput for a fixed set of model shapes
- FP8 inference on H100 (best support via TRT-LLM's calibration tools)
- NVIDIA Triton Inference Server integration
- Production deployments where the ~30% throughput advantage over vLLM justifies the complexity

**Trade-off:** TRT engine compilation ties you to specific batch sizes, sequence lengths, and dtypes. Dynamic shapes require building multiple engines or using profile-based TRT profiles. vLLM and SGLang are more flexible for variable workloads.

## Choosing a Framework

| | vLLM | SGLang | TensorRT-LLM |
|:---|:---|:---|:---|
| KV cache | PagedAttention | RadixAttention | Managed pool |
| Prefix caching | Basic | Tree-based (RadixAttention) | Manual |
| Throughput vs. vLLM | Baseline | ~5–15% better | ~20–40% better |
| First-time setup | pip install, ready | pip install, ready | 30-min compile |
| Dynamic shapes | ✓ | ✓ | Limited |
| FP8 (H100) | ✓ | ✓ | Best support |
| Structured output | Basic | First-class | Limited |
| HF Transformers integration | ✓ | ✓ | ✗ |

**Decision guide:**
- **Start with vLLM** — it works for most use cases, has excellent HuggingFace integration, and is well-documented.
- **Switch to SGLang** when you need prefix caching, structured generation, or CUDA graph replay.
- **Use TensorRT-LLM** when raw throughput is critical and you have fixed model configurations (enterprise deployment, fixed-shape batch jobs).

## Key Takeaways

- **PagedAttention** (vLLM) and **RadixAttention** (SGLang) solve KV cache fragmentation and enable near-100% memory utilization.

- **Continuous batching** maintains a full batch at all times by inserting new requests as old ones finish. This is the single largest throughput improvement over static batching.

- **Prefix caching** eliminates the cost of processing repeated prompt prefixes (system prompts, RAG context). With SGLang's RadixAttention, hit rates of 50–90% are common in chat applications.

- **TensorRT-LLM** provides the highest throughput at the cost of compilation time and reduced flexibility. Use it for production deployments with fixed configurations.

- **Chunked prefill** decouples prefill from decode, enabling bounded TTFT even under heavy load. Enable it in vLLM for latency-sensitive applications.
