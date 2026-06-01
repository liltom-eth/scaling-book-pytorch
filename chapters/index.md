# How to Scale Your Model

Much of deep learning still boils down to a kind of black magic, but optimizing the performance of your models on GPUs doesn't have to — even at huge scale. Relatively simple principles apply everywhere — from a single H100 to a thousand-GPU cluster — and understanding them lets you do many useful things:

- Ballpark how close parts of your model are to their theoretical hardware optimum.
- Make informed choices about parallelism strategies (DDP, FSDP, tensor parallelism, pipeline parallelism) at different scales.
- Estimate the cost and time required to train and serve large Transformer models.
- Understand what PyTorch and CUDA are actually doing under the hood.
- Write faster kernels with Triton or debug torch.compile issues with confidence.

**Expected background:** We assume you have a basic understanding of LLMs and the Transformer architecture but not necessarily how they run at scale. You should know the basics of LLM training and have some familiarity with PyTorch. Useful background includes [the illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) and the [original Transformer paper](https://arxiv.org/abs/1706.03762). See [the conclusions](conclusion.md) for more reading.

**Goals:** By the end, you should feel comfortable estimating the best parallelism scheme for a Transformer model on a given GPU cluster, and roughly how long training and inference should take.

**This book is inspired by and closely follows the structure of the excellent [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) by Jacob Austin et al. We cover the same core ideas but through the lens of PyTorch and NVIDIA GPUs, with additional depth on GPU-specific tooling.**

## Why should you care?

A few years ago most ML engineers could get by without understanding any of this. Today, even "small" models run so close to hardware limits that doing meaningful work at scale requires thinking about efficiency. **A 20% improvement on benchmarks is irrelevant if it comes at a 20% cost to hardware utilization.** Promising architectures routinely fail because they can't run efficiently at scale, or because nobody put in the work to make them do so.

The goal of "model scaling" is to increase the number of GPUs used for training or inference while achieving a proportional, linear increase in throughput — *strong scaling*. This requires understanding where bottlenecks arise: compute throughput, memory bandwidth, inter-GPU communication over NVLink or InfiniBand. If you understand these, you can design or reconfigure your models to avoid them.

## High-Level Outline

**Part 1: Foundations**

[Chapter 1](roofline) introduces roofline analysis — the key mental model for understanding whether an operation is compute-bound or memory-bound. [Chapter 2](gpu-architecture) dives into GPU hardware: SMs, warp scheduling, the HBM → L2 → SRAM memory hierarchy, and how H100s connect over NVLink. [Chapter 3](cuda-mental-model) explains what PyTorch is doing under the hood when it dispatches to CUDA — cuBLAS, cuDNN, NCCL — without requiring you to write CUDA yourself. [Chapter 4](sharding) covers DTensor and device meshes, the primitives PyTorch uses for distributed computation.

**Part 2: Transformers and Inference**

[Chapter 5](transformers) works through Transformer math end to end: FLOPs, parameter counts, KV cache sizing, attention complexity, and MoE. [Chapter 6](inference) then builds directly on this with the inference story: prefill vs. decode, KV cache math, batching strategies, and latency/throughput trade-offs. [Chapter 7](inference-systems) surveys the major serving frameworks: vLLM, SGLang, and TensorRT-LLM. [Chapter 8](quantization) covers FP8, INT8, GPTQ, and AWQ. [Chapter 9](applied-inference) applies everything to serving LLaMA 3 with real cost and SLO analysis.

**Part 3: Training**

[Chapter 10](training-parallelism) covers every parallelism strategy: DDP, FSDP/FSDP2, tensor parallelism, pipeline parallelism, sequence parallelism. [Chapter 11](training-efficiency) covers memory-saving techniques: gradient checkpointing, BF16/FP8 mixed precision, activation offloading. [Chapter 12](applied-training) applies everything to LLaMA 3 with concrete GPU count, cost, and time estimates.

**Part 4: Advanced Training**

[Chapter 13](deepspeed-megatron) covers DeepSpeed ZeRO stages 1/2/3 and Megatron-LM 3D parallelism — when these frameworks offer advantages over vanilla PyTorch distributed.

**Part 5: Going Deeper**

[Chapter 14](triton-kernels) covers Flash Attention and writing efficient Triton kernels. [Chapter 15](torch-compile) covers how torch.compile works and how to debug it. [Chapter 16](profiling) covers NSight Systems and the PyTorch Profiler.

## Links to Sections

**Part 1: Foundations**

* [**Chapter 1: Roofline Analysis**](roofline). Algorithms are bounded by compute, memory, and communication. Rooflines let you estimate how fast GPU code will run before you write it.

* [**Chapter 2: GPU Architecture**](gpu-architecture). SMs, warps, HBM, NVLink, H100 vs. A100 — how the hardware shapes what models you can train and serve.

* [**Chapter 3: CUDA Under the Hood**](cuda-mental-model). How PyTorch dispatches to CUDA, what cuBLAS/cuDNN/NCCL are doing, and how the memory allocator works — a mental model without raw CUDA programming.

* [**Chapter 4: Sharded Matrices and Device Meshes**](sharding). DTensor, device meshes, and how sharding maps to distributed matrix operations across GPUs.

**Part 2: Transformers and Inference**

* [**Chapter 5: Transformer Mathematics**](transformers). FLOPs, parameters, KV cache sizing, attention complexity, MoE — the numbers behind every design decision.

* [**Chapter 6: Transformer Inference**](inference). Prefill vs. decode, KV cache math, batching, latency/throughput trade-offs, speculative decoding.

* [**Chapter 7: Inference Systems**](inference-systems). vLLM paged attention, continuous batching, SGLang RadixAttention, TensorRT-LLM compiled engines.

* [**Chapter 8: Quantization**](quantization). FP8, INT8, GPTQ, AWQ — the math, the accuracy trade-offs, when to use each.

* [**Chapter 9: Serving LLaMA 3**](applied-inference). Serving cost, SLO analysis, hardware sizing for different throughput targets.

**Part 3: Training**

* [**Chapter 10: Training Parallelism**](training-parallelism). DDP, FSDP/FSDP2, tensor parallelism, pipeline parallelism, sequence parallelism — when each applies and how to combine them.

* [**Chapter 11: Training Efficiency**](training-efficiency). Gradient checkpointing, BF16/FP8 mixed precision, activation offloading, gradient accumulation.

* [**Chapter 12: Training LLaMA 3 on GPUs**](applied-training). GPU count, bandwidth, cost, and time estimates for LLaMA 3 8B and 70B.

**Part 4: Advanced Training**

* [**Chapter 13: DeepSpeed and Megatron-LM**](deepspeed-megatron). ZeRO stages 1/2/3, Megatron-style TP+PP, when to use each over vanilla PyTorch distributed.

**Part 5: Going Deeper**

* [**Chapter 14: Kernel Optimization and Triton**](triton-kernels). Flash Attention walkthrough, writing Triton kernels, when custom kernels are worth it.

* [**Chapter 15: torch.compile and torch.export**](torch-compile). How the compiler pipeline works, what it fuses, common pitfalls and how to debug them.

* [**Chapter 16: Profiling GPU Code**](profiling). NSight Systems, PyTorch Profiler, reading traces, finding and fixing bottlenecks.

**Part 6: Conclusions**

* [**Chapter 17: Conclusions and Further Reading**](conclusion). Synthesis of mental models, what changes as models scale, curated reading list.
