# Conclusions and Further Reading

**Thank you for reading the whole thing and congratulations on making it all the way to the end.**

## The Mental Models That Carry Forward

This book is organized around a handful of mental models that should serve you well even as the specific hardware and software evolve:

**Roofline analysis.** Every operation is either compute-bound or memory-bound. Knowing which one tells you what to optimize. The critical intensity — FLOPs/byte — for H100 BF16 is ~295. It will be different on the next chip generation, but the framework is eternal.

**The memory hierarchy.** Registers → L1/Shared (SRAM) → L2 → HBM → CPU RAM → NVMe. Each level is faster and smaller than the next. Algorithms that tile computation to stay in SRAM beat algorithms that spill to HBM, which beat algorithms that go to CPU.

**Prefill vs decode.** Inference is two tasks disguised as one. Prefill is compute-bound (parallel token processing). Decode is memory-bound (one token at a time, weights read fresh every step). These require different hardware configurations and optimization strategies.

**Communication bandwidth hierarchy.** NVLink (900 GB/s) is 18× faster than InfiniBand NDR (50 GB/s). Design your parallelism to keep high-bandwidth operations within-node and lower-frequency operations across nodes.

**6 × params × tokens.** Training FLOPs are always approximately this product. Everything else — MFU, cost, GPU count — follows.

## What Changes as Models Scale

The rules shift at different scales:

- **< 4B parameters:** fit on one H100 with DDP. The bottleneck is usually data loading or learning rate tuning.
- **4B–70B parameters:** FSDP is necessary. TP within a node if activation memory is the bottleneck. The engineering complexity is manageable with vanilla PyTorch.
- **70B–200B parameters:** TP=8 + FSDP across nodes becomes standard. Megatron-LM or FSDP2 + DTensor. Communication overlap matters significantly.
- **200B+:** 3D parallelism (TP + PP + DP), sequence parallelism, pipeline bubble management. The cluster size itself requires careful topology design (NVSwitch, InfiniBand fat-tree).

The math in this book applies across all scales — the regime just changes where the binding constraints are.

## Acknowledgments

This book is directly inspired by and modeled on the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, and Reiner Pope at Google DeepMind. The original book's structure, mathematical rigor, and first-principles approach are what make it excellent — this book attempts to bring those qualities to the GPU/PyTorch ecosystem.

(further-reading)=
## Further Reading

**GPU architecture and CUDA:**
- [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html) — GPU roofline analysis and performance engineering, highly recommended
- [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance**](https://siboehm.com/articles/22/CUDA-MMM) — detailed CUDA kernel optimization walkthrough
- [**GPU Mode Lectures**](https://github.com/gpu-mode/lectures) — excellent community lectures on GPU programming

**Distributed training:**
- [**HuggingFace Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — practical depth on PyTorch parallelism implementations
- [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering) — practical guide to ML infrastructure: cluster management, debugging, benchmarking
- [**Megatron-LM paper**](https://arxiv.org/abs/1909.08053) and [**Megatron-Turing NLG paper**](https://arxiv.org/abs/2201.11990) — original references for Tensor Parallelism and 3D parallelism

**Inference:**
- [**Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102) — detailed math of Transformer inference, the inspiration for the inference chapters
- [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/) — concise blog with excellent illustrations of prefill/decode math
- [**vLLM paper**](https://arxiv.org/abs/2309.06180) — PagedAttention and continuous batching

**Courses and textbooks:**
- [**Stanford CS336**](https://stanford-cs336.github.io/) — comprehensive course on LLM training and serving with assignments
- [**Triton Tutorials**](https://triton-lang.org/main/getting-started/tutorials/) — official Triton docs with worked examples (softmax, matmul, Flash Attention)
- [**CUDA by Example**](https://developer.nvidia.com/cuda-example) — accessible introduction to CUDA programming

**Key papers:**
- [Flash Attention](https://arxiv.org/abs/2205.14135) (Dao et al. 2022), [Flash Attention 2](https://arxiv.org/abs/2307.08691) (Dao 2023)
- [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437) — excellent real-world training recipe at scale
- [LLaMA 3 Technical Report](https://arxiv.org/abs/2407.21783) — Meta's published training details

## Feedback

If you find errors or want to contribute a chapter, open an issue or pull request on GitHub.
