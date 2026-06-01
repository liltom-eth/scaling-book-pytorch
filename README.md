# How To Scale Your Model: A Systems View of LLMs on GPUs

A practical book for ML engineers on GPU hardware, PyTorch distributed training, and LLM inference at scale.

**Inspired by and modeled after the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) by Jacob Austin et al.** This book covers the same core ideas — roofline analysis, transformer math, parallelism strategies, training and inference — through the lens of PyTorch and NVIDIA GPUs, with additional depth on GPU-specific tooling (Triton, torch.compile, DeepSpeed, vLLM, quantization).

The book is live at: **https://liltom-eth.github.io/scaling-book-pytorch/**

## Chapters

### Part 1: Foundations
1. [Roofline Analysis](chapters/roofline.md)
2. [GPU Architecture](chapters/gpu-architecture.md) — SMs, warps, HBM, NVLink, H100/A100
3. [CUDA Under the Hood](chapters/cuda-mental-model.md) — cuBLAS, cuDNN, NCCL mental models
4. [Sharded Matrices and Device Meshes](chapters/sharding.md) — DTensor, distributed matmuls

### Part 2: Transformers and Inference
5. [Transformer Mathematics](chapters/transformers.md) — FLOPs, parameters, KV cache, attention
6. [Transformer Inference](chapters/inference.md) — prefill/decode, KV cache, batching
7. [Inference Systems](chapters/inference-systems.md) — vLLM, SGLang, TensorRT-LLM
8. [Quantization](chapters/quantization.md) — FP8, INT8, GPTQ, AWQ
9. [Applied Inference: Serving LLaMA 3](chapters/applied-inference.md)

### Part 3: Training
10. [Training Parallelism](chapters/training-parallelism.md) — DDP, FSDP/FSDP2, TP, PP, SP
11. [Training Efficiency](chapters/training-efficiency.md) — gradient checkpointing, mixed precision, offloading
12. [Applied Training: LLaMA 3 on GPUs](chapters/applied-training.md)

### Part 4: Advanced Training
13. [DeepSpeed and Megatron-LM](chapters/deepspeed-megatron.md) — ZeRO, Megatron TP+PP

### Part 5: Going Deeper
14. [Kernel Optimization and Triton](chapters/triton-kernels.md)
15. [torch.compile and torch.export](chapters/torch-compile.md)
16. [Profiling GPU Code](chapters/profiling.md) — NSight Systems, PyTorch Profiler

### Part 6: Conclusions
17. [Conclusions and Further Reading](chapters/conclusion.md)

## Running Locally

Requires Python 3.9+:

```bash
pip install -r requirements.txt

# Build HTML
jb build .

# Open in browser
open _build/html/index.html   # macOS
xdg-open _build/html/index.html  # Linux
```

GitHub Pages deployment is handled by `.github/workflows/deploy.yml` on every push to `main`.

## Contributing

If you see errors or want to contribute a chapter, open a GitHub issue or PR.

## Acknowledgments

This book is built on the structure and ideas of the [JAX Scaling Book](https://github.com/jax-ml/scaling-book) by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson, and Reiner Pope at Google DeepMind. The website uses [Jupyter Book](https://jupyterbook.org/).

## Citation

```
@misc{scaling-book-pytorch,
  title = {How to Scale Your Model: A Systems View of LLMs on GPUs},
  author = {liltom-eth},
  howpublished = {Online},
  note = {Retrieved from https://liltom-eth.github.io/scaling-book-pytorch/},
  year = {2026}
}
```
