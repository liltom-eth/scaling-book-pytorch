# DeepSpeed and Megatron-LM

The parallelism strategies in [Chapter 6](training-parallelism.md) are available natively in PyTorch. In practice, large-scale training often uses one of two production-grade frameworks built on top of PyTorch: **DeepSpeed** (from Microsoft) and **Megatron-LM** (from NVIDIA). Both implement the same fundamental ideas but with different trade-offs in API design, composability, and optimization focus.

This chapter covers what each framework adds beyond vanilla PyTorch distributed, when to prefer one over the other, and how to set up a realistic training run.

## DeepSpeed: ZeRO and Beyond

### ZeRO Stages Revisited

DeepSpeed's primary contribution is the **ZeRO** (Zero Redundancy Optimizer) family of algorithms. Chapter 6 introduced the three stages conceptually; here's a concrete comparison:

| Feature | ZeRO-1 | ZeRO-2 | ZeRO-3 / FSDP |
|:---|:---:|:---:|:---:|
| Optimizer state sharded | ✓ | ✓ | ✓ |
| Gradients sharded | | ✓ | ✓ |
| Parameters sharded | | | ✓ |
| Communication in forward pass | None | None | AllGather |
| Peak memory (P params, N GPUs) | ~(12 + 8/N)P bytes | ~(4 + 8/N)P bytes | ~12P/N bytes |
| Typical use case | Model fits on 1 GPU; save optimizer mem | More memory savings | Model too large for 1 GPU |

For models that fit on a single GPU (parameters + some activations), ZeRO-1 or ZeRO-2 may be preferable to ZeRO-3/FSDP because they avoid the AllGather overhead during the forward pass.

### Using DeepSpeed

DeepSpeed is configured via a JSON config file:

```json
{
  "train_batch_size": 256,
  "gradient_accumulation_steps": 8,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  }
}
```

In Python:

```python
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)
    model.backward(loss)
    model.step()
```

DeepSpeed wraps the model and optimizer into a single `engine` object. The engine handles gradient accumulation, mixed precision, and ZeRO communication automatically.

### CPU Offloading (ZeRO-Infinity)

DeepSpeed's ZeRO-Infinity extends ZeRO-3 by offloading parameters and optimizer state to CPU RAM (or NVMe). This allows training models that are 5–10× larger than GPU memory allows, at the cost of slower per-step throughput (limited by PCIe bandwidth).

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

CPU offloading is a last resort — GPU → CPU PCIe bandwidth (~64 GB/s) is about 50× slower than NVLink. Use it only when the model cannot fit in GPU memory even with ZeRO-3 and gradient checkpointing.

### DeepSpeed vs. PyTorch FSDP

| | DeepSpeed ZeRO-3 | PyTorch FSDP2 |
|:---|:---|:---|
| Maturity | Battle-tested since 2020 | GA in PyTorch 2.0+ |
| Configuration | JSON config file | Python API |
| Composability with TP | Limited | Native via DTensor |
| CPU offloading | Built-in (ZeRO-Infinity) | Manual / limited |
| Integration with HF Transformers | First-class | Growing |
| 3D parallelism | Via Megatron-DeepSpeed | Via FSDP2 + DTensor |

For new projects, **PyTorch FSDP2 + DTensor** is usually the better choice because it composes cleanly with tensor parallelism and `torch.compile`. DeepSpeed ZeRO-3 remains useful when CPU offloading is needed or when integrating with existing DeepSpeed-based infrastructure.

## Megatron-LM: 3D Parallelism

Megatron-LM, developed at NVIDIA, is the reference implementation for large-scale transformer training. It provides:

1. **Tensor parallelism** (column + row parallel linear layers)
2. **Pipeline parallelism** with interleaved 1F1B scheduling
3. **Sequence parallelism** to shard activations along the sequence dimension
4. **Selective activation recomputation**

These are combined into a **3D parallelism** strategy: TP within a node, PP across node groups, DP across PP groups.

### Tensor Parallelism in Megatron

Megatron's TP follows the column-row parallel pattern from Chapter 6, but with one key addition: **sequence parallelism** for the non-matmul operations (layer norm, dropout) that are not naturally tensor-parallelized.

Without sequence parallelism, each TP GPU holds a replicated copy of the layer norm output — wasting memory proportional to sequence length. With sequence parallelism, the sequence dimension is sharded across TP ranks outside the matmuls:

```
Input[B, T, D]  →  sharded: Input[B, T/Y, D]  (sequence parallel)
     ↓ AllGather along sequence dim
Input[B, T, D]  →  W_in[D, F/Y]  →  tmp[B, T, F/Y]  (column parallel)
     ↓ W_out[F/Y, D]  →  partial[B, T, D]
     ↓ ReduceScatter along sequence dim
Out[B, T/Y, D]  (sequence parallel)
```

The AllGather and ReduceScatter replace the AllReduce, but with the same total bytes — and sequence parallelism reduces activation memory by Y × because the layer norm inputs/outputs are also sharded.

### Pipeline Parallelism in Megatron

Megatron's **interleaved 1F1B** schedule assigns multiple virtual pipeline stages per GPU to minimize bubble overhead. With V virtual stages per GPU and Z pipeline stages total:

$$\text{Bubble fraction} = \frac{Z-1}{VM + Z - 1}$$

where M is the number of micro-batches. For Z=8, V=2, M=16: bubble fraction ≈ 7/39 ≈ 18%, compared to 7/23 ≈ 30% for non-interleaved.

```
Without interleaving (V=1):
  GPU0: F0 F1 F2 F3  [idle]  B3 B2 B1 B0
  GPU1:     F0 F1 F2 F3  B3 B2 B1  [idle] B0

With interleaving (V=2, stages 0 and 4 on GPU0):
  GPU0: F0 F4 F1 F5 F2 F6 F3 F7 B7 B3 B6 B2 B5 B1 B4 B0
```

The interleaved schedule is more complex to implement but typically worth it for PP degree > 4.

### Setting Up Megatron-LM

Megatron-LM is typically run as a standalone script (not as a library), with parallelism configured via command-line arguments:

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --micro-batch-size 2 \
    --global-batch-size 1024 \
    --seq-length 4096 \
    --train-iters 500000 \
    --bf16 \
    --use-flash-attn \
    --sequence-parallel \
    --recompute-activations
```

This runs LLaMA 3 70B-scale training with:
- 8-way TP (within node, NVLink)
- 4-way PP (across 4 nodes)
- Remaining GPUs used for data parallelism
- Interleaved 1F1B scheduling (default when PP > 1)
- Sequence parallelism + selective recomputation

### Megatron-DeepSpeed

For CPU offloading with Megatron's 3D parallelism, the **Megatron-DeepSpeed** fork from Microsoft combines both:

```python
# Megatron-DeepSpeed combines Megatron's model parallelism
# with DeepSpeed's ZeRO memory optimizations
deepspeed.initialize(
    model=megatron_model,
    config=deepspeed_config,
)
```

This is what Microsoft used to train Turing-NLG and the Phi series. It is the go-to for 3D parallelism with CPU offload, but at the cost of significant additional complexity.

## Choosing a Framework

The decision tree is roughly:

```
Does the model fit on 1 GPU with FSDP2?
├── Yes → Use vanilla PyTorch (DDP or FSDP2 + DTensor TP)
└── No → How large?
    ├── < 100B params → PyTorch FSDP2 + DTensor TP (8-way NVLink TP, FSDP across nodes)
    ├── 100B–1T params → Megatron-LM (3D: TP + PP + DP, all within one framework)
    └── Need CPU offload or NVMe? → DeepSpeed ZeRO-Infinity or Megatron-DeepSpeed
```

For most practitioners training models in the 1B–70B range, **PyTorch FSDP2 with DTensor TP** covers 90% of cases with the least friction. Move to Megatron-LM when:
- The cluster has > 1,000 GPUs and 3D parallelism is necessary
- PP is needed to reduce peak memory beyond what TP+FSDP achieves
- You're building on existing Megatron-LM infrastructure

Move to DeepSpeed when:
- CPU offloading is required (model too large for GPU memory even with ZeRO-3)
- You're integrating with HuggingFace Transformers (excellent DeepSpeed support via `Trainer`)

## Key Takeaways

- **ZeRO stages 1/2/3** map to saving optimizer state / gradients / parameters respectively. ZeRO-3 is equivalent to FSDP; prefer PyTorch FSDP2 for new projects.

- **Megatron-LM** provides the reference implementation of TP + PP + SP for large-scale training. Use it when training at the 100B+ parameter scale or when building on NVIDIA's infrastructure.

- **Sequence parallelism** shards activations across the sequence dimension outside matmuls, reducing memory proportional to TP degree. It is nearly free in communication cost and should always be enabled with TP.

- **Interleaved 1F1B** (V > 1 virtual stages) significantly reduces pipeline bubble overhead when PP degree is high. Enable it in Megatron-LM with `--num-layers-per-virtual-pipeline-stage`.

- **CPU offloading (ZeRO-Infinity)** is a last resort — it is 50× slower than NVLink. Exhaust all GPU-only options (ZeRO-3, gradient checkpointing, smaller batch size) before offloading to CPU.
