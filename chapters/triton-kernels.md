# Kernel Optimization and Triton

PyTorch dispatches to hand-tuned CUDA kernels (cuBLAS, cuDNN) for standard operations. But there are many operations where no hand-tuned kernel exists — or where a fused custom kernel would be substantially faster than calling multiple standard ops sequentially. This is where **Triton** comes in.

Triton is a Python DSL for writing GPU kernels. It lets you express GPU computation at the tile level — each kernel instance processes a tile of data in SRAM — without managing raw CUDA thread hierarchies. Triton generates PTX (GPU assembly) and handles register allocation, memory coalescing, and warp-level primitives automatically.

This chapter covers why custom kernels matter, how Triton works conceptually, and walks through a real kernel: a fused row-wise softmax.

## Why Custom Kernels?

### The Bandwidth-Bound Elementwise Problem

Consider applying ReLU to a 1B-element tensor:

```python
# PyTorch eager (two separate kernel calls)
x = torch.randn(1000, 1000000, device='cuda', dtype=torch.bfloat16)
y = torch.relu(x)
```

Under the hood, PyTorch launches a CUDA kernel that:
1. Reads x from HBM → 2 GB read
2. Computes relu(x) — near-zero compute, just a max
3. Writes y to HBM → 2 GB write

Total: 4 GB of HBM traffic for 1B FLOPs of compute. Arithmetic intensity = 1B / 4G = 0.25 FLOPs/byte — massively memory-bound.

Now consider computing `relu(layer_norm(x))`. In PyTorch eager, this is two separate kernels:
- Kernel 1: reads x, computes layer_norm, writes intermediate result
- Kernel 2: reads intermediate result, computes relu, writes output

Each step reads and writes the full tensor. A **fused kernel** would read x once, compute both operations in SRAM, and write output once — halving HBM traffic.

**Kernel fusion is the primary motivation for custom kernels.** It reduces HBM reads/writes for chains of elementwise or reduction operations.

### When Custom Kernels Are Worth It

- **Fused elementwise chains:** consecutive operations that each read/write the full tensor (layer_norm + activation, attention scaling + masking + softmax)
- **Tiled reductions:** operations like softmax, layer norm, or flash attention that can be done in tiles to avoid materializing large intermediate tensors
- **Novel operations:** operations not in PyTorch's standard library (e.g., FP8 quantization with custom scaling, fused rotary embeddings)

When **not** to write custom kernels:
- The operation is already in cuBLAS or cuDNN — you can't beat their hand-tuned GEMM and attention kernels
- The operation is compute-bound with large matmuls — the kernel launch overhead is negligible
- `torch.compile` can fuse it automatically — check this first

## The Triton Programming Model

Triton programs are written at the **tile level**. Instead of specifying per-thread operations (raw CUDA), you specify operations on tiles — contiguous blocks of data that fit in SRAM.

The key concepts:

- **`tl.program_id(axis)`:** each kernel instance gets a unique ID along each launch axis. Use it to determine which tile this instance processes.
- **`tl.arange(0, BLOCK_SIZE)`:** creates a range of indices within a tile.
- **`tl.load(ptr + offsets, mask=mask)`:** load a tile from HBM with optional masking for boundary handling.
- **`tl.store(ptr + offsets, values, mask=mask)`:** store a tile to HBM.
- **`tl.sum`, `tl.max`, `tl.exp`:** reduction and elementwise operations within SRAM.

The GPU grid is defined at launch time by the number of tiles needed. Each program instance handles one tile; Triton handles SRAM allocation and memory access patterns.

## Worked Example: Fused Row-Wise Softmax

Naive PyTorch softmax on a large matrix:

```python
# Naive: three passes over the data
def naive_softmax(x):
    x_max = x.max(dim=-1, keepdim=True).values   # pass 1: read x
    z = x - x_max                                  # pass 2: read x, x_max, write z
    numerator = torch.exp(z)                       # pass 3: read z, write numerator
    denominator = numerator.sum(dim=-1, keepdim=True)  # pass 4: read numerator
    return numerator / denominator                 # pass 5: read both, write output
```

This reads the input matrix 4 times and writes 3 intermediates. A fused softmax reads input once and writes output once.

Here's the Triton implementation:

```python
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,  # must be a power of 2, >= n_cols
):
    # Each program instance handles one row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # Load the row — mask out-of-bounds elements with -inf
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))

    # Numerically stable softmax: subtract max, exp, normalize
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    # BLOCK_SIZE must be at least n_cols and a power of 2
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    output = torch.empty_like(x)
    softmax_kernel[(n_rows,)](  # launch n_rows program instances
        output, x,
        x.stride(0), output.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

**What happens in SRAM:** each program instance loads its row into SRAM, computes the max (one reduction), subtracts it (elementwise), exponentiates (elementwise), sums (one reduction), and divides (elementwise). One HBM read, one HBM write. No intermediate tensors.

**Performance:** on an A100, the fused softmax is typically 4–5× faster than the naive version for wide matrices, purely from reduced HBM traffic.

## Flash Attention as a Triton Kernel

Flash Attention (Chapter 5, Appendix A) is the canonical example of a tile-based algorithm that avoids materializing the $$O(T^2)$$ attention matrix. Its Triton implementation maintains running statistics (max and sum) in SRAM as it iterates over K/V tiles.

```python
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DHEAD: tl.constexpr,
):
    # Program ID selects: which batch, which head, which Q tile
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Load Q tile into SRAM (stays resident throughout)
    q = tl.load(Q + ...)

    # Running statistics in SRAM
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    # Iterate over K/V tiles (loaded from HBM one tile at a time)
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + ...)
        v = tl.load(V + ...)

        # Compute QK^T for this tile
        qk = tl.dot(q, k)
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Normalize and write output
    acc = acc / l_i[:, None]
    tl.store(Out + ...)
```

The real Flash Attention Triton kernel (from the `flash-attn` library) handles causal masking, GQA (grouped-query attention), and has careful tile size tuning for different head dimensions. The high-level structure above captures the core idea.

**In practice:** use `F.scaled_dot_product_attention` (Chapter 3) — PyTorch dispatches to the Flash Attention Triton or cuDNN kernel automatically on Ampere+. Only write your own attention kernel if you need a custom masking pattern or GQA variant not supported by the PyTorch backend.

## Autotuning

Triton's `@triton.autotune` decorator benchmarks multiple kernel configurations and caches the best one:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_cols'],  # reautotune when this changes
)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, ..., BLOCK_SIZE: tl.constexpr):
    ...
```

The first call with each unique `n_cols` value benchmarks all configs and picks the fastest. Results are cached. This is how PyTorch's `torch.compile` achieves per-hardware kernel tuning without manual effort.

## When to Use Triton vs torch.compile

`torch.compile` (Chapter 15) uses Triton internally to generate fused kernels from PyTorch graph IR. For most use cases, `torch.compile` provides kernel fusion automatically:

```python
# torch.compile will fuse this into a single Triton kernel automatically
@torch.compile
def fused_ops(x):
    return torch.relu(layer_norm(x))
```

Write custom Triton kernels when:
- The operation has a non-trivial tile-level algorithm (Flash Attention, online softmax, fused quantization)
- `torch.compile` fails to fuse or produces suboptimal code for your operation
- You need precise control over memory access patterns or tile sizes
- You're implementing a new operation type (`torch.compile` can only fuse existing ops)

## Key Takeaways

- **Kernel fusion reduces HBM traffic** for chains of elementwise operations. The speedup is proportional to the number of passes eliminated.

- **Triton programs operate at the tile level.** Each program instance loads a tile into SRAM, computes in SRAM, and writes results back. No thread-level indexing or CUDA PTX needed.

- **Flash Attention is the prototypical tile-based algorithm.** It maintains running statistics in SRAM to avoid materializing the $$O(T^2)$$ attention matrix, reducing HBM traffic from $$O(T^2)$$ to $$O(T)$$.

- **Use `torch.compile` before writing custom kernels.** It generates Triton-based fused kernels automatically for most operation chains. Only write custom Triton when `torch.compile` can't handle your algorithm.

- **Autotune tile sizes** per hardware and shape. The optimal BLOCK_SIZE depends on the GPU's SRAM capacity, warp count, and arithmetic intensity of the operation.
