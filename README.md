# High-Performance CUDA & Hopper sm_90 Acceleration Library for LLMs

A modern, production-grade CUDA library implementing advanced GPU acceleration kernels for Large Language Models (LLMs) and Transformer architectures. Designed with **Tile IR**, **Tensor Memory Accelerator (TMA)** asynchronous pipelines, and in-register numerical stabilization.

---

## 🚀 Key Features & Kernel Architecture

### 1. **Hopper (sm_90) FlashAttention-3**
- **Warp-Specialized Producer-Consumer Architecture:**
  - **Warp 0 (Producer):** Issues asynchronous bulk copies directly from HBM to SMEM using `cp.async` and commits to pipeline.
  - **Warps 1–3 (Consumers):** Concurrently execute matrix-multiply and Online Softmax across staged tiles.
- **Hardware Asynchronous Barriers (`mbarrier`):** Fine-grained barrier synchronization without coarse, blocking `__syncthreads()`.

### 2. **FlashAttention-2 (Forward & Backward)**
- Double-buffered asynchronous `cp.async` staging overlapping global HBM fetches with in-register math.
- In-register Online Softmax with running max $m_i$ and normalizer $l_i$.
- Exact analytical backward pass with SRAM recomputation.

### 3. **PagedAttention (vLLM Compatible) & FP8 Quantization**
- Virtual memory paging over non-contiguous physical pages for zero memory fragmentation.
- **FP8-Quantized KV-Cache:** 50% to 75% HBM memory savings with on-the-fly dequantization directly inside SRAM and FP32 register accumulators.

### 4. **Multi-Head Latent Attention (MLA - DeepSeek V2/V3)**
- Low-rank compressed KV-cache ($C_{KV}$) reducing KV footprint by 5x–8x.
- Double-buffered asynchronous projection in SRAM.

### 5. **Grouped-Query Attention (GQA) & Fused RoPE**
- Dynamic multi-head broadcasting.
- Rotary Position Embedding (RoPE) fused in-SRAM without intermediate HBM writes.

### 6. **Fully-Fused Transformer MegaBlock**
- End-to-end fused block execution:
  $$\text{RMSNorm}_1 \rightarrow \text{RoPE} \rightarrow \text{GQA Attention} \rightarrow \text{Res}_1 \rightarrow \text{RMSNorm}_2 \rightarrow \text{SwiGLU FFN} \rightarrow \text{Res}_2$$
- Zero intermediate HBM roundtrips.

---

## 📦 Building & Installation

Requires a CUDA GPU (Ampere `sm_80`, Ada `sm_89`, Hopper `sm_90` or newer):

```bash
pip install -e .
```

Or build inplace:
```bash
python setup.py build_ext --inplace
```

## 📊 Benchmarking

Run the comprehensive PyTorch comparison test suite:

```bash
python benchmark_pytorch.py
```
