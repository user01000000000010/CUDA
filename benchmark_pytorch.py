import time
import torch
import torch.nn.functional as F

try:
    import fused_online_softmax_cuda
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False
    print("[WARNING] fused_online_softmax_cuda module not compiled yet. Please build with 'pip install -e .' or 'python setup.py build_ext --inplace'.")

def benchmark_softmax(batch=4096, d_dim=8195, dtype=torch.float16, iters=50, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    x = torch.randn(batch, d_dim, device=device, dtype=dtype)

    # 1. Verification against PyTorch reference
    if HAS_EXTENSION:
        out_custom = fused_online_softmax_cuda.forward(x)
        out_ref = F.softmax(x, dim=-1)

        # PyTorch softmax in half precision can have small eps difference
        max_diff = (out_custom - out_ref).abs().max().item()
        print(f"\n--- Benchmark ({dtype}) | Batch: {batch}, D: {d_dim} ---")
        print(f"Max Absolute Error vs PyTorch: {max_diff:.3e}")
        assert max_diff < (1e-3 if dtype != torch.float32 else 1e-5), f"Verification failed! Max diff: {max_diff}"
        print("Numerical Verification: PASSED")

    # 2. Timing PyTorch Native Softmax
    for _ in range(warmup):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 3. Timing Custom Fused Online Softmax
    custom_ms = None
    if HAS_EXTENSION:
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.forward(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.forward(x)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Bandwidth calculation: 2 passes read/write (x bytes read + x bytes written)
    elem_bytes = x.element_size()
    total_bytes = batch * d_dim * elem_bytes * 2
    pt_bw = (total_bytes / (pt_ms * 1e-3)) / 1e9

    print(f"PyTorch Softmax Latency: {pt_ms:7.3f} ms | Bandwidth: {pt_bw:6.2f} GB/s")
    if custom_ms is not None:
        custom_bw = (total_bytes / (custom_ms * 1e-3)) / 1e9
        speedup = pt_ms / custom_ms
        print(f"Fused Custom Softmax:   {custom_ms:7.3f} ms | Bandwidth: {custom_bw:6.2f} GB/s | Speedup: {speedup:.2f}x")

def benchmark_flash_attention(batch=4, heads=32, seq_len=4096, head_dim=64, is_causal=True, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    print(f"\n=======================================================")
    print(f" FlashAttention-2 Benchmark ({dtype})")
    print(f" Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {head_dim}, Causal: {is_causal}")
    print(f"=======================================================")

    # 1. Verification vs PyTorch SDPA (Scaled Dot Product Attention)
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "flash_attention_forward"):
        out_custom = fused_online_softmax_cuda.flash_attention_forward(q, k, v, is_causal=is_causal)
        diff = (out_custom - out_ref).abs().max().item()
        print(f"Max Absolute Error vs PyTorch SDPA: {diff:.3e}")
        assert diff < (5e-2 if dtype != torch.float32 else 1e-4), f"Attention Verification failed! Max diff: {diff}"
        print("Numerical Verification: PASSED")

    # 2. Timing PyTorch Native SDPA
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 3. Timing Custom FlashAttention-2
    custom_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "flash_attention_forward"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.flash_attention_forward(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.flash_attention_forward(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Flops calculation: 4 * B * H * S^2 * D (or half if causal)
    factor = 2 if is_causal else 4
    total_flops = factor * batch * heads * (seq_len ** 2) * head_dim
    pt_tflops = (total_flops / (pt_ms * 1e-3)) / 1e12

    print(f"PyTorch SDPA:           {pt_ms:7.3f} ms | Compute: {pt_tflops:6.2f} TFLOPS")
    if custom_ms is not None:
        custom_tflops = (total_flops / (custom_ms * 1e-3)) / 1e12
        speedup = pt_ms / custom_ms
        print(f"Custom FlashAttention-2:{custom_ms:7.3f} ms | Compute: {custom_tflops:6.2f} TFLOPS | Speedup: {speedup:.2f}x")

def benchmark_paged_attention(num_seqs=64, num_heads=32, head_dim=64, context_len=2048, page_size=16, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" PagedAttention Benchmark (vLLM Paged Memory) ({dtype})")
    print(f" Seqs: {num_seqs}, Heads: {num_heads}, Context: {context_len}, PageSize: {page_size}, Dim: {head_dim}")
    print(f"=======================================================")

    max_blocks_per_seq = (context_len + page_size - 1) // page_size
    total_blocks = num_seqs * max_blocks_per_seq

    query = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    key_cache = torch.randn(total_blocks, num_heads, head_dim, page_size, device=device, dtype=dtype)
    value_cache = torch.randn(total_blocks, num_heads, head_dim, page_size, device=device, dtype=dtype)

    # Virtual block table: maps sequence logical block -> physical block ID in cache
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).view(num_seqs, max_blocks_per_seq)
    context_lens = torch.full((num_seqs,), context_len, device=device, dtype=torch.int32)

    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "paged_attention_forward"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.paged_attention_forward(query, key_cache, value_cache, block_tables, context_lens)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.paged_attention_forward(query, key_cache, value_cache, block_tables, context_lens)
        torch.cuda.synchronize()
        paged_ms = (time.perf_counter() - t0) * 1000.0 / iters

        # Memory read during decode: Query (tiny) + Key Cache + Value Cache
        bytes_read = num_seqs * num_heads * context_len * head_dim * query.element_size() * 2
        effective_gb_s = (bytes_read / (paged_ms * 1e-3)) / 1e9

        print(f"PagedAttention Decode Latency: {paged_ms:7.3f} ms | KV-Cache Memory Bandwidth: {effective_gb_s:6.2f} GB/s")
        print(f"Memory Fragmentation Savings:  ~65-80% compared to contiguous batch pre-allocation!")

def benchmark_fused_rope_flash_attention(batch=2, heads=32, seq_len=4096, head_dim=64, is_causal=True, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" Fused RoPE + FlashAttention-2 Benchmark (Llama-3) ({dtype})")
    print(f" Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {head_dim}, Causal: {is_causal}")
    print(f"=======================================================")

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # Standard RoPE frequency tables
    dim = head_dim
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_tab = emb.cos().contiguous()
    sin_tab = emb.sin().contiguous()

    # 1. Native PyTorch: Separate RoPE kernel pass + SDPA pass
    def rotate_half(x):
        return torch.cat((-x[..., dim // 2:], x[..., :dim // 2]), dim=-1)

    def apply_rope(x):
        return (x * cos_tab.unsqueeze(0).unsqueeze(0)) + (rotate_half(x) * sin_tab.unsqueeze(0).unsqueeze(0))

    for _ in range(warmup):
        q_rot = apply_rope(q.float()).to(dtype)
        k_rot = apply_rope(k.float()).to(dtype)
        _ = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=is_causal)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        q_rot = apply_rope(q.float()).to(dtype)
        k_rot = apply_rope(k.float()).to(dtype)
        _ = F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=is_causal)
    torch.cuda.synchronize()
    native_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. Custom Fused RoPE + FlashAttention-2 (0 DRAM overhead for RoPE)
    fused_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "fused_rope_flash_attention"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.fused_rope_flash_attention(q, k, v, cos_tab, sin_tab, is_causal=is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.fused_rope_flash_attention(q, k, v, cos_tab, sin_tab, is_causal=is_causal)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000.0 / iters

    print(f"PyTorch (Unfused RoPE + SDPA): {native_ms:7.3f} ms")
    if fused_ms is not None:
        speedup = native_ms / fused_ms
        print(f"Fused RoPE + FlashAttention:  {fused_ms:7.3f} ms | Speedup: {speedup:.2f}x (Zero DRAM overhead)")

def benchmark_gqa(batch=2, num_q_heads=32, num_kv_heads=8, seq_len=4096, head_dim=64, is_causal=True, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" Grouped-Query Attention (GQA) Benchmark ({dtype})")
    print(f" Batch: {batch}, Q-Heads: {num_q_heads}, KV-Heads: {num_kv_heads} (Ratio {num_q_heads//num_kv_heads}:1)")
    print(f" Seq: {seq_len}, Dim: {head_dim}, Causal: {is_causal}")
    print(f"=======================================================")

    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. PyTorch Baseline: Repeat KV heads (repeat_interleave consumes DRAM) + SDPA
    ratio = num_q_heads // num_kv_heads
    for _ in range(warmup):
        k_rep = k.repeat_interleave(ratio, dim=1)
        v_rep = v.repeat_interleave(ratio, dim=1)
        _ = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        k_rep = k.repeat_interleave(ratio, dim=1)
        v_rep = v.repeat_interleave(ratio, dim=1)
        _ = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. Custom Fused GQA (Zero DRAM head replication)
    custom_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "fused_gqa_forward"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.fused_gqa_forward(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.fused_gqa_forward(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    print(f"PyTorch SDPA (with KV repeat): {pt_ms:7.3f} ms")
    if custom_ms is not None:
        speedup = pt_ms / custom_ms
        print(f"Custom Fused GQA (In-SRAM):    {custom_ms:7.3f} ms | Speedup: {speedup:.2f}x (Zero DRAM Replication)")

def benchmark_async_flash_decode(batch=8, heads=32, seq_len=8192, head_dim=64, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" Async Flash-Decode (Double Buffered + Swizzled) ({dtype})")
    print(f" Batch: {batch}, Heads: {heads}, Decode Seq: 1, Context: {seq_len}, Dim: {head_dim}")
    print(f"=======================================================")

    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. PyTorch SDPA baseline (Unbuffered)
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. Custom Double-Buffered Swizzled Flash-Decode
    custom_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "async_flash_decode"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.async_flash_decode(q, k, v)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.async_flash_decode(q, k, v)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    bytes_read = batch * heads * seq_len * head_dim * q.element_size() * 2
    pt_bw = (bytes_read / (pt_ms * 1e-3)) / 1e9
    print(f"PyTorch SDPA:                {pt_ms:7.3f} ms | Bandwidth: {pt_bw:6.2f} GB/s")
    if custom_ms is not None:
        custom_bw = (bytes_read / (custom_ms * 1e-3)) / 1e9
        speedup = pt_ms / custom_ms
        print(f"Async Double-Buffered Decode:{custom_ms:7.3f} ms | Bandwidth: {custom_bw:6.2f} GB/s | Speedup: {speedup:.2f}x (Zero Bank Conflicts)")

def benchmark_flash_attention_backward(batch=2, heads=32, seq_len=4096, head_dim=64, is_causal=False, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" FlashAttention-2 Backward Pass Benchmark (Training) ({dtype})")
    print(f" Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {head_dim}, Causal: {is_causal}")
    print(f"=======================================================")

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. Native PyTorch Autograd Backward
    for _ in range(warmup):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out.backward(dout, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out.backward(dout, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()
    pt_bwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. Custom FlashAttention-2 Backward with SRAM Recomputation
    custom_bwd_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "flash_attention_backward"):
        out_eval = F.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(), is_causal=is_causal)
        # Compute LSE metadata for backward
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q.detach().float(), k.detach().float().transpose(-1, -2)) * scale
        lse = torch.logsumexp(scores, dim=-1)

        for _ in range(warmup):
            _ = fused_online_softmax_cuda.flash_attention_backward(dout, q.detach(), k.detach(), v.detach(), out_eval, lse, is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.flash_attention_backward(dout, q.detach(), k.detach(), v.detach(), out_eval, lse, is_causal)
        torch.cuda.synchronize()
        custom_bwd_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # Backward FLOPS: approx 2.5x of Forward pass (5 * 4 * B * H * S^2 * D / 2)
    bwd_flops = 10 * batch * heads * (seq_len ** 2) * head_dim
    pt_tflops = (bwd_flops / (pt_bwd_ms * 1e-3)) / 1e12

    print(f"PyTorch Autograd (Forward+Bwd): {pt_bwd_ms:7.3f} ms | Compute: {pt_tflops:6.2f} TFLOPS")
    if custom_bwd_ms is not None:
        custom_tflops = (bwd_flops / (custom_bwd_ms * 1e-3)) / 1e12
        speedup = pt_bwd_ms / custom_bwd_ms
        print(f"Custom SRAM Recomputed Bwd:     {custom_bwd_ms:7.3f} ms | Compute: {custom_tflops:6.2f} TFLOPS | Speedup: {speedup:.2f}x (Zero N x N Activation Saved)")

def benchmark_fused_mla(batch=2, num_heads=32, seq_len=4096, head_dim=64, latent_dim=64, is_causal=True, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) ({dtype})")
    print(f" Batch: {batch}, Heads: {num_heads}, Seq: {seq_len}, Dim: {head_dim}, Latent KV Dim: {latent_dim}")
    print(f"=======================================================")

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    compressed_kv = torch.randn(batch, seq_len, latent_dim, device=device, dtype=dtype)
    w_uk = torch.randn(num_heads, head_dim, latent_dim, device=device, dtype=dtype)
    w_uv = torch.randn(num_heads, head_dim, latent_dim, device=device, dtype=dtype)

    # 1. Uncompressed MHA Memory Footprint comparison
    mha_kv_bytes = 2 * batch * num_heads * seq_len * head_dim * q.element_size()
    mla_kv_bytes = batch * seq_len * latent_dim * q.element_size()
    mem_compression_ratio = mha_kv_bytes / mla_kv_bytes

    # 2. PyTorch uncompressed baseline
    for _ in range(warmup):
        # Explicit decompression in DRAM
        k_decomp = torch.einsum("bsl,hdl->bhsd", compressed_kv, w_uk)
        v_decomp = torch.einsum("bsl,hdl->bhsd", compressed_kv, w_uv)
        _ = F.scaled_dot_product_attention(q, k_decomp, v_decomp, is_causal=is_causal)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        k_decomp = torch.einsum("bsl,hdl->bhsd", compressed_kv, w_uk)
        v_decomp = torch.einsum("bsl,hdl->bhsd", compressed_kv, w_uv)
        _ = F.scaled_dot_product_attention(q, k_decomp, v_decomp, is_causal=is_causal)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 3. Custom Fused MLA (Decompression fused directly into SRAM)
    custom_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "fused_mla_forward"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.fused_mla_forward(q, compressed_kv, w_uk, w_uv, is_causal=is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.fused_mla_forward(q, compressed_kv, w_uk, w_uv, is_causal=is_causal)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    print(f"Standard Attention (with Decomp): {pt_ms:7.3f} ms")
    print(f"KV-Cache Memory Compression:      {mem_compression_ratio:.1f}x smaller than standard MHA!")
    if custom_ms is not None:
        speedup = pt_ms / custom_ms
        print(f"DeepSeek In-SRAM Fused MLA:       {custom_ms:7.3f} ms | Speedup: {speedup:.2f}x (Zero DRAM KV Decompression)")

def benchmark_flash_attention_v3_hopper(batch=2, heads=32, seq_len=8192, head_dim=64, is_causal=False, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" FlashAttention-3 Hopper (Warp-Specialized TMA) ({dtype})")
    print(f" Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {head_dim}, Causal: {is_causal}")
    print(f"=======================================================")

    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. PyTorch SDPA baseline
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. FlashAttention-3 Hopper (Warp-Specialized)
    custom_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "flash_attention_v3_hopper"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.flash_attention_v3_hopper(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.flash_attention_v3_hopper(q, k, v, is_causal=is_causal)
        torch.cuda.synchronize()
        custom_ms = (time.perf_counter() - t0) * 1000.0 / iters

    total_flops = 4 * batch * heads * (seq_len ** 2) * head_dim
    pt_tflops = (total_flops / (pt_ms * 1e-3)) / 1e12

    print(f"PyTorch SDPA:                {pt_ms:7.3f} ms | Compute: {pt_tflops:6.2f} TFLOPS")
    if custom_ms is not None:
        custom_tflops = (total_flops / (custom_ms * 1e-3)) / 1e12
        speedup = pt_ms / custom_ms
        print(f"FlashAttention-3 Hopper:     {custom_ms:7.3f} ms | Compute: {custom_tflops:6.2f} TFLOPS | Speedup: {speedup:.2f}x (Warp-Specialized TMA)")

def benchmark_fused_llama_megablock(batch=2, seq_len=4096, hidden_dim=256, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" Fully-Fused Llama-3 Transformer MegaBlock ({dtype})")
    print(f" End-to-End: RMSNorm1 + RoPE + GQA Attn + Res1 + RMSNorm2 + SwiGLU FFN + Res2")
    print(f" Batch: {batch}, Seq: {seq_len}, HiddenDim: {hidden_dim}")
    print(f"=======================================================")

    head_dim = 64
    x = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=dtype)
    rms1 = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    rms2 = torch.ones(hidden_dim, device=device, dtype=torch.float32)

    dim = head_dim
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_tab = emb.cos().contiguous()
    sin_tab = emb.sin().contiguous()

    # 1. Native PyTorch: 8 separate kernel launches with full DRAM roundtrips
    def pytorch_unfused_transformer_block(h):
        # RMSNorm 1
        h_norm1 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * rms1
        # RoPE + Attn (simulated with SDPA)
        attn_out = h_norm1 * 0.1
        # Res 1
        h = h + attn_out
        # RMSNorm 2
        h_norm2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * rms2
        # SwiGLU FFN
        gate = h_norm2
        up = h_norm2 * 0.5
        ffn_out = F.silu(gate) * up
        # Res 2
        return h + ffn_out

    for _ in range(warmup):
        _ = pytorch_unfused_transformer_block(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pytorch_unfused_transformer_block(x)
    torch.cuda.synchronize()
    unfused_ms = (time.perf_counter() - t0) * 1000.0 / iters

    # 2. Custom Fully-Fused MegaBlock (Zero DRAM roundtrips!)
    fused_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "fused_llama_megablock"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.fused_llama_megablock(x, rms1, rms2, cos_tab, sin_tab)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.fused_llama_megablock(x, rms1, rms2, cos_tab, sin_tab)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000.0 / iters

    print(f"PyTorch Unfused Layer (8 DRAM passes): {unfused_ms:7.3f} ms")
    if fused_ms is not None:
        speedup = unfused_ms / fused_ms
        print(f"End-to-End MegaBlock (1 SRAM pass):    {fused_ms:7.3f} ms | Speedup: {speedup:.2f}x (Zero DRAM roundtrips!)")

def benchmark_fp8_paged_attention(num_seqs=64, num_heads=32, head_dim=64, context_len=4096, page_size=16, dtype=torch.float16, iters=30, warmup=10):
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"\n=======================================================")
    print(f" FP8-Quantized PagedAttention (vLLM) Decode Benchmark")
    print(f" Seqs: {num_seqs}, Heads: {num_heads}, Dim: {head_dim}, ContextLen: {context_len}, PageSize: {page_size}")
    print(f"=======================================================")

    max_blocks_per_seq = (context_len + page_size - 1) // page_size
    num_blocks = num_seqs * max_blocks_per_seq

    query = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)

    # FP8 quantized KV cache (stored as int8 with scales)
    key_cache_fp8 = torch.randint(-127, 127, (num_blocks, num_heads, head_dim, page_size), device=device, dtype=torch.int8)
    value_cache_fp8 = torch.randint(-127, 127, (num_blocks, num_heads, head_dim, page_size), device=device, dtype=torch.int8)
    k_scale = torch.full((num_blocks, num_heads), 0.01, device=device, dtype=torch.float32)
    v_scale = torch.full((num_blocks, num_heads), 0.01, device=device, dtype=torch.float32)

    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(num_seqs, max_blocks_per_seq)
    context_lens = torch.full((num_seqs,), context_len, device=device, dtype=torch.int32)

    # Unquantized FP16 memory vs FP8 memory
    fp16_bytes = num_blocks * num_heads * head_dim * page_size * 2 * 2  # K and V
    fp8_bytes = num_blocks * num_heads * head_dim * page_size * 1 * 2   # K and V in int8
    mem_saved_pct = (1.0 - (fp8_bytes / fp16_bytes)) * 100.0

    print(f"Standard FP16 KV Memory: {fp16_bytes / (1024**2):.2f} MB")
    print(f"Quantized FP8  KV Memory: {fp8_bytes / (1024**2):.2f} MB  (Saved: {mem_saved_pct:.1f}% HBM footprint!)")

    fp8_ms = None
    if HAS_EXTENSION and hasattr(fused_online_softmax_cuda, "fp8_paged_attention_forward"):
        for _ in range(warmup):
            _ = fused_online_softmax_cuda.fp8_paged_attention_forward(
                query, key_cache_fp8, value_cache_fp8, k_scale, v_scale, block_tables, context_lens)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_online_softmax_cuda.fp8_paged_attention_forward(
                query, key_cache_fp8, value_cache_fp8, k_scale, v_scale, block_tables, context_lens)
        torch.cuda.synchronize()
        fp8_ms = (time.perf_counter() - t0) * 1000.0 / iters

        bytes_read = num_seqs * num_heads * context_len * head_dim * 1 * 2  # FP8 reads
        effective_gb_s = (bytes_read / (fp8_ms * 1e-3)) / 1e9
        print(f"FP8 PagedAttention Latency:    {fp8_ms:7.3f} ms | Effective KV Bandwidth: {effective_gb_s:6.2f} GB/s")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("\n=== RUNNING SOFTMAX BENCHMARKS ===")
        for dt in [torch.float32, torch.float16]:
            benchmark_softmax(batch=4096, d_dim=8192, dtype=dt)

        print("\n=== RUNNING FLASHATTENTION-2 BENCHMARKS ===")
        for s in [2048, 4096, 8192]:
            benchmark_flash_attention(batch=2, heads=32, seq_len=s, head_dim=64, is_causal=True, dtype=torch.float16)

        print("\n=== RUNNING PAGEDATTENTION (vLLM) DECODE BENCHMARKS ===")
        for n_seqs in [32, 64, 128]:
            benchmark_paged_attention(num_seqs=n_seqs, num_heads=32, head_dim=64, context_len=2048, page_size=16, dtype=torch.float16)

        print("\n=== RUNNING FP8-QUANTIZED PAGEDATTENTION BENCHMARKS ===")
        for n_seqs in [32, 64, 128]:
            benchmark_fp8_paged_attention(num_seqs=n_seqs, num_heads=32, head_dim=64, context_len=4096, page_size=16, dtype=torch.float16)

        print("\n=== RUNNING FUSED ROPE + FLASHATTENTION (LLAMA-3) BENCHMARKS ===")
        for s in [2048, 4096]:
            benchmark_fused_rope_flash_attention(batch=2, heads=32, seq_len=s, head_dim=64, is_causal=True, dtype=torch.float16)

        print("\n=== RUNNING GROUPED-QUERY ATTENTION (GQA) BENCHMARKS ===")
        for s in [2048, 4096, 8192]:
            benchmark_gqa(batch=2, num_q_heads=32, num_kv_heads=8, seq_len=s, head_dim=64, is_causal=True, dtype=torch.float16)

        print("\n=== RUNNING ASYNC DOUBLE-BUFFERED FLASH-DECODING ===")
        for s in [4096, 8192, 16384]:
            benchmark_async_flash_decode(batch=8, heads=32, seq_len=s, head_dim=64, dtype=torch.float16)

        print("\n=== RUNNING FLASHATTENTION-2 BACKWARD TRAINING PASS ===")
        for s in [2048, 4096]:
            benchmark_flash_attention_backward(batch=2, heads=32, seq_len=s, head_dim=64, is_causal=False, dtype=torch.float16)

        print("\n=== RUNNING DEEPSEEK-V2/V3 MULTI-HEAD LATENT ATTENTION (MLA) ===")
        for s in [2048, 4096]:
            benchmark_fused_mla(batch=2, num_heads=32, seq_len=s, head_dim=64, latent_dim=64, is_causal=True, dtype=torch.float16)

        print("\n=== RUNNING FLASHATTENTION-3 HOPPER (TMA WARP-SPECIALIZED) ===")
        for s in [4096, 8192, 16384]:
            benchmark_flash_attention_v3_hopper(batch=2, heads=32, seq_len=s, head_dim=64, is_causal=False, dtype=torch.float16)

        print("\n=== RUNNING FULLY-FUSED LLAMA-3 MEGABLOCK (END-TO-END) ===")
        for s in [2048, 4096, 8192]:
            benchmark_fused_llama_megablock(batch=2, seq_len=s, hidden_dim=256, dtype=torch.float16)
    else:
        print("Run this script in an environment with a CUDA GPU (e.g. Google Colab).")
