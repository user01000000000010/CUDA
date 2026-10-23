"""Test helpers and fp32 PyTorch reference implementations.

All references compute in float32 and are the *specification* the CUDA
kernels must match.
"""
import math

import pytest
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def require(ext, fn_name):
    """Skip if the kernel was not compiled into this build (KERNELS=...)."""
    if not hasattr(ext, fn_name):
        pytest.skip(f"'{fn_name}' not in this build (set KERNELS=... in setup.py)")
    return getattr(ext, fn_name)


TOL = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def assert_close(out, ref, dtype, name="", tol=None):
    atol, rtol = tol or TOL[dtype]
    out_f, ref_f = out.float(), ref.float()
    assert out_f.shape == ref_f.shape, f"{name}: shape {out_f.shape} != {ref_f.shape}"
    assert torch.isfinite(out_f).all(), f"{name}: non-finite values in output"
    max_err = (out_f - ref_f).abs().max().item()
    assert torch.allclose(out_f, ref_f, atol=atol, rtol=rtol), (
        f"{name}: max abs error {max_err:.3e} (atol={atol}, rtol={rtol})"
    )


def default_scale(head_dim):
    return 1.0 / math.sqrt(head_dim)


# ----------------------------------------------------------------------------
# Attention references
# ----------------------------------------------------------------------------
def ref_attention(q, k, v, is_causal=False, scale=None):
    """q,k,v: [B, H, S, D] -> [B, H, S, D] (fp32)."""
    scale = default_scale(q.shape[-1]) if scale is None else scale
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=is_causal, scale=scale
    )


def ref_lse(q, k, is_causal=False, scale=None):
    """Natural-log logsumexp of scaled (and masked) scores: [B, H, S]."""
    scale = default_scale(q.shape[-1]) if scale is None else scale
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if is_causal:
        sq, sk = s.shape[-2], s.shape[-1]
        mask = torch.ones(sq, sk, dtype=torch.bool, device=s.device).tril()
        s = s.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(s, dim=-1)


def ref_gqa(q, k, v, is_causal=True, scale=None):
    """q: [B, Hq, S, D], k/v: [B, Hkv, S, D]."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    return ref_attention(q, k, v, is_causal=is_causal, scale=scale)


def rope_tables(seq_len, head_dim, device, base=10000.0):
    """cos/sin tables [S, D] in the rotate_half (Llama/HF) convention."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    emb = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
    return emb.cos().contiguous(), emb.sin().contiguous()


def apply_rope(x, cos, sin):
    """x: [..., S, D], cos/sin: [S, D]."""
    d = x.shape[-1]
    x = x.float()
    rot = torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)
    return x * cos + rot * sin


def ref_rope_attention(q, k, v, cos, sin, is_causal=True, scale=None):
    return ref_attention(apply_rope(q, cos, sin), apply_rope(k, cos, sin), v,
                         is_causal=is_causal, scale=scale)


def ref_mla(q, c_kv, w_uk, w_uv, is_causal=True, scale=None):
    """q: [B,H,S,D], c_kv: [B,S,L], w_uk/w_uv: [H,D,L]."""
    k = torch.einsum("bsl,hdl->bhsd", c_kv.float(), w_uk.float())
    v = torch.einsum("bsl,hdl->bhsd", c_kv.float(), w_uv.float())
    return ref_attention(q, k, v, is_causal=is_causal, scale=scale)


# ----------------------------------------------------------------------------
# Paged KV-cache helpers
# Layout (matches kernels): cache[num_blocks, num_heads, head_dim, page_size]
# ----------------------------------------------------------------------------
def make_paged_cache(k, v, page_size, shuffle=True):
    """k, v: [N, H, T, D] contiguous per sequence (T = max context).

    Returns key_cache, value_cache, block_tables (int32).
    Physical blocks are randomly permuted to exercise the indirection.
    """
    n, h, t, d = k.shape
    blocks_per_seq = (t + page_size - 1) // page_size
    pad = blocks_per_seq * page_size - t
    if pad:
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    # [N, H, blocks, page, D] -> [N, blocks, H, D, page]
    kb = k.view(n, h, blocks_per_seq, page_size, d).permute(0, 2, 1, 4, 3)
    vb = v.view(n, h, blocks_per_seq, page_size, d).permute(0, 2, 1, 4, 3)
    num_blocks = n * blocks_per_seq
    kb = kb.reshape(num_blocks, h, d, page_size)
    vb = vb.reshape(num_blocks, h, d, page_size)

    logical = torch.arange(num_blocks, device=k.device)
    perm = torch.randperm(num_blocks, device=k.device) if shuffle else logical
    key_cache = torch.empty_like(kb)
    value_cache = torch.empty_like(vb)
    key_cache[perm] = kb
    value_cache[perm] = vb
    block_tables = perm.view(n, blocks_per_seq).to(torch.int32).contiguous()
    return key_cache.contiguous(), value_cache.contiguous(), block_tables


def ref_decode(q, k, v, context_lens, scale=None):
    """q: [N,H,D], k/v: [N,H,T,D], context_lens: [N] -> [N,H,D]."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for i, L in enumerate(context_lens.tolist()):
        out[i] = ref_attention(q[i:i + 1, :, None, :], k[i:i + 1, :, :L], v[i:i + 1, :, :L],
                               scale=scale)[0, :, 0]
    return out


# ----------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------
def ref_megablock(x, w1, w2, cos, sin, eps=1e-6, scale=None, head_dim=64):
    """Reference of the *current documented* MegaBlock math (no weights):
    RMSNorm1 -> RoPE*scale -> +res -> RMSNorm2 -> silu(x)*0.5x -> +res.
    NOTE: semantics pending decision (make real vs. remove) – see plan Fas 1e.
    """
    scale = default_scale(head_dim) if scale is None else scale
    b, s, hid = x.shape
    xf = x.float()
    n1 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w1.float()
    heads = n1.view(b, s, hid // head_dim, head_dim)
    rot = apply_rope(heads.transpose(1, 2), cos[:s], sin[:s]).transpose(1, 2)
    h1 = xf + (rot * scale).reshape(b, s, hid)
    n2 = h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + eps) * w2.float()
    return h1 + F.silu(n2) * (0.5 * n2)
