"""Correctness tests for the prefill-style attention kernels.

Small shapes on purpose: these are correctness tests, not benchmarks.
seq_len values include non-multiples of the 64-wide tiles to exercise
boundary handling.
"""
import pytest
import torch

from _utils import (assert_close, ref_attention, ref_gqa, ref_lse, ref_mla,
                    ref_rope_attention, require, rope_tables)

DTYPES = [torch.float16, torch.bfloat16]
HEAD_DIMS = [64, 128]
SEQ_LENS = [64, 100, 256]


def _qkv(b, h, s, d, dtype, hkv=None):
    hkv = hkv or h
    q = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, hkv, s, d, device="cuda", dtype=dtype)
    return q, k, v


# ----------------------------------------------------------------------------
# FlashAttention-2 / FlashAttention-3 forward (same API)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("fn_name", ["flash_attention_forward", "flash_attention_v3_hopper"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_forward(ext, fn_name, dtype, head_dim, seq_len, is_causal):
    fn = require(ext, fn_name)
    q, k, v = _qkv(2, 4, seq_len, head_dim, dtype)
    out = fn(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    assert_close(out, ref_attention(q, k, v, is_causal), dtype, fn_name)


# ----------------------------------------------------------------------------
# GQA
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("hq,hkv", [(8, 2), (8, 8), (4, 1)])
@pytest.mark.parametrize("is_causal", [False, True])
def test_gqa(ext, dtype, head_dim, hq, hkv, is_causal):
    fn = require(ext, "fused_gqa_forward")
    q, k, v = _qkv(1, hq, 100, head_dim, dtype, hkv=hkv)
    out = fn(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    assert_close(out, ref_gqa(q, k, v, is_causal), dtype, "gqa")


# ----------------------------------------------------------------------------
# Fused RoPE + attention
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_rope_attention(ext, dtype, head_dim, is_causal):
    fn = require(ext, "fused_rope_flash_attention")
    s = 100
    q, k, v = _qkv(1, 4, s, head_dim, dtype)
    cos, sin = rope_tables(s, head_dim, "cuda")
    out = fn(q, k, v, cos, sin, is_causal=is_causal)
    torch.cuda.synchronize()
    assert_close(out, ref_rope_attention(q, k, v, cos, sin, is_causal), dtype, "rope")


# ----------------------------------------------------------------------------
# Decode (q_len = 1, contiguous KV)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("kv_len", [1, 63, 64, 1000])
def test_decode(ext, dtype, head_dim, kv_len):
    fn = require(ext, "async_flash_decode")
    q = torch.randn(2, 4, 1, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, kv_len, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(2, 4, kv_len, head_dim, device="cuda", dtype=dtype)
    out = fn(q, k, v)
    torch.cuda.synchronize()
    assert_close(out, ref_attention(q, k, v), dtype, "decode")


# ----------------------------------------------------------------------------
# MLA
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("is_causal", [False, True])
def test_mla(ext, dtype, head_dim, is_causal):
    fn = require(ext, "fused_mla_forward")
    b, h, s, latent = 1, 4, 100, 64
    q = torch.randn(b, h, s, head_dim, device="cuda", dtype=dtype)
    c_kv = torch.randn(b, s, latent, device="cuda", dtype=dtype)
    # Scale weights so K/V have ~unit variance (keeps softmax well-conditioned).
    w_uk = (torch.randn(h, head_dim, latent, device="cuda") / latent ** 0.5).to(dtype)
    w_uv = (torch.randn(h, head_dim, latent, device="cuda") / latent ** 0.5).to(dtype)
    out = fn(q, c_kv, w_uk, w_uv, is_causal=is_causal)
    torch.cuda.synchronize()
    assert_close(out, ref_mla(q, c_kv, w_uk, w_uv, is_causal), dtype, "mla")


# ----------------------------------------------------------------------------
# Backward
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("seq_len", [64, 100])
@pytest.mark.parametrize("is_causal", [False, True])
def test_backward(ext, dtype, head_dim, seq_len, is_causal):
    fn = require(ext, "flash_attention_backward")
    q, k, v = _qkv(1, 2, seq_len, head_dim, dtype)
    dout = torch.randn_like(q)

    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    out_ref = ref_attention(qf, kf, vf, is_causal)
    out_ref.backward(dout.float())

    out = out_ref.detach().to(dtype)
    lse = ref_lse(q, k, is_causal)
    dq, dk, dv = fn(dout, q, k, v, out, lse, is_causal=is_causal)
    torch.cuda.synchronize()

    tol = (5e-2, 5e-2)  # gradients accumulate more rounding error
    assert_close(dq, qf.grad, dtype, "dq", tol)
    assert_close(dk, kf.grad, dtype, "dk", tol)
    assert_close(dv, vf.grad, dtype, "dv", tol)
