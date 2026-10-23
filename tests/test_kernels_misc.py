"""Correctness tests for paged / FP8-paged attention, softmax and MegaBlock."""
import pytest
import torch
import torch.nn.functional as F

from _utils import (assert_close, make_paged_cache, ref_decode, ref_megablock,
                    require, rope_tables)

DTYPES = [torch.float16, torch.bfloat16]
PAGE = 16


def _paged_inputs(dtype, head_dim, context_lens):
    n, h = len(context_lens), 4
    t = max(context_lens)
    q = torch.randn(n, h, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(n, h, t, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(n, h, t, head_dim, device="cuda", dtype=dtype)
    lens = torch.tensor(context_lens, device="cuda", dtype=torch.int32)
    return q, k, v, lens


# ----------------------------------------------------------------------------
# PagedAttention (fp16/bf16 cache)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("context_lens", [[1, 16, 17], [300, 129, 64]])
def test_paged_attention(ext, dtype, head_dim, context_lens):
    fn = require(ext, "paged_attention_forward")
    q, k, v, lens = _paged_inputs(dtype, head_dim, context_lens)
    kc, vc, bt = make_paged_cache(k, v, PAGE)
    out = fn(q, kc, vc, bt, lens)
    torch.cuda.synchronize()
    assert_close(out, ref_decode(q, k, v, lens), dtype, "paged")


# ----------------------------------------------------------------------------
# FP8 PagedAttention – tests the *intended* FP8 (E4M3) semantics.
# Expected to fail until Phase 1c replaces the int8 cast with real FP8 decode.
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch lacks float8_e4m3fn")
@pytest.mark.xfail(reason="Fas 1c: kernel interprets bytes as int8, not FP8-E4M3", strict=False)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fp8_paged_attention_e4m3(ext, dtype, head_dim):
    fn = require(ext, "fp8_paged_attention_forward")
    q, k, v, lens = _paged_inputs(dtype, head_dim, [300, 129, 64])
    h = q.shape[1]

    # Per-head scales so values fill the E4M3 range (max 448).
    k_scale = (k.float().abs().amax(dim=(0, 2, 3)) / 448.0).clamp_min(1e-8)
    v_scale = (v.float().abs().amax(dim=(0, 2, 3)) / 448.0).clamp_min(1e-8)
    k8 = (k.float() / k_scale.view(1, h, 1, 1)).to(torch.float8_e4m3fn)
    v8 = (v.float() / v_scale.view(1, h, 1, 1)).to(torch.float8_e4m3fn)

    # Reference uses the *dequantized* values -> isolates kernel error from quantization error.
    k_deq = k8.float() * k_scale.view(1, h, 1, 1)
    v_deq = v8.float() * v_scale.view(1, h, 1, 1)

    kc, vc, bt = make_paged_cache(k8.view(torch.uint8), v8.view(torch.uint8), PAGE)
    out = fn(q, kc, vc, k_scale, v_scale, bt, lens)
    torch.cuda.synchronize()
    assert_close(out, ref_decode(q, k_deq, v_deq, lens), dtype, "fp8_paged")


# ----------------------------------------------------------------------------
# Fused online softmax
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32] + DTYPES)
@pytest.mark.parametrize("shape", [(7, 33), (64, 1024), (16, 8195)])
def test_softmax(ext, dtype, shape):
    fn = require(ext, "forward")
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 4
    out = fn(x)
    torch.cuda.synchronize()
    assert_close(out, F.softmax(x.float(), dim=-1), dtype, "softmax",
                 tol=(1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 2e-2))


# ----------------------------------------------------------------------------
# MegaBlock (current weightless semantics – see plan Fas 1e)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [256, 512])
def test_megablock(ext, dtype, hidden):
    fn = require(ext, "fused_llama_megablock")
    b, s = 2, 37
    x = torch.randn(b, s, hidden, device="cuda", dtype=dtype)
    w1 = torch.rand(hidden, device="cuda") + 0.5
    w2 = torch.rand(hidden, device="cuda") + 0.5
    cos, sin = rope_tables(s, 64, "cuda")
    out = fn(x, w1, w2, cos, sin)
    torch.cuda.synchronize()
    assert_close(out, ref_megablock(x, w1, w2, cos, sin), dtype, "megablock")
