// Host-compiled binding: include ONLY declarations (see kernels_api.h).
// Each kernel is compiled in only when setup.py defines WITH_<KERNEL>.
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <torch/extension.h>

#include "kernels_api.h"

#ifdef WITH_SOFTMAX
// PyTorch bindings for Fused Online Softmax
torch::Tensor fused_online_softmax_forward(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "Input tensor must have at least 2 dimensions");

  auto output = torch::empty_like(input);

  const int d_dim = input.size(-1);
  const int batch_size = input.numel() / d_dim;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Float) {
    dispatch_fused_online_softmax<float>(input.data_ptr<float>(),
                                         output.data_ptr<float>(), batch_size,
                                         d_dim, stream);
  } else if (input.scalar_type() == at::ScalarType::Half) {
    dispatch_fused_online_softmax<__half>(
        reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        d_dim, stream);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fused_online_softmax<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, d_dim, stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for fused_online_softmax. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_SOFTMAX

#ifdef WITH_FA2
// PyTorch bindings for FlashAttention-2 Forward
torch::Tensor flash_attention_v2_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false,
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "Inputs must be contiguous");
  TORCH_CHECK(q.dim() == 4,
              "Expected 4D tensor: [batch, num_heads, seq_len, head_dim]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "Q, K, V must have identical shapes");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 are currently supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_flash_attention_v2<float>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_flash_attention_v2<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_flash_attention_v2<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, softmax_scale, is_causal,
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for FlashAttention. Supported: "
                       "Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_FA2

#ifdef WITH_PAGED
// PyTorch bindings for PagedAttention (vLLM style Virtual Memory Paging)
torch::Tensor paged_attention_forward(
    torch::Tensor query,        // [num_seqs, num_heads, head_dim]
    torch::Tensor key_cache,    // [num_blocks, num_heads, head_dim, page_size]
    torch::Tensor value_cache,  // [num_blocks, num_heads, head_dim, page_size]
    torch::Tensor block_tables, // [num_seqs, max_blocks_per_seq]
    torch::Tensor context_lens, // [num_seqs]
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(block_tables.is_cuda() && context_lens.is_cuda(),
              "Metadata tensors must be on CUDA");
  TORCH_CHECK(query.dim() == 3,
              "Query must be 3D [num_seqs, num_heads, head_dim]");
  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4,
      "KV caches must be 4D [num_blocks, num_heads, head_dim, page_size]");

  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int max_blocks_per_seq = block_tables.size(1);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(query);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (query.scalar_type() == at::ScalarType::Float) {
    dispatch_paged_attention<float>(
        query.data_ptr<float>(), key_cache.data_ptr<float>(),
        value_cache.data_ptr<float>(), block_tables.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(), output.data_ptr<float>(), num_seqs,
        num_heads, head_dim, max_blocks_per_seq, softmax_scale, stream);
  } else if (query.scalar_type() == at::ScalarType::Half) {
    dispatch_paged_attention<__half>(
        reinterpret_cast<const __half *>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(value_cache.data_ptr<at::Half>()),
        block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), num_seqs,
        num_heads, head_dim, max_blocks_per_seq, softmax_scale, stream);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_paged_attention<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(
            key_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(
            value_cache.data_ptr<at::BFloat16>()),
        block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        num_seqs, num_heads, head_dim, max_blocks_per_seq, softmax_scale,
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for PagedAttention. Supported: "
                       "Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_PAGED

#ifdef WITH_ROPE
// PyTorch bindings for Fused RoPE & FlashAttention-2 (Llama-3 Architecture)
torch::Tensor fused_rope_flash_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cos_tab,
    torch::Tensor sin_tab, bool is_causal = true,
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(cos_tab.is_cuda() && sin_tab.is_cuda(),
              "RoPE tables must be on CUDA");
  TORCH_CHECK(q.dim() == 4,
              "Expected 4D tensor: [batch, num_heads, seq_len, head_dim]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "Q, K, V must have identical shapes");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto cos_f32 = cos_tab.to(torch::kFloat32).contiguous();
  auto sin_f32 = sin_tab.to(torch::kFloat32).contiguous();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_fused_rope_flash_attention<float>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        cos_f32.data_ptr<float>(), sin_f32.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_fused_rope_flash_attention<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        cos_f32.data_ptr<float>(), sin_f32.data_ptr<float>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fused_rope_flash_attention<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        cos_f32.data_ptr<float>(), sin_f32.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, softmax_scale, is_causal,
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for Fused RoPE Attention. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_ROPE

#ifdef WITH_GQA
// PyTorch bindings for Grouped-Query Attention (GQA)
torch::Tensor
fused_gqa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  bool is_causal = true,
                  c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "Inputs must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");

  const int batch_size = q.size(0);
  const int num_q_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);
  const int num_kv_heads = k.size(1);

  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be divisible by num_kv_heads");
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_fused_gqa<float>(q.data_ptr<float>(), k.data_ptr<float>(),
                              v.data_ptr<float>(), output.data_ptr<float>(),
                              batch_size, num_q_heads, num_kv_heads, seq_len,
                              head_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_fused_gqa<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_q_heads, num_kv_heads, seq_len, head_dim, softmax_scale, is_causal,
        stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fused_gqa<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, softmax_scale,
        is_causal, stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for GQA. Supported: Float32, "
                       "Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_GQA

#ifdef WITH_DECODE
// PyTorch bindings for Asynchronous Double-Buffered Flash-Decoding
torch::Tensor async_flash_decode_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "Inputs must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(q.size(2) == 1,
              "Query sequence length must be 1 for Flash-Decoding");
  TORCH_CHECK(k.sizes() == v.sizes(), "K and V must have identical shapes");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = k.size(2);
  const int head_dim = q.size(3);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_async_flash_decode<float>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        softmax_scale, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_async_flash_decode<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, softmax_scale, stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_async_flash_decode<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, softmax_scale, stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for Async Flash-Decode. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_DECODE

#ifdef WITH_BWD
// PyTorch bindings for FlashAttention-2 Backward Pass (Exact autograd
// gradients)
std::vector<torch::Tensor> flash_attention_v2_backward_forward(
    torch::Tensor dout, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse, bool is_causal = false,
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(dout.is_cuda() && q.is_cuda() && k.is_cuda() && v.is_cuda() &&
                  out.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(lse.is_cuda(), "LSE tensor must be on CUDA");
  TORCH_CHECK(q.dim() == 4,
              "Expected 4D tensor: [batch, num_heads, seq_len, head_dim]");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto lse_f32 = lse.to(torch::kFloat32).contiguous();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_flash_attention_v2_backward<float>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(), dout.data_ptr<float>(),
        lse_f32.data_ptr<float>(), dq.data_ptr<float>(), dk.data_ptr<float>(),
        dv.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_flash_attention_v2_backward<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(out.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(dout.data_ptr<at::Half>()),
        lse_f32.data_ptr<float>(),
        reinterpret_cast<__half *>(dq.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(dk.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(dv.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_flash_attention_v2_backward<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(dout.data_ptr<at::BFloat16>()),
        lse_f32.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(dq.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(dv.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, softmax_scale, is_causal,
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for FlashAttention Backward. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return {dq, dk, dv};
}

#endif // WITH_BWD

#ifdef WITH_MLA
// PyTorch bindings for Multi-Head Latent Attention (DeepSeek-V2/V3 MLA)
torch::Tensor
fused_mla_forward(torch::Tensor q, // [batch, num_heads, seq_len, head_dim]
                  torch::Tensor compressed_kv, // [batch, seq_len, latent_dim]
                  torch::Tensor w_uk, // [num_heads, head_dim, latent_dim]
                  torch::Tensor w_uv, // [num_heads, head_dim, latent_dim]
                  bool is_causal = true,
                  c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && compressed_kv.is_cuda() && w_uk.is_cuda() &&
                  w_uv.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.dim() == 4, "Query must be 4D [batch, heads, seq, dim]");
  TORCH_CHECK(compressed_kv.dim() == 3,
              "Compressed KV must be 3D [batch, seq, latent_dim]");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);
  const int latent_dim = compressed_kv.size(2);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");
  TORCH_CHECK(latent_dim == 64, "Currently optimized for latent_dim 64");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_fused_mla<float>(
        q.data_ptr<float>(), compressed_kv.data_ptr<float>(),
        w_uk.data_ptr<float>(), w_uv.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        latent_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_fused_mla<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(compressed_kv.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(w_uk.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(w_uv.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, latent_dim, softmax_scale, is_causal,
        stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fused_mla<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(
            compressed_kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(w_uk.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(w_uv.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, latent_dim, softmax_scale,
        is_causal, stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for MLA. Supported: Float32, "
                       "Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_MLA

#ifdef WITH_FA3
// PyTorch bindings for FlashAttention-3 Hopper (Warp-Specialized Architecture)
torch::Tensor flash_attention_v3_hopper_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false,
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(q.dim() == 4,
              "Expected 4D tensor: [batch, num_heads, seq_len, head_dim]");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "Q, K, V must have identical shapes");

  const int batch_size = q.size(0);
  const int num_heads = q.size(1);
  const int seq_len = q.size(2);
  const int head_dim = q.size(3);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");

  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(q);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (q.scalar_type() == at::ScalarType::Float) {
    dispatch_flash_attention_v3_hopper<float>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, seq_len, head_dim,
        softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::Half) {
    dispatch_flash_attention_v3_hopper<__half>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        num_heads, seq_len, head_dim, softmax_scale, is_causal, stream);
  } else if (q.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_flash_attention_v3_hopper<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, num_heads, seq_len, head_dim, softmax_scale, is_causal,
        stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for FlashAttention-3. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_FA3

#ifdef WITH_MEGABLOCK
// PyTorch bindings for The Fully-Fused Llama-3 Transformer MegaBlock
torch::Tensor fused_llama_megablock_forward(
    torch::Tensor x, torch::Tensor rms_weight1, torch::Tensor rms_weight2,
    torch::Tensor cos_tab, torch::Tensor sin_tab, double eps = 1e-6,
    c10::optional<double> softmax_scale_opt = c10::nullopt,
    bool is_causal = true) {
  TORCH_CHECK(x.is_cuda() && rms_weight1.is_cuda() && rms_weight2.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(cos_tab.is_cuda() && sin_tab.is_cuda(),
              "RoPE tables must be on CUDA");
  TORCH_CHECK(x.dim() == 3, "Input X must be 3D [batch, seq_len, hidden_dim]");

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);

  TORCH_CHECK(hidden_dim == 256 || hidden_dim == 512,
              "Only hidden_dim 256 and 512 supported");

  constexpr int HEAD_DIM = 64;
  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

  auto output = torch::empty_like(x);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto rms1_f32 = rms_weight1.to(torch::kFloat32).contiguous();
  auto rms2_f32 = rms_weight2.to(torch::kFloat32).contiguous();
  auto cos_f32 = cos_tab.to(torch::kFloat32).contiguous();
  auto sin_f32 = sin_tab.to(torch::kFloat32).contiguous();

  if (x.scalar_type() == at::ScalarType::Float) {
    dispatch_fused_llama_megablock<float>(
        x.data_ptr<float>(), rms1_f32.data_ptr<float>(),
        rms2_f32.data_ptr<float>(), cos_f32.data_ptr<float>(),
        sin_f32.data_ptr<float>(), output.data_ptr<float>(), batch_size,
        seq_len, hidden_dim, static_cast<float>(eps), softmax_scale, is_causal,
        stream);
  } else if (x.scalar_type() == at::ScalarType::Half) {
    dispatch_fused_llama_megablock<__half>(
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>()),
        rms1_f32.data_ptr<float>(), rms2_f32.data_ptr<float>(),
        cos_f32.data_ptr<float>(), sin_f32.data_ptr<float>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), batch_size,
        seq_len, hidden_dim, static_cast<float>(eps), softmax_scale, is_causal,
        stream);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fused_llama_megablock<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),
        rms1_f32.data_ptr<float>(), rms2_f32.data_ptr<float>(),
        cos_f32.data_ptr<float>(), sin_f32.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        batch_size, seq_len, hidden_dim, static_cast<float>(eps), softmax_scale,
        is_causal, stream);
  } else {
    TORCH_CHECK(false, "Unsupported scalar type for Llama MegaBlock. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}

#endif // WITH_MEGABLOCK

#ifdef WITH_FP8_PAGED
// PyTorch bindings for Quantized FP8 PagedAttention (vLLM style extreme memory
// compression)
torch::Tensor fp8_paged_attention_forward(
    torch::Tensor query, // [num_seqs, num_heads, head_dim]
    torch::Tensor
        key_cache_fp8, // [num_blocks, num_heads, head_dim, page_size] (int8)
    torch::Tensor
        value_cache_fp8, // [num_blocks, num_heads, head_dim, page_size] (int8)
    torch::Tensor k_scale, // [num_blocks, num_heads] or [num_heads] (float32)
    torch::Tensor v_scale, // [num_blocks, num_heads] or [num_heads] (float32)
    torch::Tensor block_tables, // [num_seqs, max_blocks_per_seq] (int32)
    torch::Tensor context_lens, // [num_seqs] (int32)
    c10::optional<double> softmax_scale_opt = c10::nullopt) {
  TORCH_CHECK(query.is_cuda() && key_cache_fp8.is_cuda() &&
                  value_cache_fp8.is_cuda(),
              "Inputs must be on CUDA");
  TORCH_CHECK(k_scale.is_cuda() && v_scale.is_cuda(),
              "Scale tensors must be on CUDA");
  TORCH_CHECK(block_tables.is_cuda() && context_lens.is_cuda(),
              "Metadata tensors must be on CUDA");
  TORCH_CHECK(query.dim() == 3,
              "Query must be 3D [num_seqs, num_heads, head_dim]");
  TORCH_CHECK(
      key_cache_fp8.dim() == 4 && value_cache_fp8.dim() == 4,
      "KV caches must be 4D [num_blocks, num_heads, head_dim, page_size]");
  TORCH_CHECK(key_cache_fp8.scalar_type() == at::ScalarType::Char ||
                  key_cache_fp8.scalar_type() == at::ScalarType::Byte,
              "KV caches must be 8-bit quantized (int8 or uint8)");

  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  const int head_dim = query.size(2);
  const int page_size = key_cache_fp8.size(3);
  const int max_blocks_per_seq = block_tables.size(1);

  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "Only head_dim 64 and 128 supported");
  TORCH_CHECK(page_size == 16, "Only page_size 16 is currently supported");

  const bool per_block_scales = (k_scale.dim() == 2);
  const float softmax_scale =
      softmax_scale_opt.has_value()
          ? static_cast<float>(softmax_scale_opt.value())
          : 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto output = torch::empty_like(query);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto k_scale_f32 = k_scale.to(torch::kFloat32).contiguous();
  auto v_scale_f32 = v_scale.to(torch::kFloat32).contiguous();

  if (query.scalar_type() == at::ScalarType::Float) {
    dispatch_fp8_paged_attention<float, float>(
        query.data_ptr<float>(),
        reinterpret_cast<const int8_t *>(key_cache_fp8.data_ptr()),
        reinterpret_cast<const int8_t *>(value_cache_fp8.data_ptr()),
        k_scale_f32.data_ptr<float>(), v_scale_f32.data_ptr<float>(),
        block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
        output.data_ptr<float>(), num_seqs, num_heads, head_dim,
        max_blocks_per_seq, page_size, softmax_scale, per_block_scales, stream);
  } else if (query.scalar_type() == at::ScalarType::Half) {
    dispatch_fp8_paged_attention<__half, __half>(
        reinterpret_cast<const __half *>(query.data_ptr<at::Half>()),
        reinterpret_cast<const int8_t *>(key_cache_fp8.data_ptr()),
        reinterpret_cast<const int8_t *>(value_cache_fp8.data_ptr()),
        k_scale_f32.data_ptr<float>(), v_scale_f32.data_ptr<float>(),
        block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
        reinterpret_cast<__half *>(output.data_ptr<at::Half>()), num_seqs,
        num_heads, head_dim, max_blocks_per_seq, page_size, softmax_scale,
        per_block_scales, stream);
  } else if (query.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_fp8_paged_attention<__nv_bfloat16, __nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<const int8_t *>(key_cache_fp8.data_ptr()),
        reinterpret_cast<const int8_t *>(value_cache_fp8.data_ptr()),
        k_scale_f32.data_ptr<float>(), v_scale_f32.data_ptr<float>(),
        block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
        num_seqs, num_heads, head_dim, max_blocks_per_seq, page_size,
        softmax_scale, per_block_scales, stream);
  } else {
    TORCH_CHECK(false, "Unsupported query scalar type for FP8 PagedAttention. "
                       "Supported: Float32, Float16, BFloat16.");
  }

  return output;
}
#endif // WITH_FP8_PAGED

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_SOFTMAX
  m.def("forward", &fused_online_softmax_forward,
        "Fused Online Softmax forward (CUDA)");
#endif
#ifdef WITH_FA2
  m.def("flash_attention_forward", &flash_attention_v2_forward,
        "FlashAttention-2 Forward with Online Softmax and SRAM Tiling (CUDA)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("is_causal") = false,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_PAGED
  m.def(
      "paged_attention_forward", &paged_attention_forward,
      "PagedAttention Forward (vLLM style virtual memory paging for KV cache)",
      pybind11::arg("query"), pybind11::arg("key_cache"),
      pybind11::arg("value_cache"), pybind11::arg("block_tables"),
      pybind11::arg("context_lens"),
      pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_ROPE
  m.def("fused_rope_flash_attention", &fused_rope_flash_attention_forward,
        "Fused RoPE + FlashAttention-2 (Llama-3 Architecture with in-SRAM "
        "rotary embedding)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("cos_tab"), pybind11::arg("sin_tab"),
        pybind11::arg("is_causal") = true,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_GQA
  m.def("fused_gqa_forward", &fused_gqa_forward,
        "Grouped-Query Attention (GQA) Forward with dynamic head-broadcasting",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("is_causal") = true,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_DECODE
  m.def("async_flash_decode", &async_flash_decode_forward,
        "Asynchronous Double-Buffered Flash-Decoding with XOR Shared Memory "
        "Swizzling",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_BWD
  m.def("flash_attention_backward", &flash_attention_v2_backward_forward,
        "FlashAttention-2 Backward Pass with SRAM Recomputation",
        pybind11::arg("dout"), pybind11::arg("q"), pybind11::arg("k"),
        pybind11::arg("v"), pybind11::arg("out"), pybind11::arg("lse"),
        pybind11::arg("is_causal") = false,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_MLA
  m.def("fused_mla_forward", &fused_mla_forward,
        "Multi-Head Latent Attention (DeepSeek-V2/V3 MLA with compressed KV "
        "cache)",
        pybind11::arg("q"), pybind11::arg("compressed_kv"),
        pybind11::arg("w_uk"), pybind11::arg("w_uv"),
        pybind11::arg("is_causal") = true,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_FA3
  m.def("flash_attention_v3_hopper", &flash_attention_v3_hopper_forward,
        "FlashAttention-3 Hopper TMA & Warp-Specialized Forward Kernel (sm_90)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("is_causal") = false,
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
#ifdef WITH_MEGABLOCK
  m.def("fused_llama_megablock", &fused_llama_megablock_forward,
        "The Fully-Fused Llama-3 Transformer MegaBlock (RMSNorm + RoPE + GQA + "
        "SwiGLU FFN in SRAM)",
        pybind11::arg("x"), pybind11::arg("rms_weight1"),
        pybind11::arg("rms_weight2"), pybind11::arg("cos_tab"),
        pybind11::arg("sin_tab"), pybind11::arg("eps") = 1e-6,
        pybind11::arg("softmax_scale") = pybind11::none(),
        pybind11::arg("is_causal") = true);
#endif
#ifdef WITH_FP8_PAGED
  m.def("fp8_paged_attention_forward", &fp8_paged_attention_forward,
        "FP8 Quantized PagedAttention Forward (50-75% KV-cache memory "
        "reduction with in-SRAM dequantization)",
        pybind11::arg("query"), pybind11::arg("key_cache_fp8"),
        pybind11::arg("value_cache_fp8"), pybind11::arg("k_scale"),
        pybind11::arg("v_scale"), pybind11::arg("block_tables"),
        pybind11::arg("context_lens"),
        pybind11::arg("softmax_scale") = pybind11::none());
#endif
}
