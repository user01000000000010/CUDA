#include "fused_mla_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fused_mla<float>(
    const float* q,
    const float* compressed_kv,
    const float* w_uk,
    const float* w_uv,
    float* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int latent_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_mla<__half>(
    const __half* q,
    const __half* compressed_kv,
    const __half* w_uk,
    const __half* w_uv,
    __half* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int latent_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_mla<__nv_bfloat16>(
    const __nv_bfloat16* q,
    const __nv_bfloat16* compressed_kv,
    const __nv_bfloat16* w_uk,
    const __nv_bfloat16* w_uv,
    __nv_bfloat16* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int latent_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);
