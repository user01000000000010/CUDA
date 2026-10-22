#include "fused_rope_flash_attention_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fused_rope_flash_attention<float>(
    const float* q,
    const float* k,
    const float* v,
    const float* cos_tab,
    const float* sin_tab,
    float* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_rope_flash_attention<__half>(
    const __half* q,
    const __half* k,
    const __half* v,
    const float* cos_tab,
    const float* sin_tab,
    __half* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_rope_flash_attention<__nv_bfloat16>(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* cos_tab,
    const float* sin_tab,
    __nv_bfloat16* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);
