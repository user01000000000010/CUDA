#include "flash_attention_bwd_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_flash_attention_v2_backward<float>(
    const float* q,
    const float* k,
    const float* v,
    const float* out,
    const float* dout,
    const float* lse,
    float* dq,
    float* dk,
    float* dv,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_flash_attention_v2_backward<__half>(
    const __half* q,
    const __half* k,
    const __half* v,
    const __half* out,
    const __half* dout,
    const float* lse,
    __half* dq,
    __half* dk,
    __half* dv,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_flash_attention_v2_backward<__nv_bfloat16>(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* out,
    const __nv_bfloat16* dout,
    const float* lse,
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* dv,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);
