#include "fused_gqa_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fused_gqa<float>(
    const float* q,
    const float* k,
    const float* v,
    float* o,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_gqa<__half>(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* o,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_gqa<__nv_bfloat16>(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* o,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);
