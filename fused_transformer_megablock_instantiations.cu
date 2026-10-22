#include "fused_transformer_megablock.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fused_llama_megablock<float>(
    const float* x,
    const float* rms_weight1,
    const float* rms_weight2,
    const float* cos_tab,
    const float* sin_tab,
    float* out,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_llama_megablock<__half>(
    const __half* x,
    const float* rms_weight1,
    const float* rms_weight2,
    const float* cos_tab,
    const float* sin_tab,
    __half* out,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);

template void dispatch_fused_llama_megablock<__nv_bfloat16>(
    const __nv_bfloat16* x,
    const float* rms_weight1,
    const float* rms_weight2,
    const float* cos_tab,
    const float* sin_tab,
    __nv_bfloat16* out,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream);
