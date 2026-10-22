#include "fused_online_softmax_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fused_online_softmax<float>(
    const float* input,
    float* output,
    int batch_size,
    int d_dim,
    cudaStream_t stream);

template void dispatch_fused_online_softmax<__half>(
    const __half* input,
    __half* output,
    int batch_size,
    int d_dim,
    cudaStream_t stream);

template void dispatch_fused_online_softmax<__nv_bfloat16>(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch_size,
    int d_dim,
    cudaStream_t stream);
