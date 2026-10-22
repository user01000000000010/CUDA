#include "async_flash_decode_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_async_flash_decode<float>(
    const float* q,
    const float* k,
    const float* v,
    float* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream);

template void dispatch_async_flash_decode<__half>(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream);

template void dispatch_async_flash_decode<__nv_bfloat16>(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream);
