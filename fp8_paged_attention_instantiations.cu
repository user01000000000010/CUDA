#include "fp8_paged_attention_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_fp8_paged_attention<float, float>(
    const float* query,
    const int8_t* key_cache_fp8,
    const int8_t* value_cache_fp8,
    const float* k_scale,
    const float* v_scale,
    const int32_t* block_tables,
    const int32_t* context_lens,
    float* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    int page_size,
    float softmax_scale,
    bool per_block_scales,
    cudaStream_t stream);

template void dispatch_fp8_paged_attention<__half, __half>(
    const __half* query,
    const int8_t* key_cache_fp8,
    const int8_t* value_cache_fp8,
    const float* k_scale,
    const float* v_scale,
    const int32_t* block_tables,
    const int32_t* context_lens,
    __half* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    int page_size,
    float softmax_scale,
    bool per_block_scales,
    cudaStream_t stream);

template void dispatch_fp8_paged_attention<__nv_bfloat16, __nv_bfloat16>(
    const __nv_bfloat16* query,
    const int8_t* key_cache_fp8,
    const int8_t* value_cache_fp8,
    const float* k_scale,
    const float* v_scale,
    const int32_t* block_tables,
    const int32_t* context_lens,
    __nv_bfloat16* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    int page_size,
    float softmax_scale,
    bool per_block_scales,
    cudaStream_t stream);
