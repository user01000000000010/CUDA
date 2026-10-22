#include "paged_attention_kernel.cuh"

// Explicit template instantiations compiled by NVCC
template void dispatch_paged_attention<float>(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    float* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    float softmax_scale,
    cudaStream_t stream);

template void dispatch_paged_attention<__half>(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    __half* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    float softmax_scale,
    cudaStream_t stream);

template void dispatch_paged_attention<__nv_bfloat16>(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    __nv_bfloat16* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    float softmax_scale,
    cudaStream_t stream);
