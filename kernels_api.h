#pragma once
// ============================================================================
// Host-side kernel API (declarations only)
//
// The PyTorch binding (compiled by the *host* C++ compiler) must only see
// these declarations. The definitions live in the *.cuh headers and are
// explicitly instantiated by nvcc in the *_instantiations.cu files.
//
// Previously the binding #included the .cuh files directly. Without
// __CUDACC__ that pulled in the host stubs from cuda_compat.h, instantiating
// a CPU "kernel" that ran on device pointers and violated the ODR against the
// real nvcc instantiations.
//
// NOTE: no default arguments here – the binding always passes the stream.
// ============================================================================
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
void dispatch_fused_online_softmax(const T* input, T* output, int batch_size,
                                   int d_dim, cudaStream_t stream);

template <typename T>
void dispatch_flash_attention_v2(const T* q, const T* k, const T* v, T* o,
                                 int batch_size, int num_heads, int seq_len,
                                 int head_dim, float softmax_scale,
                                 bool is_causal, cudaStream_t stream);

template <typename T>
void dispatch_paged_attention(const T* query, const T* key_cache,
                              const T* value_cache, const int32_t* block_tables,
                              const int32_t* context_lens, T* out, int num_seqs,
                              int num_heads, int head_dim,
                              int max_blocks_per_seq, float softmax_scale,
                              cudaStream_t stream);

template <typename T>
void dispatch_fused_rope_flash_attention(const T* q, const T* k, const T* v,
                                         const float* cos_tab,
                                         const float* sin_tab, T* o,
                                         int batch_size, int num_heads,
                                         int seq_len, int head_dim,
                                         float softmax_scale, bool is_causal,
                                         cudaStream_t stream);

template <typename T>
void dispatch_fused_gqa(const T* q, const T* k, const T* v, T* o,
                        int batch_size, int num_q_heads, int num_kv_heads,
                        int seq_len, int head_dim, float softmax_scale,
                        bool is_causal, cudaStream_t stream);

template <typename T>
void dispatch_async_flash_decode(const T* q, const T* k, const T* v, T* o,
                                 int batch_size, int num_heads, int seq_len,
                                 int head_dim, float softmax_scale,
                                 cudaStream_t stream);

template <typename T>
void dispatch_flash_attention_v2_backward(
    const T* q, const T* k, const T* v, const T* out, const T* dout,
    const float* lse, T* dq, T* dk, T* dv, int batch_size, int num_heads,
    int seq_len, int head_dim, float softmax_scale, bool is_causal,
    cudaStream_t stream);

template <typename T>
void dispatch_fused_mla(const T* q, const T* compressed_kv, const T* w_uk,
                        const T* w_uv, T* o, int batch_size, int num_heads,
                        int seq_len, int head_dim, int latent_dim,
                        float softmax_scale, bool is_causal,
                        cudaStream_t stream);

template <typename T>
void dispatch_flash_attention_v3_hopper(const T* q, const T* k, const T* v,
                                        T* o, int batch_size, int num_heads,
                                        int seq_len, int head_dim,
                                        float softmax_scale, bool is_causal,
                                        cudaStream_t stream);

template <typename T>
void dispatch_fused_llama_megablock(const T* x, const float* rms_weight1,
                                    const float* rms_weight2,
                                    const float* cos_tab, const float* sin_tab,
                                    T* out, int batch_size, int seq_len,
                                    int hidden_dim, float eps,
                                    float softmax_scale, bool is_causal,
                                    cudaStream_t stream);

template <typename T_OUT, typename T_Q>
void dispatch_fp8_paged_attention(
    const T_Q* query, const int8_t* key_cache_fp8,
    const int8_t* value_cache_fp8, const float* k_scale, const float* v_scale,
    const int32_t* block_tables, const int32_t* context_lens, T_OUT* out,
    int num_seqs, int num_heads, int head_dim, int max_blocks_per_seq,
    int page_size, float softmax_scale, bool per_block_scales,
    cudaStream_t stream);
