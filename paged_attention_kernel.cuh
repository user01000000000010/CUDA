#pragma once
#include "cuda_compat.h"
#include <cstdint>
#include <algorithm>

// ============================================================================
// PagedAttention v1/v2 Kernel (vLLM style Virtual Memory Paging for KV Cache)
//
// Key Architectural Concepts:
// 1. Dynamic Virtual Memory Paging:
//    KV Cache is allocated in non-contiguous physical pages (blocks of size PAGE_SIZE, e.g., 16).
//    Block tables translate logical token positions to physical memory blocks.
// 2. Decode Phase Specialization:
//    Query length is 1 (M=1).
//    Threads in a block cooperate across pages to compute dot products and aggregate with Online Softmax.
// 3. Zero Memory Fragmentation:
//    Eliminates internal and external fragmentation during concurrent batch generation.
// ============================================================================

template <typename T, int PAGE_SIZE = 16, int HEAD_DIM = 64, int NUM_THREADS = 128>
__launch_bounds__(NUM_THREADS, 4)
__global__ void paged_attention_kernel(
    const T* __restrict__ query,            // [num_seqs, num_heads, HEAD_DIM]
    const T* __restrict__ key_cache,        // [num_blocks, num_heads, HEAD_DIM, PAGE_SIZE]
    const T* __restrict__ value_cache,      // [num_blocks, num_heads, HEAD_DIM, PAGE_SIZE]
    const int32_t* __restrict__ block_tables, // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ context_lens, // [num_seqs]
    T* __restrict__ out,                    // [num_seqs, num_heads, HEAD_DIM]
    const int max_blocks_per_seq,
    const int num_heads,
    const float softmax_scale)
{
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    // Pointer to this sequence and head's 1-token query vector
    const size_t q_offset = (static_cast<size_t>(seq_idx) * num_heads + head_idx) * HEAD_DIM;
    const T* q_vec = query + q_offset;
    T* out_vec = out + q_offset;

    // Load query into registers / shared memory
    __shared__ float s_query[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
        s_query[d] = to_float(q_vec[d]);
    }
    __syncthreads();

    // Online Softmax State tracking in registers across all pages
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;
    float acc_out[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc_out[d] = 0.0f;
    }

    const int32_t* seq_block_table = block_tables + static_cast<size_t>(seq_idx) * max_blocks_per_seq;
    const int num_context_blocks = (context_len + PAGE_SIZE - 1) / PAGE_SIZE;

    // Shared scratchpad for block-level reduction
    __shared__ float s_logits[PAGE_SIZE];

    // Iterate across the virtual pages (blocks) for this sequence
    for (int block_idx = 0; block_idx < num_context_blocks; ++block_idx) {
        const int32_t physical_block = seq_block_table[block_idx];

        // Tokens in this specific page
        const int valid_tokens_in_page = std::min(PAGE_SIZE, context_len - block_idx * PAGE_SIZE);

        // Compute Q * K^T for the tokens in this block
        // Strides for KV-cache [num_blocks, num_heads, HEAD_DIM, PAGE_SIZE]
        const size_t kv_head_offset = (static_cast<size_t>(physical_block) * num_heads + head_idx) * (HEAD_DIM * PAGE_SIZE);
        const T* k_block = key_cache + kv_head_offset;
        const T* v_block = value_cache + kv_head_offset;

        // Each thread computes dot product for token positions it covers
        for (int tok = tid; tok < PAGE_SIZE; tok += NUM_THREADS) {
            if (tok < valid_tokens_in_page) {
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float k_val = to_float(k_block[d * PAGE_SIZE + tok]);
                    dot += s_query[d] * k_val;
                }
                s_logits[tok] = dot * softmax_scale;
            } else {
                s_logits[tok] = -CUDART_INF_F;
            }
        }
        __syncthreads();

        // 1. Find max of current page logits
        float page_max = -CUDART_INF_F;
        for (int tok = 0; tok < valid_tokens_in_page; ++tok) {
            page_max = fmaxf(page_max, s_logits[tok]);
        }

        // 2. Online Softmax Update
        const float m_new = fmaxf(m_i, page_max);
        const float alpha = (m_i == -CUDART_INF_F) ? 0.0f : __expf(m_i - m_new);

        float page_sum_exp = 0.0f;
        float page_p[PAGE_SIZE];
        #pragma unroll
        for (int tok = 0; tok < PAGE_SIZE; ++tok) {
            if (tok < valid_tokens_in_page) {
                float p = __expf(s_logits[tok] - m_new);
                page_p[tok] = p;
                page_sum_exp += p;
            } else {
                page_p[tok] = 0.0f;
            }
        }

        const float l_new = alpha * l_i + page_sum_exp;

        // 3. Accumulate weighted values: acc = alpha * acc + P * V
        for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
            float pv = 0.0f;
            #pragma unroll
            for (int tok = 0; tok < PAGE_SIZE; ++tok) {
                if (tok < valid_tokens_in_page) {
                    float v_val = to_float(v_block[d * PAGE_SIZE + tok]);
                    pv += page_p[tok] * v_val;
                }
            }
            acc_out[d] = alpha * acc_out[d] + pv;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    // Final normalization: out = acc_out / l_i
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = tid; d < HEAD_DIM; d += NUM_THREADS) {
        float final_val = acc_out[d] * inv_l;
        out_vec[d] = from_float<T>(final_val);
    }
}

// Host Dispatcher
template <typename T>
void dispatch_paged_attention(
    const T* query,
    const T* key_cache,
    const T* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    T* out,
    int num_seqs,
    int num_heads,
    int head_dim,
    int max_blocks_per_seq,
    float softmax_scale,
    cudaStream_t stream = nullptr)
{
    constexpr int PAGE_SIZE = 16;
    constexpr int NUM_THREADS = 128;

    dim3 grid(num_seqs, num_heads);
    dim3 block(NUM_THREADS);

    if (head_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (paged_attention_kernel<T, PAGE_SIZE, 64, NUM_THREADS>),
            grid, block, 0, stream,
            query, key_cache, value_cache, block_tables, context_lens,
            out, max_blocks_per_seq, num_heads, softmax_scale);
    } else if (head_dim == 128) {
        CUDA_KERNEL_LAUNCH(
            (paged_attention_kernel<T, PAGE_SIZE, 128, NUM_THREADS>),
            grid, block, 0, stream,
            query, key_cache, value_cache, block_tables, context_lens,
            out, max_blocks_per_seq, num_heads, softmax_scale);
    }
}
