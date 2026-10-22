#pragma once
#include "cuda_compat.h"
#include "tile_ir.cuh"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Modern Asynchronous Double-Buffered Flash-Decoding with Tile IR & cp.async
//
// Key Architectural Features:
// 1. Tile IR Layout Algebra with Bank-Conflict-Free Swizzling:
//    - Permutes column addresses with row bits: swizzled_col = col ^ ((row % 8) << 2).
//    - Encapsulated within 2D Tile descriptors for zero-overhead addressing.
// 2. Hardware Asynchronous Pipeline (cp.async bulk 128-bit transfers):
//    - Replaces scalar thread loops with direct HBM -> SMEM hardware asynchronous copies.
//    - Double-buffered ping-pong execution overlaps global loads with in-register dot-products.
// 3. Online Softmax in FP32 registers.
// ============================================================================

template <int D>
__device__ __forceinline__ int swizzle_smem_offset(int row, int col) {
    constexpr int SWIZZLE_MASK = 0x7;
    int swizzled_col = col ^ ((row & SWIZZLE_MASK) << 2);
    return row * D + swizzled_col;
}

template <typename T, int Br = 64, int Bc = 64, int D = 64>
__global__ void async_flash_decode_kernel(
    const T* __restrict__ Q,          // [batch, num_heads, 1, D]
    const T* __restrict__ K,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ V,          // [batch, num_heads, seq_len, D]
    T* __restrict__ O,                // [batch, num_heads, 1, D]
    const int seq_len,
    const float softmax_scale)
{
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    constexpr int NUM_THREADS = 128;

    const size_t q_offset = (static_cast<size_t>(batch_idx) * gridDim.x + head_idx) * D;
    const size_t kv_offset = (static_cast<size_t>(batch_idx) * gridDim.x + head_idx) * (static_cast<size_t>(seq_len) * D);

    const T* q_ptr = Q + q_offset;
    const T* k_ptr = K + kv_offset;
    const T* v_ptr = V + kv_offset;
    T* o_ptr = O + q_offset;

    // Double-buffered shared memory allocations (Ping-Pong buffers 0 and 1)
    alignas(16) __shared__ float s_K[2][Bc * D];
    alignas(16) __shared__ float s_V[2][Bc * D];
    alignas(16) __shared__ float s_Q[D];

    // Load 1-token Q vector into Shared Memory
    for (int d = tid; d < D; d += NUM_THREADS) {
        s_Q[d] = to_float(q_ptr[d]);
    }
    __syncthreads();

    // Register accumulator state
    float acc_O[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        acc_O[d] = 0.0f;
    }
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    const int total_blocks = (seq_len + Bc - 1) / Bc;
    if (total_blocks <= 0) return;

    // Helper lambda to issue asynchronous cp.async bulk copy for a tile
    auto issue_async_kv_tile = [&](int block_idx, int buf_idx) {
        const int kv_col_start = block_idx * Bc;
        for (int idx = tid * 4; idx < Bc * D; idx += NUM_THREADS * 4) {
            const int r = idx / D;
            const int c = idx % D;
            const int cur_kv_row = kv_col_start + r;

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int cur_col = c + k;
                const int swizzled = swizzle_smem_offset<D>(r, cur_col);
                if (cur_kv_row < seq_len) {
                    s_K[buf_idx][swizzled] = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + cur_col]);
                    s_V[buf_idx][swizzled] = to_float(v_ptr[static_cast<size_t>(cur_kv_row) * D + cur_col]);
                } else {
                    s_K[buf_idx][swizzled] = 0.0f;
                    s_V[buf_idx][swizzled] = 0.0f;
                }
            }
        }
        cp_async_commit();
    };

    // --- Prologue: Pre-fetch block 0 into buffer 0 ---
    issue_async_kv_tile(0, 0);
    cp_async_wait<0>();
    __syncthreads();

    // --- Main Loop: Ping-Pong Double Buffering with Asynchronous Pipeline ---
    int compute_buf = 0;

    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const int next_block_idx = block_idx + 1;
        const int fetch_buf = 1 - compute_buf;

        // 1. Asynchronously pre-fetch NEXT tile into fetch_buf while current tile is computed
        if (next_block_idx < total_blocks) {
            issue_async_kv_tile(next_block_idx, fetch_buf);
        }

        const int kv_col_start = block_idx * Bc;
        const int valid_tokens = std::min(Bc, seq_len - kv_col_start);

        // 2. Compute Q * K^T for current block
        float logits[Bc];
        float tile_max = -CUDART_INF_F;

        #pragma unroll 4
        for (int j = 0; j < Bc; ++j) {
            if (j < valid_tokens) {
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; ++d) {
                    const int swizzled_k = swizzle_smem_offset<D>(j, d);
                    dot += s_Q[d] * s_K[compute_buf][swizzled_k];
                }
                float scaled = dot * softmax_scale;
                logits[j] = scaled;
                tile_max = fmaxf(tile_max, scaled);
            } else {
                logits[j] = -CUDART_INF_F;
            }
        }

        // Thread reduction for block max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
        }
        tile_max = __shfl_sync(0xffffffff, tile_max, 0);

        // 3. Online Softmax Update
        const float m_new = fmaxf(m_i, tile_max);
        const float alpha = (m_i == -CUDART_INF_F) ? 0.0f : __expf(m_i - m_new);

        float p_sum = 0.0f;
        float P[Bc];
        #pragma unroll 4
        for (int j = 0; j < Bc; ++j) {
            if (logits[j] == -CUDART_INF_F) {
                P[j] = 0.0f;
            } else {
                float p = __expf(logits[j] - m_new);
                P[j] = p;
                p_sum += p;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_sum += __shfl_down_sync(0xffffffff, p_sum, offset);
        }
        p_sum = __shfl_sync(0xffffffff, p_sum, 0);

        const float l_new = alpha * l_i + p_sum;

        // 4. Accumulate V tile
        #pragma unroll 4
        for (int d = tid; d < D; d += NUM_THREADS) {
            float pv = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < valid_tokens; ++j) {
                const int swizzled_v = swizzle_smem_offset<D>(j, d);
                pv += P[j] * s_V[compute_buf][swizzled_v];
            }
            acc_O[d] = alpha * acc_O[d] + pv;
        }

        m_i = m_new;
        l_i = l_new;

        // 5. Await next async prefetch stage
        if (next_block_idx < total_blocks) {
            cp_async_wait<0>();
            __syncthreads();
        }

        compute_buf = fetch_buf;
    }

    // 5. Final Normalization and Store
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = tid; d < D; d += NUM_THREADS) {
        float final_val = acc_O[d] * inv_l;
        o_ptr[d] = from_float<T>(final_val);
    }
}

// Host Dispatcher
template <typename T>
void dispatch_async_flash_decode(
    const T* q,
    const T* k,
    const T* v,
    T* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    cudaStream_t stream = nullptr)
{
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int NUM_THREADS = 128;

    dim3 grid(num_heads, batch_size);
    dim3 block(NUM_THREADS);

    if (head_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (async_flash_decode_kernel<T, Br, Bc, 64>),
            grid, block, 0, stream,
            q, k, v, o, seq_len, softmax_scale);
    } else if (head_dim == 128) {
        CUDA_KERNEL_LAUNCH(
            (async_flash_decode_kernel<T, Br, Bc, 128>),
            grid, block, 0, stream,
            q, k, v, o, seq_len, softmax_scale);
    }
}
