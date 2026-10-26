#pragma once
#include "cuda_compat.h"
#include "tile_ir.cuh"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Modern Multi-Head Latent Attention (MLA) with Tile IR & cp.async Pipeline
// (DeepSeek-V2 / V3 Compressed KV Architecture)
//
// Key Architectural Features:
// 1. Low-Rank KV Compression (Latent Cache):
//    - Caches a single compressed latent vector C_KV (dim = LATENT_DIM).
//    - Reduces KV-cache footprint by up to 5x-8x compared to full MHA.
// 2. Tile IR Tensormodellering:
//    - Tensors s_Q (Br x D), s_W_UK (D x LATENT_DIM), s_W_UV (D x LATENT_DIM),
//      and double-buffered s_CKV (Bc x LATENT_DIM) modeled as Tile descriptors.
// 3. Asynchronous Pipeline (cp.async):
//    - Overlaps global C_KV fetches with low-rank projection and in-register Online Softmax.
// ============================================================================

template <typename T, int Br = 64, int Bc = 64, int D = 64, int LATENT_DIM = 64>
__launch_bounds__(128, 2)
__global__ void fused_mla_forward_kernel(
    const T* __restrict__ Q,              // [batch, num_heads, seq_len, D]
    const T* __restrict__ Compressed_KV,  // [batch, seq_len, LATENT_DIM]
    const T* __restrict__ W_UK,           // [num_heads, D, LATENT_DIM]
    const T* __restrict__ W_UV,           // [num_heads, D, LATENT_DIM]
    T* __restrict__ O,                    // [batch, num_heads, seq_len, D]
    const int seq_len,
    const int num_heads,
    const float softmax_scale,
    const bool is_causal)
{
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int NUM_THREADS = 128;

    // Strides
    const size_t q_bh_offset = (static_cast<size_t>(batch_idx) * num_heads + head_idx) * (static_cast<size_t>(seq_len) * D);
    const size_t ckv_b_offset = static_cast<size_t>(batch_idx) * (static_cast<size_t>(seq_len) * LATENT_DIM);
    const size_t w_head_offset = static_cast<size_t>(head_idx) * (static_cast<size_t>(D) * LATENT_DIM);

    const T* q_ptr = Q + q_bh_offset;
    const T* ckv_ptr = Compressed_KV + ckv_b_offset;
    const T* w_uk_ptr = W_UK + w_head_offset;
    const T* w_uv_ptr = W_UV + w_head_offset;
    T* o_ptr = O + q_bh_offset;

    // Double-buffered Shared Memory staging for tiles
    alignas(16) __shared__ float s_Q_data[Br * D];
    alignas(16) __shared__ float s_CKV_data[2][Bc * LATENT_DIM];
    alignas(16) __shared__ float s_W_UK_data[D * LATENT_DIM];
    alignas(16) __shared__ float s_W_UV_data[D * LATENT_DIM];

    auto s_Q = tile_ir::make_row_major_tensor<Br, D>(s_Q_data);
    auto s_W_UK = tile_ir::make_row_major_tensor<D, LATENT_DIM>(s_W_UK_data);
    auto s_W_UV = tile_ir::make_row_major_tensor<D, LATENT_DIM>(s_W_UV_data);

    const int local_row = tid;
    const int q_row_start = q_block_idx * Br;
    const int global_q_row = q_row_start + local_row;

    // 1. Load Q tile and projection weights into SRAM
    for (int idx = tid * 4; idx < Br * D; idx += NUM_THREADS * 4) {
        const int r = idx / D;
        const int c = idx % D;
        const int cur_q_row = q_row_start + r;
        if (cur_q_row < seq_len) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                s_Q(r, c + k) = to_float(q_ptr[static_cast<size_t>(cur_q_row) * D + (c + k)]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                s_Q(r, c + k) = 0.0f;
            }
        }
    }

    for (int idx = tid * 4; idx < D * LATENT_DIM; idx += NUM_THREADS * 4) {
        const int r = idx / LATENT_DIM;
        const int c = idx % LATENT_DIM;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            s_W_UK(r, c + k) = to_float(w_uk_ptr[idx + k]);
            s_W_UV(r, c + k) = to_float(w_uv_ptr[idx + k]);
        }
    }
    __syncthreads();

    // Register accumulator state in latent space (huge complexity reduction)
    float acc_latent[LATENT_DIM];
    #pragma unroll
    for (int l = 0; l < LATENT_DIM; ++l) {
        acc_latent[l] = 0.0f;
    }

    // Precompute Q projection onto latent space: Q_proj = Q * W_UK
    float q_proj[LATENT_DIM];
    if (local_row < Br && global_q_row < seq_len) {
        #pragma unroll 4
        for (int l = 0; l < LATENT_DIM; ++l) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                sum += s_Q(local_row, d) * s_W_UK(d, l);
            }
            q_proj[l] = sum;
        }
    }

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    const int max_kv_block = is_causal ? std::min(num_kv_blocks, (q_row_start + Br + Bc - 1) / Bc) : num_kv_blocks;

    // Helper lambda to load a compressed C_KV tile
    auto load_ckv_stage = [&](int block_idx, int stage_idx) {
        auto s_CKV = tile_ir::make_row_major_tensor<Bc, LATENT_DIM>(s_CKV_data[stage_idx]);
        const int kv_col_start = block_idx * Bc;

        for (int idx = tid * 4; idx < Bc * LATENT_DIM; idx += NUM_THREADS * 4) {
            const int r = idx / LATENT_DIM;
            const int c = idx % LATENT_DIM;
            const int cur_kv_row = kv_col_start + r;

            if (cur_kv_row < seq_len) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(__CUDACC_RTC__)
                cp_async_16(&s_CKV_data[stage_idx][idx], &ckv_ptr[static_cast<size_t>(cur_kv_row) * LATENT_DIM + c]);
#else
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s_CKV(r, c + k) = to_float(ckv_ptr[static_cast<size_t>(cur_kv_row) * LATENT_DIM + (c + k)]);
                }
#endif
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s_CKV(r, c + k) = 0.0f;
                }
            }
        }
        cp_async_commit();
    };

    // 2. Prime the pipeline
    if (max_kv_block > 0) {
        load_ckv_stage(0, 0);
        cp_async_wait<0>();
        __syncthreads();
    }

    // 3. Double-buffered main loop
    int curr_stage = 0;
    for (int kv_block_idx = 0; kv_block_idx < max_kv_block; ++kv_block_idx) {
        const int next_stage = curr_stage ^ 1;
        const int next_block_idx = kv_block_idx + 1;

        if (next_block_idx < max_kv_block) {
            load_ckv_stage(next_block_idx, next_stage);
        }

        auto s_CKV = tile_ir::make_row_major_tensor<Bc, LATENT_DIM>(s_CKV_data[curr_stage]);
        const int kv_col_start = kv_block_idx * Bc;

        if (local_row < Br && global_q_row < seq_len) {
            float S_row[Bc];
            float block_max = -CUDART_INF_F;

            #pragma unroll 4
            for (int j = 0; j < Bc; ++j) {
                const int global_k_col = kv_col_start + j;
                if (global_k_col >= seq_len || (is_causal && global_k_col > global_q_row)) {
                    S_row[j] = -CUDART_INF_F;
                } else {
                    // Pre-projected Q_proj * C_KV: O(LATENT_DIM) instead of O(D * LATENT_DIM)
                    float dot = 0.0f;
                    #pragma unroll 8
                    for (int l = 0; l < LATENT_DIM; ++l) {
                        dot += q_proj[l] * s_CKV(j, l);
                    }
                    float scaled = dot * softmax_scale;
                    S_row[j] = scaled;
                    block_max = fmaxf(block_max, scaled);
                }
            }

            // Online Softmax in FP32 registers
            const float m_new = fmaxf(m_i, block_max);
            const float alpha = (m_i == -CUDART_INF_F) ? 0.0f : __expf(m_i - m_new);

            float block_sum_exp = 0.0f;
            float P_row[Bc];
            #pragma unroll 4
            for (int j = 0; j < Bc; ++j) {
                if (S_row[j] == -CUDART_INF_F) {
                    P_row[j] = 0.0f;
                } else {
                    float p = __expf(S_row[j] - m_new);
                    P_row[j] = p;
                    block_sum_exp += p;
                }
            }

            const float l_new = alpha * l_i + block_sum_exp;

            // Accumulate into latent space directly: O(Bc * LATENT_DIM)
            #pragma unroll 4
            for (int l = 0; l < LATENT_DIM; ++l) {
                float pl = 0.0f;
                #pragma unroll 4
                for (int j = 0; j < Bc; ++j) {
                    pl += P_row[j] * s_CKV(j, l);
                }
                acc_latent[l] = alpha * acc_latent[l] + pl;
            }

            m_i = m_new;
            l_i = l_new;
        }

        if (next_block_idx < max_kv_block) {
            cp_async_wait<0>();
            __syncthreads();
        }
        curr_stage = next_stage;
    }

    // 4. Final normalization and project back from latent space to head_dim: O = (acc_latent / l_i) * W_UV^T
    if (local_row < Br && global_q_row < seq_len) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float out_d = 0.0f;
            #pragma unroll 8
            for (int l = 0; l < LATENT_DIM; ++l) {
                out_d += (acc_latent[l] * inv_l) * s_W_UV(d, l);
            }
            o_ptr[static_cast<size_t>(global_q_row) * D + d] = from_float<T>(out_d);
        }
    }
}

// Host Dispatcher
template <typename T>
void dispatch_fused_mla(
    const T* q,
    const T* compressed_kv,
    const T* w_uk,
    const T* w_uv,
    T* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int latent_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream = nullptr)
{
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int NUM_THREADS = 128;

    const int num_q_blocks = (seq_len + Br - 1) / Br;
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(NUM_THREADS);

    if (head_dim == 64 && latent_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (fused_mla_forward_kernel<T, Br, Bc, 64, 64>),
            grid, block, 0, stream,
            q, compressed_kv, w_uk, w_uv, o,
            seq_len, num_heads, softmax_scale, is_causal);
    } else if (head_dim == 128 && latent_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (fused_mla_forward_kernel<T, Br, Bc, 128, 64>),
            grid, block, 0, stream,
            q, compressed_kv, w_uk, w_uv, o,
            seq_len, num_heads, softmax_scale, is_causal);
    }
}
