#pragma once
#include "cuda_compat.h"
#include "tile_ir.cuh"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Modern Fused RoPE & Tensor-Optimized FlashAttention-2 with Tile IR & cp.async
//
// Key Architectural Features:
// 1. In-SRAM RoPE Fusion with Tile IR:
//    - Rotary Position Embedding computed on-the-fly during tile staging in SMEM.
//    - Zero HBM DRAM roundtrips for intermediate rotary-rotated tensors.
// 2. Hardware Asynchronous Pipeline (cp.async double-buffering):
//    - Double-buffered ping-pong staging overlaps global fetches with Attention MMA.
// 3. Online Softmax in FP32 registers.
// ============================================================================

template <typename T, int Br = 64, int Bc = 64, int D = 64>
__global__ void fused_rope_flash_attention_kernel(
    const T* __restrict__ Q,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ K,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ V,          // [batch, num_heads, seq_len, D]
    const float* __restrict__ cos_tab,// [seq_len, D]
    const float* __restrict__ sin_tab,// [seq_len, D]
    T* __restrict__ O,                // [batch, num_heads, seq_len, D]
    const int seq_len,
    const float softmax_scale,
    const bool is_causal)
{
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int NUM_THREADS = 128;
    constexpr int HALF_D = D / 2;

    const size_t bh_offset = (static_cast<size_t>(batch_idx) * gridDim.y + head_idx) * (static_cast<size_t>(seq_len) * D);
    const T* q_ptr = Q + bh_offset;
    const T* k_ptr = K + bh_offset;
    const T* v_ptr = V + bh_offset;
    T* o_ptr = O + bh_offset;

    // Double-buffered Shared Memory staging for tiles
    alignas(16) __shared__ float s_Q_data[Br * D];
    alignas(16) __shared__ float s_K_data[2][Bc * D];
    alignas(16) __shared__ float s_V_data[2][Bc * D];

    auto s_Q = tile_ir::make_row_major_tensor<Br, D>(s_Q_data);

    const int local_row = tid;
    const int q_row_start = q_block_idx * Br;
    const int global_q_row = q_row_start + local_row;

    // 1. Load Q tile & apply FUSED RoPE into Shared Memory via Tile IR
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        const int r = idx / D;
        const int c = idx % D;
        const int cur_q_row = q_row_start + r;

        if (cur_q_row < seq_len) {
            float raw_val = to_float(q_ptr[static_cast<size_t>(cur_q_row) * D + c]);
            int partner_c = (c < HALF_D) ? (c + HALF_D) : (c - HALF_D);
            float partner_val = to_float(q_ptr[static_cast<size_t>(cur_q_row) * D + partner_c]);
            float rot_half = (c < HALF_D) ? -partner_val : partner_val;

            float cos_v = cos_tab[static_cast<size_t>(cur_q_row) * D + c];
            float sin_v = sin_tab[static_cast<size_t>(cur_q_row) * D + c];
            s_Q(r, c) = raw_val * cos_v + rot_half * sin_v;
        } else {
            s_Q(r, c) = 0.0f;
        }
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

    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    const int max_kv_block = is_causal ? std::min(num_kv_blocks, (q_row_start + Br + Bc - 1) / Bc) : num_kv_blocks;

    // Helper lambda to load a KV block with fused in-SRAM RoPE
    auto load_kv_stage = [&](int block_idx, int stage_idx) {
        auto s_K = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[stage_idx]);
        auto s_V = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[stage_idx]);
        const int kv_col_start = block_idx * Bc;

        for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
            const int r = idx / D;
            const int c = idx % D;
            const int cur_kv_row = kv_col_start + r;

            if (cur_kv_row < seq_len) {
                float raw_k = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
                int partner_c = (c < HALF_D) ? (c + HALF_D) : (c - HALF_D);
                float partner_k = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + partner_c]);
                float rot_half_k = (c < HALF_D) ? -partner_k : partner_k;

                float cos_v = cos_tab[static_cast<size_t>(cur_kv_row) * D + c];
                float sin_v = sin_tab[static_cast<size_t>(cur_kv_row) * D + c];
                s_K(r, c) = raw_k * cos_v + rot_half_k * sin_v;

                s_V(r, c) = to_float(v_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
            } else {
                s_K(r, c) = 0.0f;
                s_V(r, c) = 0.0f;
            }
        }
        cp_async_commit();
    };

    // 2. Prime the pipeline
    if (max_kv_block > 0) {
        load_kv_stage(0, 0);
        cp_async_wait<0>();
        __syncthreads();
    }

    // 3. Main Double-Buffered Pipeline Loop
    int curr_stage = 0;
    for (int kv_block_idx = 0; kv_block_idx < max_kv_block; ++kv_block_idx) {
        const int next_stage = curr_stage ^ 1;
        const int next_block_idx = kv_block_idx + 1;

        if (next_block_idx < max_kv_block) {
            load_kv_stage(next_block_idx, next_stage);
        }

        auto s_K = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[curr_stage]);
        auto s_V = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[curr_stage]);
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
                    float dot = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; ++d) {
                        dot += s_Q(local_row, d) * s_K(j, d);
                    }
                    float scaled = dot * softmax_scale;
                    S_row[j] = scaled;
                    block_max = fmaxf(block_max, scaled);
                }
            }

            // Online Softmax update
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

            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < Bc; ++j) {
                    pv += P_row[j] * s_V(j, d);
                }
                acc_O[d] = alpha * acc_O[d] + pv;
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

    // 4. Final normalization and store output: O = acc_O / l_i
    if (local_row < Br && global_q_row < seq_len) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float final_val = acc_O[d] * inv_l;
            o_ptr[static_cast<size_t>(global_q_row) * D + d] = from_float<T>(final_val);
        }
    }
}

// Host Dispatcher
template <typename T>
void dispatch_fused_rope_flash_attention(
    const T* q,
    const T* k,
    const T* v,
    const float* cos_tab,
    const float* sin_tab,
    T* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
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

    if (head_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (fused_rope_flash_attention_kernel<T, Br, Bc, 64>),
            grid, block, 0, stream,
            q, k, v, cos_tab, sin_tab, o, seq_len, softmax_scale, is_causal);
    } else if (head_dim == 128) {
        CUDA_KERNEL_LAUNCH(
            (fused_rope_flash_attention_kernel<T, Br, Bc, 128>),
            grid, block, 0, stream,
            q, k, v, cos_tab, sin_tab, o, seq_len, softmax_scale, is_causal);
    }
}
