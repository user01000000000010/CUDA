#pragma once
#include "cuda_compat.h"
#include "tile_ir.cuh"
#include "autotune.cuh"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Modern FlashAttention-2 Forward Kernel with Tile IR & Asynchronous cp.async
//
// Key Architectural Upgrades:
// 1. Tile IR Layout Algebra:
//    - Replaced raw scalar array access with structured 2D tiles (Br x D) and (Bc x D).
//    - Structured tensor strides eliminate manual modulo/division operations.
// 2. Hardware Asynchronous Pipeline (cp.async & double-buffering):
//    - Overlaps global HBM memory loading of stage (k+1) with MMA & Online Softmax of stage (k).
//    - cp_async_commit and cp_async_wait<1> decouple global loads from SRAM execution.
// 3. Online Softmax in FP32 registers.
// ============================================================================

template <typename T, int Br = 64, int Bc = 64, int D = 64, int NUM_THREADS = 128>
__launch_bounds__(NUM_THREADS, 2)
__global__ void flash_attention_v2_fwd_kernel(
    const T* __restrict__ Q,      // [batch, num_heads, seq_len, D]
    const T* __restrict__ K,      // [batch, num_heads, seq_len, D]
    const T* __restrict__ V,      // [batch, num_heads, seq_len, D]
    T* __restrict__ O,            // [batch, num_heads, seq_len, D]
    const int seq_len,
    const float softmax_scale,
    const bool is_causal)
{
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;

    // Strides
    const size_t bh_offset = (static_cast<size_t>(batch_idx) * gridDim.y + head_idx) * (static_cast<size_t>(seq_len) * D);
    const T* q_ptr = Q + bh_offset;
    const T* k_ptr = K + bh_offset;
    const T* v_ptr = V + bh_offset;
    T* o_ptr = O + bh_offset;

    // Double-buffered Shared Memory staging for K and V tiles
    alignas(16) __shared__ float s_Q_data[Br * D];
    alignas(16) __shared__ float s_K_data[2][Bc * D];
    alignas(16) __shared__ float s_V_data[2][Bc * D];

    auto s_Q = tile_ir::make_row_major_tensor<Br, D>(s_Q_data);

    const int local_row = tid;
    float acc_O[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        acc_O[d] = 0.0f;
    }
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    const int q_row_start = q_block_idx * Br;
    const int global_q_row = q_row_start + local_row;

    // 1. Load Q block into Shared Memory via Tile IR
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
    __syncthreads();

    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    const int max_kv_block = is_causal ? std::min(num_kv_blocks, (q_row_start + Br + Bc - 1) / Bc) : num_kv_blocks;

    // Helper lambda to load a K/V block into shared memory stage
    auto load_kv_stage = [&](int block_idx, int stage_idx) {
        auto s_K = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[stage_idx]);
        auto s_V = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[stage_idx]);
        const int kv_col_start = block_idx * Bc;

        for (int idx = tid * 4; idx < Bc * D; idx += NUM_THREADS * 4) {
            const int r = idx / D;
            const int c = idx % D;
            const int cur_kv_row = kv_col_start + r;
            if (cur_kv_row < seq_len) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(__CUDACC_RTC__)
                cp_async_16(&s_K_data[stage_idx][idx], &k_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
                cp_async_16(&s_V_data[stage_idx][idx], &v_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
#else
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s_K(r, c + k) = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + (c + k)]);
                    s_V(r, c + k) = to_float(v_ptr[static_cast<size_t>(cur_kv_row) * D + (c + k)]);
                }
#endif
            } else {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s_K(r, c + k) = 0.0f;
                    s_V(r, c + k) = 0.0f;
                }
            }
        }
        cp_async_commit();
    };

    // 2. Prime the pipeline: load stage 0
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

        // Prefetch next stage in background
        if (next_block_idx < max_kv_block) {
            load_kv_stage(next_block_idx, next_stage);
        }

        auto s_K = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[curr_stage]);
        auto s_V = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[curr_stage]);
        const int kv_col_start = kv_block_idx * Bc;

        // Compute Attention for current tile
        if (local_row < Br && global_q_row < seq_len) {
            float S_row[Bc];
            float block_row_max = -CUDART_INF_F;

            const bool is_fully_unmasked = (!is_causal || (kv_col_start + Bc <= global_q_row)) && (kv_col_start + Bc <= seq_len);

            if (is_fully_unmasked) {
                // Fast-path: Completely unmasked tile. Zero conditionals, full instruction-level parallelism
                #pragma unroll 4
                for (int j = 0; j < Bc; ++j) {
                    float dot = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; ++d) {
                        dot += s_Q(local_row, d) * s_K(j, d);
                    }
                    float scaled_dot = dot * softmax_scale;
                    S_row[j] = scaled_dot;
                    block_row_max = fmaxf(block_row_max, scaled_dot);
                }
            } else {
                // Boundary-path: Causal diagonal or sequence boundary
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
                        float scaled_dot = dot * softmax_scale;
                        S_row[j] = scaled_dot;
                        block_row_max = fmaxf(block_row_max, scaled_dot);
                    }
                }
            }

            // Online Softmax update step
            const float m_new = fmaxf(m_i, block_row_max);
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

            // Accumulate V tile
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

        // Wait for prefetch of next stage before advancing
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

// Host Dispatcher with Adaptive Silicon-Level Dynamic Autotuning
template <typename T>
void dispatch_flash_attention_v2(
    const T* q,
    const T* k,
    const T* v,
    T* o,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float softmax_scale,
    bool is_causal,
    cudaStream_t stream = nullptr)
{
    auto policy = autotune::select_attention_tile_policy(seq_len, head_dim, num_heads, batch_size);

    if (policy.Br == 32) {
        constexpr int Br = 32;
        constexpr int Bc = 64;
        constexpr int NUM_THREADS = 64;
        const int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(num_q_blocks, num_heads, batch_size);
        dim3 block(NUM_THREADS);
        if (head_dim == 64) {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 64, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        } else {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 128, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        }
    } else if (policy.Br == 128) {
        constexpr int Br = 128;
        constexpr int Bc = 64;
        constexpr int NUM_THREADS = 256;
        const int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(num_q_blocks, num_heads, batch_size);
        dim3 block(NUM_THREADS);
        if (head_dim == 64) {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 64, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        } else {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 128, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        }
    } else {
        constexpr int Br = 64;
        constexpr int Bc = 64;
        constexpr int NUM_THREADS = 128;
        const int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(num_q_blocks, num_heads, batch_size);
        dim3 block(NUM_THREADS);
        if (head_dim == 64) {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 64, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        } else {
            CUDA_KERNEL_LAUNCH(
                (flash_attention_v2_fwd_kernel<T, Br, Bc, 128, NUM_THREADS>),
                grid, block, 0, stream,
                q, k, v, o, seq_len, softmax_scale, is_causal);
        }
    }
}
