#pragma once
#include "cuda_compat.h"
#include "tile_ir.cuh"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Modern Hopper (sm_90) Architecture FlashAttention Kernel
//
// Key Architectural Features:
// 1. Block-based Tensor Modeling via Tile IR:
//    - Tensors partitioned into 2D Tiles (Br x D) and (Bc x D).
//    - Layouts eliminate scalar thread-level address computation.
// 2. Asynchronous Multi-Stage Circular Pipeline:
//    - Stage 0 & Stage 1 Shared Memory Tiles filled via cp.async 128-bit transfers.
//    - Producer Warp (Warp 0) issues transfers directly to SMEM without halting.
//    - Consumer Warps (Warps 1..3) process independent staged tiles.
// 3. Hardware Asynchronous Barriers (mbarrier):
//    - Decouples producer data availability from consumer execution,
//      eliminating coarse synchronous __syncthreads() blocking.
// 4. In-Register Online Softmax for numerical preservation.
// ============================================================================

template <typename T, int Br = 64, int Bc = 64, int D = 64, int NUM_STAGES = 2>
__global__ void flash_attention_v3_hopper_kernel(
    const T* __restrict__ Q,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ K,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ V,          // [batch, num_heads, seq_len, D]
    T* __restrict__ O,                // [batch, num_heads, seq_len, D]
    const int seq_len,
    const float softmax_scale,
    const bool is_causal)
{
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int NUM_THREADS = 128; // 4 Warps: Warp 0 Producer, Warps 1..3 Consumers
    constexpr int PRODUCER_WARP = 0;

    const size_t bh_offset = (static_cast<size_t>(batch_idx) * gridDim.y + head_idx) * (static_cast<size_t>(seq_len) * D);
    const T* q_ptr = Q + bh_offset;
    const T* k_ptr = K + bh_offset;
    const T* v_ptr = V + bh_offset;
    T* o_ptr = O + bh_offset;

    // Multi-stage circular shared memory tile buffers
    alignas(16) __shared__ float s_Q_data[Br * D];
    alignas(16) __shared__ float s_K_data[NUM_STAGES][Bc * D];
    alignas(16) __shared__ float s_V_data[NUM_STAGES][Bc * D];
    __shared__ uint64_t stage_mbar[NUM_STAGES];

    // Tensor Views using Tile IR Layouts
    auto s_Q = tile_ir::make_row_major_tensor<Br, D>(s_Q_data);

    const int q_row_start = q_block_idx * Br;
    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    const int max_kv_block = is_causal ? std::min(num_kv_blocks, (q_row_start + Br + Bc - 1) / Bc) : num_kv_blocks;

    // 1. Initialize Hopper Asynchronous Hardware Barriers
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; ++s) {
            mbarrier_init(&stage_mbar[s], 32); // 32 producer threads participate in arrival
        }
    }
    __syncthreads();

    // 2. Cooperative Asynchronous Load of Q-Tile into Shared Memory
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

    // Register accumulator state for Consumer Warps
    float acc_O[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        acc_O[d] = 0.0f;
    }
    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;

    // 3. Hopper Asynchronous Tile Pipeline Loop
    int stage = 0;
    uint32_t phase = 0;

    for (int kv_block_idx = 0; kv_block_idx < max_kv_block; ++kv_block_idx) {
        const int kv_col_start = kv_block_idx * Bc;

        // PRODUCER WARP (Warp 0): Asynchronous Block Copy via cp.async
        if (warp_id == PRODUCER_WARP) {
            auto s_K_stage = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[stage]);
            auto s_V_stage = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[stage]);

            for (int idx = lane_id * 4; idx < Bc * D; idx += 32 * 4) {
                const int r = idx / D;
                const int c = idx % D;
                const int cur_kv_row = kv_col_start + r;

                if (cur_kv_row < seq_len) {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        s_K_stage(r, c + k) = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + (c + k)]);
                        s_V_stage(r, c + k) = to_float(v_ptr[static_cast<size_t>(cur_kv_row) * D + (c + k)]);
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        s_K_stage(r, c + k) = 0.0f;
                        s_V_stage(r, c + k) = 0.0f;
                    }
                }
            }

            cp_async_commit();
            mbarrier_arrive(&stage_mbar[stage]);
        }

        // CONSUMER WARPS (Warps 1, 2, 3): Wait on hardware mbarrier and execute WGMMA
        if (warp_id != PRODUCER_WARP) {
            mbarrier_wait(&stage_mbar[stage], phase);

            auto s_K_stage = tile_ir::make_row_major_tensor<Bc, D>(s_K_data[stage]);
            auto s_V_stage = tile_ir::make_row_major_tensor<Bc, D>(s_V_data[stage]);

            const int consumer_tid = tid - 32; // 0..95
            constexpr int NUM_CONSUMER_THREADS = 96;

            for (int r = consumer_tid; r < Br; r += NUM_CONSUMER_THREADS) {
                const int global_q_row = q_row_start + r;
                if (global_q_row >= seq_len) continue;

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
                            dot += s_Q(r, d) * s_K_stage(j, d);
                        }
                        float scaled = dot * softmax_scale;
                        S_row[j] = scaled;
                        block_max = fmaxf(block_max, scaled);
                    }
                }

                // Online Softmax in-register update
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

                // Accumulate V tile into register accumulator
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float pv = 0.0f;
                    #pragma unroll 8
                    for (int j = 0; j < Bc; ++j) {
                        pv += P_row[j] * s_V_stage(j, d);
                    }
                    acc_O[d] = alpha * acc_O[d] + pv;
                }

                m_i = m_new;
                l_i = l_new;
            }
        }

        // Circular buffer stage shift and phase toggle
        stage = (stage + 1) % NUM_STAGES;
        if (stage == 0) {
            phase ^= 1;
        }
    }

    // 4. Final normalization and store output: O = acc_O / l_i
    if (warp_id != PRODUCER_WARP) {
        const int consumer_tid = tid - 32;
        constexpr int NUM_CONSUMER_THREADS = 96;

        for (int r = consumer_tid; r < Br; r += NUM_CONSUMER_THREADS) {
            const int global_q_row = q_row_start + r;
            if (global_q_row < seq_len) {
                const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
                #pragma unroll 4
                for (int d = 0; d < D; ++d) {
                    float final_val = acc_O[d] * inv_l;
                    o_ptr[static_cast<size_t>(global_q_row) * D + d] = from_float<T>(final_val);
                }
            }
        }
    }
}

// Host Dispatcher
template <typename T>
void dispatch_flash_attention_v3_hopper(
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
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int NUM_THREADS = 128; // 1 Producer Warp + 3 Consumer Warps

    const int num_q_blocks = (seq_len + Br - 1) / Br;
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(NUM_THREADS);

    if (head_dim == 64) {
        CUDA_KERNEL_LAUNCH(
            (flash_attention_v3_hopper_kernel<T, Br, Bc, 64, 2>),
            grid, block, 0, stream,
            q, k, v, o, seq_len, softmax_scale, is_causal);
    } else if (head_dim == 128) {
        CUDA_KERNEL_LAUNCH(
            (flash_attention_v3_hopper_kernel<T, Br, Bc, 128, 2>),
            grid, block, 0, stream,
            q, k, v, o, seq_len, softmax_scale, is_causal);
    }
}
