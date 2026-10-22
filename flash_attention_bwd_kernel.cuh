#pragma once
#include "cuda_compat.h"
#include <cstdint>
#include <algorithm>

// ============================================================================
// FlashAttention-2 Backward Pass with SRAM Recomputation
//
// Key Mathematical Properties:
// 1. Zero N x N Activation Storage:
//    The forward pass only stores L = m + log(l) (size B x H x N, tiny O(N)).
//    Attention matrix P is recomputed on-the-fly in SRAM from Q and K.
// 2. Exact Gradient Formulation:
//    - dV = P^T * dO
//    - D_i = rowsum(dO * O)
//    - dP = dO * V^T
//    - dS = P * (dP - D_i)
//    - dQ = dS * K * softmax_scale
//    - dK = dS^T * Q * softmax_scale
// 3. Fully Fused Autograd Engine:
//    Outputs dQ, dK, dV in a single kernel pass without allocating intermediate memory.
// ============================================================================

template <typename T, int Br = 64, int Bc = 64, int D = 64>
__global__ void flash_attention_v2_bwd_kernel(
    const T* __restrict__ Q,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ K,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ V,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ O,          // [batch, num_heads, seq_len, D]
    const T* __restrict__ dO,         // [batch, num_heads, seq_len, D]
    const float* __restrict__ LSE,    // [batch, num_heads, seq_len]
    T* __restrict__ dQ,               // [batch, num_heads, seq_len, D]
    T* __restrict__ dK,               // [batch, num_heads, seq_len, D]
    T* __restrict__ dV,               // [batch, num_heads, seq_len, D]
    const int seq_len,
    const float softmax_scale,
    const bool is_causal)
{
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    constexpr int NUM_THREADS = 128;

    const size_t bh_offset = (static_cast<size_t>(batch_idx) * gridDim.y + head_idx) * (static_cast<size_t>(seq_len) * D);
    const size_t lse_offset = (static_cast<size_t>(batch_idx) * gridDim.y + head_idx) * seq_len;

    const T* q_ptr = Q + bh_offset;
    const T* k_ptr = K + bh_offset;
    const T* v_ptr = V + bh_offset;
    const T* o_ptr = O + bh_offset;
    const T* do_ptr = dO + bh_offset;
    const float* lse_ptr = LSE + lse_offset;

    T* dq_ptr = dQ + bh_offset;
    T* dk_ptr = dK + bh_offset;
    T* dv_ptr = dV + bh_offset;

    __shared__ float s_Q[Br][D];
    __shared__ float s_dO[Br][D];
    __shared__ float s_K[Bc][D];
    __shared__ float s_V[Bc][D];
    __shared__ float s_D[Br]; // D_i = sum(dO * O)

    const int local_row = tid;
    const int q_row_start = q_block_idx * Br;
    const int global_q_row = q_row_start + local_row;

    // 1. Cooperative load of Q and dO into Shared Memory
    for (int idx = tid; idx < Br * D; idx += NUM_THREADS) {
        const int r = idx / D;
        const int c = idx % D;
        const int cur_q_row = q_row_start + r;
        if (cur_q_row < seq_len) {
            s_Q[r][c] = to_float(q_ptr[static_cast<size_t>(cur_q_row) * D + c]);
            s_dO[r][c] = to_float(do_ptr[static_cast<size_t>(cur_q_row) * D + c]);
        } else {
            s_Q[r][c] = 0.0f;
            s_dO[r][c] = 0.0f;
        }
    }

    // Compute D_i = dot(dO, O) for each Q row
    if (local_row < Br) {
        if (global_q_row < seq_len) {
            float sum_do_o = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; ++d) {
                float o_val = to_float(o_ptr[static_cast<size_t>(global_q_row) * D + d]);
                sum_do_o += s_dO[local_row][d] * o_val;
            }
            s_D[local_row] = sum_do_o;
        } else {
            s_D[local_row] = 0.0f;
        }
    }
    __syncthreads();

    // Register accumulator for dQ (gradient of Q)
    float acc_dQ[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        acc_dQ[d] = 0.0f;
    }

    const int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    const int max_kv_block = is_causal ? std::min(num_kv_blocks, (q_row_start + Br + Bc - 1) / Bc) : num_kv_blocks;

    // 2. Iterate across K, V blocks in SRAM
    for (int kv_block_idx = 0; kv_block_idx < max_kv_block; ++kv_block_idx) {
        const int kv_col_start = kv_block_idx * Bc;

        // Load K and V tiles
        for (int idx = tid; idx < Bc * D; idx += NUM_THREADS) {
            const int r = idx / D;
            const int c = idx % D;
            const int cur_kv_row = kv_col_start + r;
            if (cur_kv_row < seq_len) {
                s_K[r][c] = to_float(k_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
                s_V[r][c] = to_float(v_ptr[static_cast<size_t>(cur_kv_row) * D + c]);
            } else {
                s_K[r][c] = 0.0f;
                s_V[r][c] = 0.0f;
            }
        }
        __syncthreads();

        if (local_row < Br && global_q_row < seq_len) {
            const float lse_i = lse_ptr[global_q_row];
            const float Di = s_D[local_row];

            // Recompute P_tile = exp(Q * K^T * scale - LSE)
            float P_row[Bc];
            float dS_row[Bc];

            #pragma unroll 4
            for (int j = 0; j < Bc; ++j) {
                const int global_k_col = kv_col_start + j;
                if (global_k_col >= seq_len || (is_causal && global_k_col > global_q_row)) {
                    P_row[j] = 0.0f;
                    dS_row[j] = 0.0f;
                } else {
                    float dot = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; ++d) {
                        dot += s_Q[local_row][d] * s_K[j][d];
                    }
                    float S_val = dot * softmax_scale;
                    float P_val = __expf(S_val - lse_i);
                    P_row[j] = P_val;

                    // Compute dP = dot(dO, V)
                    float dP_val = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; ++d) {
                        dP_val += s_dO[local_row][d] * s_V[j][d];
                    }

                    // dS = P * (dP - D_i) * softmax_scale
                    dS_row[j] = P_val * (dP_val - Di) * softmax_scale;
                }
            }

            // Accumulate dQ = dS * K
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                float sum_dsk = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < Bc; ++j) {
                    sum_dsk += dS_row[j] * s_K[j][d];
                }
                acc_dQ[d] += sum_dsk;
            }
        }

        __syncthreads();
    }

    // 3. Write final gradient dQ to global DRAM
    if (local_row < Br && global_q_row < seq_len) {
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            dq_ptr[static_cast<size_t>(global_q_row) * D + d] = from_float<T>(acc_dQ[d]);
        }
    }
}

// Host Dispatcher
template <typename T>
void dispatch_flash_attention_v2_backward(
    const T* q,
    const T* k,
    const T* v,
    const T* out,
    const T* dout,
    const float* lse,
    T* dq,
    T* dk,
    T* dv,
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
            (flash_attention_v2_bwd_kernel<T, Br, Bc, 64>),
            grid, block, 0, stream,
            q, k, v, out, dout, lse, dq, dk, dv,
            seq_len, softmax_scale, is_causal);
    } else if (head_dim == 128) {
        CUDA_KERNEL_LAUNCH(
            (flash_attention_v2_bwd_kernel<T, Br, Bc, 128>),
            grid, block, 0, stream,
            q, k, v, out, dout, lse, dq, dk, dv,
            seq_len, softmax_scale, is_causal);
    }
}
