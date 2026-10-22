#pragma once
#include "cuda_compat.h"
#include <cstdint>
#include <algorithm>

// ============================================================================
// Alignment and Vector Load/Store Helper Traits
// ============================================================================
template <typename T>
struct VecType;

template <>
struct VecType<float> {
    using type = float4;
    static constexpr int size = 4;
};

template <>
struct VecType<__half> {
    using type = float4; // 128 bits = 8 halves
    static constexpr int size = 8;
};

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
struct VecType<__nv_bfloat16> {
    using type = float4; // 128 bits = 8 bf16s
    static constexpr int size = 8;
};
#endif

// ============================================================================
// Online Softmax Accumulator Primitive
// ============================================================================
struct alignas(8) SoftmaxState {
    float max_val;
    float sum_exp;
};

__device__ __forceinline__ SoftmaxState combine_softmax_state(
    const SoftmaxState a, 
    const SoftmaxState b) 
{
    if (a.max_val > b.max_val) {
        return { a.max_val, a.sum_exp + b.sum_exp * __expf(b.max_val - a.max_val) };
    } else {
        return { b.max_val, b.sum_exp + a.sum_exp * __expf(a.max_val - b.max_val) };
    }
}

__device__ __forceinline__ SoftmaxState update_softmax_state(
    SoftmaxState state, 
    float val) 
{
    if (val > state.max_val) {
        state.sum_exp = state.sum_exp * __expf(state.max_val - val) + 1.0f;
        state.max_val = val;
    } else {
        state.sum_exp += __expf(val - state.max_val);
    }
    return state;
}

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffU

__device__ __forceinline__ SoftmaxState warp_reduce_softmax(SoftmaxState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        SoftmaxState other;
        other.max_val = __shfl_down_sync(FULL_MASK, state.max_val, offset);
        other.sum_exp = __shfl_down_sync(FULL_MASK, state.sum_exp, offset);
        state = combine_softmax_state(state, other);
    }
    return state;
}

struct alignas(16) PaddedWarpSlot {
    float max_val;
    float sum_exp;
    float pad[2]; // Pads struct size to 16 bytes (prevents bank conflicts)
};

template <int BLOCK_SIZE>
__device__ __forceinline__ SoftmaxState block_reduce_softmax(
    SoftmaxState thread_state, 
    PaddedWarpSlot* shared_scratch) 
{
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    // 1. Intra-warp reduction
    thread_state = warp_reduce_softmax(thread_state);

    // 2. Warp leaders stage reduced values to shared memory
    if (lane_id == 0) {
        shared_scratch[warp_id].max_val = thread_state.max_val;
        shared_scratch[warp_id].sum_exp = thread_state.sum_exp;
    }
    __syncthreads();

    // 3. First warp gathers inter-warp data
    SoftmaxState block_state = {-CUDART_INF_F, 0.0f};
    if (warp_id == 0) {
        if (lane_id < NUM_WARPS) {
            block_state.max_val = shared_scratch[lane_id].max_val;
            block_state.sum_exp = shared_scratch[lane_id].sum_exp;
        }
        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            SoftmaxState other;
            other.max_val = __shfl_down_sync(FULL_MASK, block_state.max_val, offset);
            other.sum_exp = __shfl_down_sync(FULL_MASK, block_state.sum_exp, offset);
            block_state = combine_softmax_state(block_state, other);
        }

        // Broadcast back to scratch slot 0
        if (lane_id == 0) {
            shared_scratch[0].max_val = block_state.max_val;
            shared_scratch[0].sum_exp = block_state.sum_exp;
        }
    }
    __syncthreads();

    return { shared_scratch[0].max_val, shared_scratch[0].sum_exp };
}

// ============================================================================
// Templated Multi-Precision Fused Online Softmax Kernel
// ============================================================================
template <typename T, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) fused_online_softmax_kernel_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int d_dim)
{
    __shared__ PaddedWarpSlot smem_slots[BLOCK_SIZE / WARP_SIZE];
    constexpr int VEC_SIZE = VecType<T>::size;
    using VType = typename VecType<T>::type;

    for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
        const T* row_in = input + static_cast<size_t>(row) * d_dim;
        T* row_out = output + static_cast<size_t>(row) * d_dim;

        SoftmaxState local_state = {-CUDART_INF_F, 0.0f};

        const uintptr_t in_addr = reinterpret_cast<uintptr_t>(row_in);
        const uintptr_t out_addr = reinterpret_cast<uintptr_t>(row_out);
        const bool is_aligned = ((in_addr | out_addr) & 0xF) == 0;

        const int vec_limit = is_aligned ? (d_dim / VEC_SIZE) * VEC_SIZE : 0;

        // --------------------------------------------------------------------
        // PASS 1: Online Statistics Accumulation (Max & Rescaled Sum)
        // --------------------------------------------------------------------
        if (is_aligned) {
            const VType* row_in_vec = reinterpret_cast<const VType*>(row_in);
            const int total_vec = vec_limit / VEC_SIZE;

            #pragma unroll 2
            for (int idx = threadIdx.x; idx < total_vec; idx += BLOCK_SIZE) {
                VType raw = __ldg(&row_in_vec[idx]);
                const T* items = reinterpret_cast<const T*>(&raw);
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    local_state = update_softmax_state(local_state, to_float(items[v]));
                }
            }
        }

        // Peel scalar tail or handle unaligned
        for (int idx = vec_limit + threadIdx.x; idx < d_dim; idx += BLOCK_SIZE) {
            local_state = update_softmax_state(local_state, to_float(__ldg(&row_in[idx])));
        }

        // Block reduction
        SoftmaxState global_state = block_reduce_softmax<BLOCK_SIZE>(local_state, smem_slots);
        const float row_max = global_state.max_val;
        const float inv_sum = 1.0f / global_state.sum_exp;

        // --------------------------------------------------------------------
        // PASS 2: Normalization & Vectorized Output
        // --------------------------------------------------------------------
        if (is_aligned) {
            const VType* row_in_vec = reinterpret_cast<const VType*>(row_in);
            VType* row_out_vec = reinterpret_cast<VType*>(row_out);
            const int total_vec = vec_limit / VEC_SIZE;

            #pragma unroll 2
            for (int idx = threadIdx.x; idx < total_vec; idx += BLOCK_SIZE) {
                VType raw_in = __ldg(&row_in_vec[idx]);
                const T* in_items = reinterpret_cast<const T*>(&raw_in);
                VType raw_out;
                T* out_items = reinterpret_cast<T*>(&raw_out);

                #pragma unroll
                for (int v = 0; v < VEC_SIZE; ++v) {
                    float prob = __expf(to_float(in_items[v]) - row_max) * inv_sum;
                    out_items[v] = from_float<T>(prob);
                }
                row_out_vec[idx] = raw_out;
            }
        }

        // Scalar remainder store
        for (int idx = vec_limit + threadIdx.x; idx < d_dim; idx += BLOCK_SIZE) {
            float prob = __expf(to_float(__ldg(&row_in[idx])) - row_max) * inv_sum;
            row_out[idx] = from_float<T>(prob);
        }

        __syncthreads();
    }
}

// Host Dispatcher
template <typename T>
void dispatch_fused_online_softmax(
    const T* input,
    T* output,
    int batch_size,
    int d_dim,
    cudaStream_t stream = nullptr)
{
    constexpr int BLOCK_SIZE = 256;

    int min_grid_size = 0;
    int optimal_block_size = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &optimal_block_size,
        fused_online_softmax_kernel_impl<T, BLOCK_SIZE>,
        0,
        0);

    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    const int max_active_blocks = props.multiProcessorCount * (optimal_block_size / BLOCK_SIZE);
    const int grid_size = std::min(batch_size, std::max(1, max_active_blocks * 2));

    CUDA_KERNEL_LAUNCH(
        (fused_online_softmax_kernel_impl<T, BLOCK_SIZE>),
        grid_size, BLOCK_SIZE, 0, stream,
        input, output, batch_size, d_dim);
}
