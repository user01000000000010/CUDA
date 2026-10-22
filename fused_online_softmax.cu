#include "cuda_compat.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <cstdint>

// ============================================================================
// Zero-Overhead Diagnostic & Assertion Macros
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err),               \
                    cudaGetErrorName(err));                                    \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
    #define KERNEL_ASSERT(cond)                                                \
        do {                                                                   \
            if (!(cond)) {                                                     \
                printf("[KERNEL TRAP] %s:%d: Assertion '%s' failed.\n",        \
                       __FILE__, __LINE__, #cond);                             \
                __trap();                                                      \
            }                                                                  \
        } while (0)
#else
    #define KERNEL_ASSERT(cond) ((void)0)
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

// ============================================================================
// Warp & Block Reduction Primitives
// ============================================================================
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

// Padded shared memory structure to avoid any bank serialization
// Bank count is 32; stride by 33 elements if 2D, or pad structures by 1 word.
struct alignas(16) PaddedWarpSlot {
    float max_val;
    float sum_exp;
    float pad[2]; // Pads struct size to 16 bytes (4 words), preventing conflict
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
// Production Fused Online Softmax Kernel
// ============================================================================
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) fused_online_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int d_dim)
{
    // External or static shared memory staging for warp leaders
    __shared__ PaddedWarpSlot smem_slots[BLOCK_SIZE / WARP_SIZE];

    // Grid mapping: 1 Block per Row; Grid-stride loop over batch dimension
    for (int row = blockIdx.x; row < batch_size; row += gridDim.x) {
        const float* row_in = input + static_cast<size_t>(row) * d_dim;
        float* row_out = output + static_cast<size_t>(row) * d_dim;

        SoftmaxState local_state = {-CUDART_INF_F, 0.0f};

        // Determine 128-bit pointer vectorizability
        const uintptr_t in_addr = reinterpret_cast<uintptr_t>(row_in);
        const uintptr_t out_addr = reinterpret_cast<uintptr_t>(row_out);
        const bool is_aligned = ((in_addr | out_addr) & 0xF) == 0;

        const int vec_limit = is_aligned ? (d_dim / 4) * 4 : 0;

        // --------------------------------------------------------------------
        // PASS 1: Online Statistics Accumulation (Max & Rescaled Sum)
        // --------------------------------------------------------------------
        if (is_aligned) {
            const float4* row_in_v4 = reinterpret_cast<const float4*>(row_in);
            const int total_v4 = vec_limit / 4;

            #pragma unroll 2
            for (int idx = threadIdx.x; idx < total_v4; idx += BLOCK_SIZE) {
                float4 v = __ldg(&row_in_v4[idx]);
                local_state = update_softmax_state(local_state, v.x);
                local_state = update_softmax_state(local_state, v.y);
                local_state = update_softmax_state(local_state, v.z);
                local_state = update_softmax_state(local_state, v.w);
            }
        }

        // Peel scalar tail or handle unaligned fallback
        for (int idx = vec_limit + threadIdx.x; idx < d_dim; idx += BLOCK_SIZE) {
            float val = __ldg(&row_in[idx]);
            local_state = update_softmax_state(local_state, val);
        }

        // Cross-block reduction to produce exact row max and row sum
        SoftmaxState global_state = block_reduce_softmax<BLOCK_SIZE>(local_state, smem_slots);
        const float row_max = global_state.max_val;
        const float inv_sum = 1.0f / global_state.sum_exp;

        // --------------------------------------------------------------------
        // PASS 2: Normalization & Vectorized Output Materialization
        // --------------------------------------------------------------------
        if (is_aligned) {
            const float4* row_in_v4 = reinterpret_cast<const float4*>(row_in);
            float4* row_out_v4 = reinterpret_cast<float4*>(row_out);
            const int total_v4 = vec_limit / 4;

            #pragma unroll 2
            for (int idx = threadIdx.x; idx < total_v4; idx += BLOCK_SIZE) {
                float4 in_v = __ldg(&row_in_v4[idx]);
                float4 out_v;
                out_v.x = __expf(in_v.x - row_max) * inv_sum;
                out_v.y = __expf(in_v.y - row_max) * inv_sum;
                out_v.z = __expf(in_v.z - row_max) * inv_sum;
                out_v.w = __expf(in_v.w - row_max) * inv_sum;
                row_out_v4[idx] = out_v;
            }
        }

        // Scalar remainder store
        for (int idx = vec_limit + threadIdx.x; idx < d_dim; idx += BLOCK_SIZE) {
            float val = __ldg(&row_in[idx]);
            row_out[idx] = __expf(val - row_max) * inv_sum;
        }

        __syncthreads();
    }
}

// ============================================================================
// Driver Interface & Dynamic Occupancy Engine
// ============================================================================
extern "C" void launch_fused_online_softmax(
    const float* input,
    float* output,
    int batch_size,
    int d_dim,
    cudaStream_t stream = nullptr)
{
    constexpr int BLOCK_SIZE = 256;

    int min_grid_size = 0;
    int optimal_block_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &optimal_block_size,
        fused_online_softmax_kernel<BLOCK_SIZE>,
        0,
        0));

    // Choose grid envelope: Saturate SMs or map 1:1 with rows
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    // Determine grid size based on multi-processor count to maximize wave occupancy
    const int max_active_blocks = props.multiProcessorCount * (optimal_block_size / BLOCK_SIZE);
    const int grid_size = std::min(batch_size, std::max(1, max_active_blocks * 2));

    CUDA_KERNEL_LAUNCH(
        fused_online_softmax_kernel<BLOCK_SIZE>,
        grid_size, BLOCK_SIZE, 0, stream,
        input, output, batch_size, d_dim);

    CUDA_CHECK(cudaGetLastError());
}

// Conforming host interface defined in requirements
extern "C" void fused_online_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int d_dim)
{
    launch_fused_online_softmax(input, output, batch_size, d_dim);
}

// ============================================================================
// Verification & Benchmark Harness
// ============================================================================
int main() {
    constexpr int BATCH = 4096;
    constexpr int D = 8195; // Deliberate prime/unaligned dimension
    const size_t total_elements = static_cast<size_t>(BATCH) * D;
    const size_t bytes = total_elements * sizeof(float);

    printf("Allocating %.2f MB on Host & Device...\n", (bytes * 2) / (1024.0f * 1024.0f));

    float* h_in = (float*)malloc(bytes);
    float* h_out_gpu = (float*)malloc(bytes);
    float* h_out_cpu = (float*)malloc(bytes);

    for (size_t i = 0; i < total_elements; ++i) {
        h_in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Warmup
    launch_fused_online_softmax(d_in, d_out, BATCH, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Performance profiling loop
    constexpr int ITERS = 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        launch_fused_online_softmax(d_in, d_out, BATCH, D);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= ITERS;

    // 2 read passes (implicit cache) + 1 write pass
    double effective_bytes = static_cast<double>(total_elements) * sizeof(float) * 2.0; 
    double bandwidth_gb_s = (effective_bytes / (ms * 1e-3)) / 1e9;

    printf("Execution Finished.\n");
    printf("Avg Kernel Latency: %7.3f ms | Bandwidth: %6.2f GB/s\n", ms, bandwidth_gb_s);

    // Verify Output Integrity against CPU Reference
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    float max_abs_diff = 0.0f;
    for (int r = 0; r < std::min(BATCH, 16); ++r) {
        float cpu_max = -FLT_MAX;
        for (int c = 0; c < D; ++c) {
            cpu_max = std::max(cpu_max, h_in[r * D + c]);
        }
        float cpu_sum = 0.0f;
        for (int c = 0; c < D; ++c) {
            cpu_sum += std::exp(h_in[r * D + c] - cpu_max);
        }
        for (int c = 0; c < D; ++c) {
            float cpu_val = std::exp(h_in[r * D + c] - cpu_max) / cpu_sum;
            float gpu_val = h_out_gpu[r * D + c];
            max_abs_diff = std::max(max_abs_diff, std::abs(cpu_val - gpu_val));
        }
    }

    printf("Maximum Absolute Error (GPU vs CPU Reference): %e\n", max_abs_diff);

    // Teardown
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return (max_abs_diff < 1e-5f) ? 0 : 1;
}
