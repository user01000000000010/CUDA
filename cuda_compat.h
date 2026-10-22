#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
#include <cuda_bf16.h>
#endif
#include <math_constants.h>
#define CUDA_KERNEL_LAUNCH(fn, grid, block, smem, stream, ...) fn<<<grid, block, smem, stream>>>(__VA_ARGS__)

// ============================================================================
// Modern CUDA Block/Tensor & Asynchronous Pipeline Primitives
// ============================================================================

// 1. Asynchronous Copy from Global to Shared Memory (cp.async 16-byte bulk)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
#else
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}

// 2. Commit async group to pipeline
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

// 3. Wait for at most N async groups to remain pending
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
#endif
}

// 4. Hopper Hardware Asynchronous Barrier (mbarrier)
__device__ __forceinline__ void mbarrier_init(uint64_t* bar_ptr, uint32_t thread_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_ptr));
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(thread_count)
        : "memory"
    );
#else
    *bar_ptr = thread_count;
#endif
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_ptr));
    asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];\n"
        :
        : "r"(smem_addr)
        : "memory"
    );
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar_ptr, uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_ptr));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p nanosleep.u32 20;\n\t"
        "@!p bra.uni LAB_WAIT;\n"
        "}\n"
        :
        : "r"(smem_addr), "r"(phase)
        : "memory"
    );
#else
    (void)bar_ptr;
    (void)phase;
#endif
}

#else
// ============================================================================
// Host/macOS Clangd Language Server Emulation Stubs
// Eliminates diagnostics/errors in IDE while preserving raw CUDA on target NVCC
// ============================================================================
#include <cmath>
#include <cstdint>
#include <cstdlib>

#ifndef __global__
#define __global__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __restrict__
#define __restrict__
#endif
#ifndef __shared__
#define __shared__ static
#endif
#ifndef CUDART_INF_F
#define CUDART_INF_F INFINITY
#endif

// Kernel launch syntax mock for host C++ parsing
#define CUDA_KERNEL_LAUNCH(fn, grid, block, smem, stream, ...) fn(__VA_ARGS__)

// CUDA built-in vector types
struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};
struct float4 { float x, y, z, w; };

// Half-precision & BFloat16 stubs
struct __half {
    uint16_t __x;
};
struct __nv_bfloat16 {
    uint16_t __x;
};

inline float __half2float(__half h) {
    (void)h;
    return 0.0f;
}
inline __half __float2half(float f) {
    (void)f;
    __half h;
    h.__x = 0;
    return h;
}
inline float __bfloat162float(__nv_bfloat16 bf) {
    (void)bf;
    return 0.0f;
}
inline __nv_bfloat16 __float2bfloat16(float f) {
    (void)f;
    __nv_bfloat16 bf;
    bf.__x = 0;
    return bf;
}

template <typename T>
inline float to_float(T val) {
    return static_cast<float>(val);
}
template <>
inline float to_float<__half>(__half val) {
    return __half2float(val);
}
template <>
inline float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T>
inline T from_float(float val) {
    return static_cast<T>(val);
}
template <>
inline __half from_float<__half>(float val) {
    return __float2half(val);
}
template <>
inline __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

// Built-in CUDA variables & warp intrinsics
static const uint3 threadIdx = {0, 0, 0};
static const uint3 blockIdx = {0, 0, 0};
static const dim3 gridDim(1, 1, 1);
static const dim3 blockDim(1, 1, 1);

inline void __syncthreads() {}
inline void __trap() {}
template <typename T> inline T __shfl_down_sync(unsigned int, T val, int) { return val; }
template <typename T> inline T __shfl_sync(unsigned int, T val, int) { return val; }
inline float __expf(float val) { return std::exp(val); }
inline float rsqrtf(float val) { return 1.0f / std::sqrt(val); }
template <typename T> inline T __ldg(const T* ptr) { return *ptr; }

// CUDA Driver/Runtime APIs
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

struct cudaDeviceProp {
    int multiProcessorCount;
};

inline const char* cudaGetErrorString(cudaError_t) { return "Success"; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline cudaError_t cudaGetLastError() { return 0; }
inline cudaError_t cudaGetDevice(int* dev) { if (dev) *dev = 0; return 0; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) { if (prop) prop->multiProcessorCount = 1; return 0; }
template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t) { if (ptr) *ptr = nullptr; return 0; }
inline cudaError_t cudaFree(void*) { return 0; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return 0; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return 0; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return 0; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { if (ms) *ms = 0.0f; return 0; }
template <typename F>
inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int* min_grid, int* opt_block, F, size_t = 0, int = 0) {
    if (min_grid) *min_grid = 1;
    if (opt_block) *opt_block = 256;
    return 0;
}

// Host stubs for async primitives
inline void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
}
inline void cp_async_commit() {}
template <int N> inline void cp_async_wait() {}
inline void mbarrier_init(uint64_t* bar_ptr, uint32_t thread_count) { if (bar_ptr) *bar_ptr = thread_count; }
inline void mbarrier_arrive(uint64_t*) {}
inline void mbarrier_wait(uint64_t*, uint32_t) {}

#endif
