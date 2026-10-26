#pragma once

#ifdef __CUDACC__
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math_constants.h>
#define CUDA_KERNEL_LAUNCH(fn, grid, block, smem, stream, ...) fn<<<grid, block, smem, stream>>>(__VA_ARGS__)

// ============================================================================
// Scalar type conversion helpers (device + host)
// Previously only defined in the host-stub branch below, which made every
// kernel that uses them fail to compile under nvcc.
// ============================================================================
template <typename T>
__host__ __device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}
template <>
__host__ __device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}
template <>
__host__ __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T>
__host__ __device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}
template <>
__host__ __device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}
template <>
__host__ __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

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

// 5. Hopper mbarrier expect-tx (TMA Hardware Transaction Tracking)
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar_ptr, uint32_t tx_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_ptr));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
        :
        : "r"(smem_addr), "r"(tx_bytes)
        : "memory"
    );
#else
    (void)bar_ptr;
    (void)tx_bytes;
#endif
}

// 6. Hopper Hardware TMA 2D Bulk Tensor Async Copy (cp.async.bulk.tensor)
__device__ __forceinline__ void cp_async_bulk_tensor_2d(void* smem_dst, const void* tma_desc, int crd0, int crd1, uint64_t* bar_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar_ptr));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :
        : "r"(smem_addr), "l"(tma_desc), "r"(crd0), "r"(crd1), "r"(bar_addr)
        : "memory"
    );
#else
    (void)smem_dst;
    (void)tma_desc;
    (void)crd0;
    (void)crd1;
    (void)bar_ptr;
#endif
}

// 7. Hopper Thread Block Clusters & Distributed Shared Memory (DSMEM)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define CLUSTER_DIMS(x, y, z) __cluster_dims__(x, y, z)
#else
#define CLUSTER_DIMS(x, y, z)
#endif

__device__ __forceinline__ void cluster_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ void cluster_arrive() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ void cluster_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
#else
    __syncthreads();
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
#include <cstring>
#include <algorithm>
#include <type_traits>

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
    uint32_t sign = (static_cast<uint32_t>(h.__x) & 0x8000) << 16;
    uint32_t exp  = (static_cast<uint32_t>(h.__x) & 0x7C00) >> 10;
    uint32_t mant = (static_cast<uint32_t>(h.__x) & 0x03FF);

    uint32_t f_bits = 0;
    if (exp == 0) {
        if (mant != 0) {
            // Subnormal in FP16 -> normalize in FP32
            exp = 127 - 14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            f_bits = sign | (exp << 23) | (mant << 13);
        } else {
            f_bits = sign; // Zero
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        f_bits = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normalized number: re-bias exponent (127 - 15 = 112)
        exp = exp + 112;
        f_bits = sign | (exp << 23) | (mant << 13);
    }
    float res;
    std::memcpy(&res, &f_bits, sizeof(float));
    return res;
}

inline __half __float2half(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));
    uint32_t sign = (f_bits >> 16) & 0x8000;
    int32_t exp  = static_cast<int32_t>((f_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f_bits & 0x007FFFFF);

    uint16_t h_bits = 0;
    if (exp <= 0) {
        if (exp < -10) {
            h_bits = static_cast<uint16_t>(sign);
        } else {
            mant = (mant | 0x00800000) >> (1 - exp);
            h_bits = static_cast<uint16_t>(sign | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
        }
    } else if (exp >= 31) {
        h_bits = static_cast<uint16_t>(sign | 0x7C00); // Inf
    } else {
        h_bits = static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x0FFF + ((mant >> 13) & 1)) >> 13));
    }
    __half h;
    h.__x = h_bits;
    return h;
}

inline float __bfloat162float(__nv_bfloat16 bf) {
    uint32_t f_bits = static_cast<uint32_t>(bf.__x) << 16;
    float res;
    std::memcpy(&res, &f_bits, sizeof(float));
    return res;
}

inline __nv_bfloat16 __float2bfloat16(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));
    // Round to nearest even:
    uint32_t lsb = (f_bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    f_bits += rounding_bias;
    __nv_bfloat16 bf;
    bf.__x = static_cast<uint16_t>(f_bits >> 16);
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

template <typename T>
inline T atomicAdd(T* address, T val) {
    T old = *address;
    *address += val;
    return old;
}
template <typename T>
inline T atomicMax(T* address, T val) {
    T old = *address;
    if (val > old) *address = val;
    return old;
}

// CUDA Driver/Runtime APIs
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };

struct cudaDeviceProp {
    int multiProcessorCount;
};

inline const char* cudaGetErrorString(cudaError_t) { return "Success"; }
inline const char* cudaGetErrorName(cudaError_t) { return "cudaSuccess"; }
inline cudaError_t cudaGetLastError() { return 0; }
inline cudaError_t cudaGetDevice(int* dev) { if (dev) *dev = 0; return 0; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) { if (prop) prop->multiProcessorCount = 1; return 0; }
template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t size) { 
    if (ptr) *ptr = reinterpret_cast<T*>(std::malloc(size)); 
    return 0; 
}
inline cudaError_t cudaFree(void* ptr) { 
    if (ptr) std::free(ptr); 
    return 0; 
}
inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (devPtr && count > 0) std::memset(devPtr, value, count);
    return 0;
}
inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t = nullptr) {
    if (devPtr && count > 0) std::memset(devPtr, value, count);
    return 0;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int) { 
    if (dst && src && count > 0) std::memcpy(dst, src, count); 
    return 0; 
}
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaDeviceReset() { return 0; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { if (s) *s = nullptr; return 0; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { if (e) *e = nullptr; return 0; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) { return 0; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { if (ms) *ms = 0.0f; return 0; }
template <typename F>
inline cudaError_t cudaOccupancyMaxPotentialBlockSize(int* min_grid, int* opt_block, F, size_t = 0, int = 0) {
    if (min_grid) *min_grid = 1;
    if (opt_block) *opt_block = 256;
    return 0;
}

// Host stubs for async primitives & Thread Block Clusters
#define CLUSTER_DIMS(x, y, z)

inline void cluster_sync() {}
inline void cluster_arrive() {}
inline void cluster_wait() {}

inline void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
}
inline void cp_async_commit() {}
template <int N> inline void cp_async_wait() {}
inline void mbarrier_init(uint64_t* bar_ptr, uint32_t thread_count) { if (bar_ptr) *bar_ptr = thread_count; }
inline void mbarrier_arrive(uint64_t*) {}
inline void mbarrier_arrive_expect_tx(uint64_t* bar_ptr, uint32_t tx_bytes) { if (bar_ptr) *bar_ptr += tx_bytes; }
inline void mbarrier_wait(uint64_t*, uint32_t) {}

// Software Hopper TMA 2D Descriptor for Host Testing & Compilation
struct HopperTMADescriptor2D {
    const void* global_address = nullptr;
    int global_dim0 = 0; // width (e.g. D or hidden)
    int global_dim1 = 0; // height (e.g. seq_len)
    int stride_bytes = 0;
    int box_dim0 = 0;    // tile width
    int box_dim1 = 0;    // tile height
    int element_size_bytes = 4;
};

inline void cp_async_bulk_tensor_2d(void* smem_dst, const HopperTMADescriptor2D* desc, int crd0, int crd1, uint64_t* bar_ptr) {
    if (!desc || !smem_dst || !desc->global_address) return;
    const char* g_base = reinterpret_cast<const char*>(desc->global_address);
    char* s_dst = reinterpret_cast<char*>(smem_dst);
    for (int r = 0; r < desc->box_dim1; ++r) {
        int gr = crd1 + r;
        if (gr < desc->global_dim1) {
            const char* src_row = g_base + static_cast<size_t>(gr) * desc->stride_bytes + static_cast<size_t>(crd0) * desc->element_size_bytes;
            char* dst_row = s_dst + static_cast<size_t>(r) * desc->box_dim0 * desc->element_size_bytes;
            std::memcpy(dst_row, src_row, desc->box_dim0 * desc->element_size_bytes);
        } else {
            char* dst_row = s_dst + static_cast<size_t>(r) * desc->box_dim0 * desc->element_size_bytes;
            std::memset(dst_row, 0, desc->box_dim0 * desc->element_size_bytes);
        }
    }
    (void)bar_ptr;
}

// ============================================================================
// OCP Microscaling & Standard FP8 (E4M3 & E5M2) Emulation
// ============================================================================
struct alignas(1) __nv_fp8_e4m3 { uint8_t __x; };
struct alignas(1) __nv_fp8_e5m2 { uint8_t __x; };

inline float __fp8_e4m3_to_float(__nv_fp8_e4m3 val) {
    if (val.__x == 0x7F || val.__x == 0xFF) return 0.0f; // NaN in E4M3
    int sign = (val.__x & 0x80) ? -1 : 1;
    int exp = (val.__x & 0x78) >> 3;
    int mant = val.__x & 0x07;
    if (exp == 0) {
        return sign * std::ldexp(static_cast<float>(mant), -9);
    }
    return sign * std::ldexp(1.0f + mant / 8.0f, exp - 7);
}

inline __nv_fp8_e4m3 __float_to_fp8_e4m3(float f) {
    __nv_fp8_e4m3 res{0};
    if (std::isnan(f)) return __nv_fp8_e4m3{0x7F};
    uint8_t sign = (f < 0.0f || (f == 0.0f && std::signbit(f))) ? 0x80 : 0x00;
    float abs_f = std::abs(f);
    if (abs_f > 448.0f) abs_f = 448.0f;
    int exp;
    float frac = std::frexp(abs_f, &exp);
    int biased_exp = exp + 6;
    if (biased_exp <= 0) {
        int mant = static_cast<int>(std::round(abs_f * 512.0f));
        res.__x = sign | static_cast<uint8_t>(std::min(7, mant));
    } else if (biased_exp >= 15) {
        res.__x = sign | 0x7E;
    } else {
        int mant = static_cast<int>(std::round((frac * 2.0f - 1.0f) * 8.0f));
        res.__x = sign | static_cast<uint8_t>((biased_exp << 3) | (mant & 0x07));
    }
    return res;
}

template <>
inline float to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 val) {
    return __fp8_e4m3_to_float(val);
}

template <>
inline __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float val) {
    return __float_to_fp8_e4m3(val);
}

// ============================================================================
// Bit-exact nvcuda::wmma Tensor Core Emulation for Host / macOS / Clangd
// Enables seamless C++ testing of 16x16x16 MMA Matrix Multiplication
// ============================================================================
namespace nvcuda {
namespace wmma {
    struct matrix_a {};
    struct matrix_b {};
    struct accumulator {};
    struct row_major {};
    struct col_major {};
    enum layout_t { mem_row_major = 0, mem_col_major = 1 };

    template <typename Use, int M, int N, int K, typename T, typename Layout = void>
    class fragment {
    public:
        static constexpr int num_elements = (std::is_same<Use, matrix_a>::value ? (M * K) : (std::is_same<Use, matrix_b>::value ? (K * N) : (M * N)));
        T x[num_elements];
        inline T& operator[](size_t i) { return x[i]; }
        inline const T& operator[](size_t i) const { return x[i]; }
    };

    template <typename Frag, typename T>
    inline void fill_fragment(Frag& frag, T val) {
        for (size_t i = 0; i < Frag::num_elements; ++i) {
            frag.x[i] = from_float<typename std::remove_reference<decltype(frag.x[0])>::type>(to_float(val));
        }
    }

    template <typename Use, int M, int N, int K, typename T, typename Layout>
    inline void load_matrix_sync(fragment<Use, M, N, K, T, Layout>& frag, const T* ptr, unsigned ldm) {
        if constexpr (std::is_same<Use, matrix_a>::value) {
            if constexpr (std::is_same<Layout, row_major>::value) {
                for (int r = 0; r < M; ++r) {
                    for (int c = 0; c < K; ++c) {
                        frag.x[r * K + c] = ptr[r * ldm + c];
                    }
                }
            } else {
                for (int r = 0; r < M; ++r) {
                    for (int c = 0; c < K; ++c) {
                        frag.x[r * K + c] = ptr[c * ldm + r];
                    }
                }
            }
        } else if constexpr (std::is_same<Use, matrix_b>::value) {
            if constexpr (std::is_same<Layout, row_major>::value) {
                for (int r = 0; r < K; ++r) {
                    for (int c = 0; c < N; ++c) {
                        frag.x[r * N + c] = ptr[r * ldm + c];
                    }
                }
            } else {
                for (int r = 0; r < K; ++r) {
                    for (int c = 0; c < N; ++c) {
                        frag.x[r * N + c] = ptr[c * ldm + r];
                    }
                }
            }
        } else if constexpr (std::is_same<Use, accumulator>::value) {
            for (int r = 0; r < M; ++r) {
                for (int c = 0; c < N; ++c) {
                    frag.x[r * N + c] = ptr[r * ldm + c];
                }
            }
        }
    }

    template <int M, int N, int K, typename T, typename T_Frag>
    inline void store_matrix_sync(T* ptr, const fragment<accumulator, M, N, K, T_Frag>& frag, unsigned ldm, layout_t layout) {
        if (layout == mem_row_major) {
            for (int r = 0; r < M; ++r) {
                for (int c = 0; c < N; ++c) {
                    ptr[r * ldm + c] = from_float<T>(to_float(frag.x[r * N + c]));
                }
            }
        } else {
            for (int r = 0; r < M; ++r) {
                for (int c = 0; c < N; ++c) {
                    ptr[c * ldm + r] = from_float<T>(to_float(frag.x[r * N + c]));
                }
            }
        }
    }

    template <int M, int N, int K, typename T_In, typename LayoutA, typename LayoutB, typename T_Out>
    inline void mma_sync(
        fragment<accumulator, M, N, K, T_Out>& d,
        const fragment<matrix_a, M, N, K, T_In, LayoutA>& a,
        const fragment<matrix_b, M, N, K, T_In, LayoutB>& b,
        const fragment<accumulator, M, N, K, T_Out>& c)
    {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = to_float(c.x[i * N + j]);
                for (int k = 0; k < K; ++k) {
                    sum += to_float(a.x[i * K + k]) * to_float(b.x[k * N + j]);
                }
                d.x[i * N + j] = from_float<T_Out>(sum);
            }
        }
    }
} // namespace wmma
} // namespace nvcuda

#endif
