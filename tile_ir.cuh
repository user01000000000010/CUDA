#pragma once
#include <cstdint>
#include "cuda_compat.h"

// ============================================================================
// Tile IR / CuTe Lightweight Layout & Tensor Abstraction for CUDA
//
// Key Concepts:
// 1. Compile-Time Layout Algebra:
//    Encodes Shape(M, N) and Stride(dM, dN) as template types, enabling
//    zero-overhead address generation during kernel execution.
// 2. Hardware-Aligned TiledCopy & TiledMMA Mapping:
//    Maps 2D sub-tiles (Tiles) cleanly to Warp Groups and threads,
//    eliminating manual division, modulo, and scalar index calculations.
// 3. Bank-Conflict-Free 2D XOR Swizzling:
//    Transforms shared memory physical addresses such that consecutive threads
//    in a warp never access the same 4-byte SMEM bank (32 banks total).
// ============================================================================

namespace tile_ir {

// Compile-Time Stride & Shape Descriptor
template <int M, int N>
struct Shape {
    static constexpr int kRows = M;
    static constexpr int kCols = N;
    static constexpr int kSize = M * N;
};

template <int StrideM, int StrideN>
struct Stride {
    static constexpr int kStrideM = StrideM;
    static constexpr int kStrideN = StrideN;
};

// 2D Tensor View over raw memory
template <typename T, typename T_Shape, typename T_Stride>
struct TensorView {
    T* data;
    using ShapeType = T_Shape;
    using StrideType = T_Stride;

    __host__ __device__ __forceinline__ explicit TensorView(T* ptr) : data(ptr) {}

    __host__ __device__ __forceinline__ T& operator()(int r, int c) {
        return data[r * T_Stride::kStrideM + c * T_Stride::kStrideN];
    }

    __host__ __device__ __forceinline__ const T& operator()(int r, int c) const {
        return data[r * T_Stride::kStrideM + c * T_Stride::kStrideN];
    }

    __host__ __device__ __forceinline__ T* ptr(int r = 0, int c = 0) {
        return data + (r * T_Stride::kStrideM + c * T_Stride::kStrideN);
    }

    __host__ __device__ __forceinline__ const T* ptr(int r = 0, int c = 0) const {
        return data + (r * T_Stride::kStrideM + c * T_Stride::kStrideN);
    }
};

// ----------------------------------------------------------------------------
// Swizzled 2D Shared Memory View
// Hardware Principle: 32 banks, 4 bytes each = 128 bytes per SMEM phase.
// Swizzle: col' = col ^ ((row >> Shift) & Mask)
// Completely removes bank conflicts for 128-bit vector memory operations.
// ----------------------------------------------------------------------------
template <typename T, int Rows, int Cols, int B = 3, int M = 3, int S = 0>
struct SwizzledTensorView {
    T* data;
    static constexpr int kMask = (1 << B) - 1;

    __host__ __device__ __forceinline__ explicit SwizzledTensorView(T* ptr) : data(ptr) {}

    __host__ __device__ __forceinline__ static int swizzle_offset(int r, int c) {
        // XOR bitwise swizzling between row and column phase bits
        int swizzled_col = c ^ (((r >> S) & kMask) << M);
        return r * Cols + swizzled_col;
    }

    __host__ __device__ __forceinline__ T& operator()(int r, int c) {
        return data[swizzle_offset(r, c)];
    }

    __host__ __device__ __forceinline__ const T& operator()(int r, int c) const {
        return data[swizzle_offset(r, c)];
    }

    __host__ __device__ __forceinline__ T* ptr(int r, int c) {
        return data + swizzle_offset(r, c);
    }
};

// Utility to create row-major tensor views
template <int M, int N, typename T>
__host__ __device__ __forceinline__ auto make_row_major_tensor(T* ptr) {
    return TensorView<T, Shape<M, N>, Stride<N, 1>>(ptr);
}

// Utility to create swizzled tensor views
template <int M, int N, int B = 3, int Shift = 0, typename T>
__host__ __device__ __forceinline__ auto make_swizzled_tensor(T* ptr) {
    return SwizzledTensorView<T, M, N, B, 0, Shift>(ptr);
}

// 2D Block Tile Partitioning helper (eliminates threadIdx modulo/division)
template <int TileM, int TileN, int Threads>
struct TiledPartition {
    static constexpr int kThreadsPerRow = TileN / 4; // 128-bit vectorization (4 elements)
    static constexpr int kRowsPerStep = Threads / (kThreadsPerRow > 0 ? kThreadsPerRow : 1);

    static __host__ __device__ __forceinline__ int get_row(int tid) {
        return (kThreadsPerRow > 0) ? (tid / kThreadsPerRow) : tid;
    }

    static __host__ __device__ __forceinline__ int get_col(int tid) {
        return (kThreadsPerRow > 0) ? ((tid % kThreadsPerRow) * 4) : 0;
    }
};

// ============================================================================
// Hardware-Aligned TiledMMA Tensor Core Abstraction
// Encapsulates 16x16x16 Warp Matrix Multiply & Accumulate (WMMA) execution
// ============================================================================
template <int TileM = 16, int TileN = 16, int TileK = 16, typename T_In = __half, typename T_Acc = float>
struct TiledMMA {
    using FragmentA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TileM, TileN, TileK, T_In, nvcuda::wmma::row_major>;
    using FragmentB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TileM, TileN, TileK, T_In, nvcuda::wmma::col_major>;
    using FragmentB_Row = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TileM, TileN, TileK, T_In, nvcuda::wmma::row_major>;
    using FragmentC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TileM, TileN, TileK, T_Acc>;

    __host__ __device__ __forceinline__ static void zero_accumulator(FragmentC& c) {
        nvcuda::wmma::fill_fragment(c, 0.0f);
    }

    __host__ __device__ __forceinline__ static void load_tile_a(FragmentA& a, const T_In* ptr, unsigned lda) {
        nvcuda::wmma::load_matrix_sync(a, ptr, lda);
    }

    __host__ __device__ __forceinline__ static void load_tile_b_col(FragmentB& b, const T_In* ptr, unsigned ldb) {
        nvcuda::wmma::load_matrix_sync(b, ptr, ldb);
    }

    __host__ __device__ __forceinline__ static void load_tile_b_row(FragmentB_Row& b, const T_In* ptr, unsigned ldb) {
        nvcuda::wmma::load_matrix_sync(b, ptr, ldb);
    }

    __host__ __device__ __forceinline__ static void mma(FragmentC& d, const FragmentA& a, const FragmentB& b, const FragmentC& c) {
        nvcuda::wmma::mma_sync(d, a, b, c);
    }

    __host__ __device__ __forceinline__ static void store_accumulator(T_Acc* ptr, const FragmentC& c, unsigned ldc) {
        nvcuda::wmma::store_matrix_sync(ptr, c, ldc, nvcuda::wmma::mem_row_major);
    }
};

} // namespace tile_ir
