#pragma once
#include <cstdint>

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

    __device__ __forceinline__ explicit TensorView(T* ptr) : data(ptr) {}

    __device__ __forceinline__ T& operator()(int r, int c) {
        return data[r * T_Stride::kStrideM + c * T_Stride::kStrideN];
    }

    __device__ __forceinline__ const T& operator()(int r, int c) const {
        return data[r * T_Stride::kStrideM + c * T_Stride::kStrideN];
    }

    __device__ __forceinline__ T* ptr(int r = 0, int c = 0) {
        return data + (r * T_Stride::kStrideM + c * T_Stride::kStrideN);
    }

    __device__ __forceinline__ const T* ptr(int r = 0, int c = 0) const {
        return data + (r * T_Stride::kStrideM + c * T_Stride::kStrideN);
    }
};

// Utility to create row-major tensor views
template <int M, int N, typename T>
__device__ __forceinline__ auto make_row_major_tensor(T* ptr) {
    return TensorView<T, Shape<M, N>, Stride<N, 1>>(ptr);
}

// 2D Block Tile Partitioning helper (eliminates threadIdx modulo/division)
template <int TileM, int TileN, int Threads>
struct TiledPartition {
    static constexpr int kThreadsPerRow = TileN / 4; // 128-bit vectorization (4 elements)
    static constexpr int kRowsPerStep = Threads / (kThreadsPerRow > 0 ? kThreadsPerRow : 1);

    static __device__ __forceinline__ int get_row(int tid) {
        return (kThreadsPerRow > 0) ? (tid / kThreadsPerRow) : tid;
    }

    static __device__ __forceinline__ int get_col(int tid) {
        return (kThreadsPerRow > 0) ? ((tid % kThreadsPerRow) * 4) : 0;
    }
};

} // namespace tile_ir
