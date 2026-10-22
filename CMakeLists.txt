cmake_minimum_required(VERSION 3.22)
project(FusedOnlineSoftmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Ampere (sm_80), Ada (sm_89), or Hopper (sm_90) as appropriate
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)

# Optimization and warning flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall,-Wextra --expt-relaxed-constexpr --use_fast_math")

add_executable(fused_softmax fused_online_softmax.cu)

set_target_properties(fused_softmax PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
