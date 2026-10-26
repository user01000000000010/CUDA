cmake_minimum_required(VERSION 3.22)
project(HighPerfCUDA_LLM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_FOUND TRUE)
    message(STATUS "CUDA Compiler detected: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_FOUND FALSE)
    message(STATUS "No CUDA compiler detected. Configuring in Host-Emulation mode for C++ testing.")
endif()

# ----------------------------------------------------------------------------
# GPU Architectures (SM80 Ampere, SM89 Ada, SM90 Hopper, SM100 Blackwell)
# ----------------------------------------------------------------------------
if(CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
    endif()
    message(STATUS "Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    set(CUDA_OPT_FLAGS
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        -lineinfo
        -Xcompiler -Wall,-Wextra,-fPIC
    )

    set(LIB_KERNEL_SOURCES
        fused_online_softmax.cu
        flash_attention_v2_instantiations.cu
        flash_attention_v3_hopper_instantiations.cu
        flash_attention_bwd_instantiations.cu
        paged_attention_instantiations.cu
        fp8_paged_attention_instantiations.cu
        fused_gqa_instantiations.cu
        fused_mla_instantiations.cu
        fused_rope_flash_attention_instantiations.cu
        async_flash_decode_instantiations.cu
        fused_transformer_megablock_instantiations.cu
        fused_microscaled_attention_instantiations.cu
    )

    # Target 1: Static Library (libcuda_llm_static.a)
    add_library(cuda_llm_static STATIC ${LIB_KERNEL_SOURCES})
    target_compile_options(cuda_llm_static PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPT_FLAGS}>)
    target_include_directories(cuda_llm_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    set_target_properties(cuda_llm_static PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "cuda_llm"
    )

    # Target 2: Shared Library (libcuda_llm.so / .dylib)
    add_library(cuda_llm_shared SHARED ${LIB_KERNEL_SOURCES})
    target_compile_options(cuda_llm_shared PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPT_FLAGS}>)
    target_include_directories(cuda_llm_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    set_target_properties(cuda_llm_shared PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "cuda_llm"
    )

    # Target 3: Standalone Executable for Fused Softmax
    add_executable(fused_softmax_bin fused_online_softmax.cu)
    target_compile_options(fused_softmax_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPT_FLAGS}>)
    set_target_properties(fused_softmax_bin PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        OUTPUT_NAME "fused_softmax"
    )

    # Target 4: Native C++/CUDA Benchmark Suite
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_cuda_native.cu")
        add_executable(bench_cuda_native bench_cuda_native.cu)
        target_link_libraries(bench_cuda_native PRIVATE cuda_llm_static)
        target_compile_options(bench_cuda_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPT_FLAGS}>)
    endif()
endif()

# ----------------------------------------------------------------------------
# Target 5: Native Verification Tests (Runs on both CUDA and Host Platforms)
# ----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernels_native.cpp")
    enable_testing()
    add_executable(test_kernels_native tests/test_kernels_native.cpp)
    target_include_directories(test_kernels_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(CUDA_FOUND)
        target_link_libraries(test_kernels_native PRIVATE cuda_llm_static)
    endif()
    add_test(NAME TestKernelsNative COMMAND test_kernels_native)
endif()

# ----------------------------------------------------------------------------
# Install Rules
# ----------------------------------------------------------------------------
include(GNUInstallDirs)
if(CUDA_FOUND)
    install(TARGETS cuda_llm_static cuda_llm_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

install(FILES
    kernels_api.h
    cuda_compat.h
    tile_ir.cuh
    cuda_async_pipeline.cuh
    fused_online_softmax_kernel.cuh
    flash_attention_v2_kernel.cuh
    flash_attention_v3_hopper_kernel.cuh
    flash_attention_bwd_kernel.cuh
    paged_attention_kernel.cuh
    fp8_paged_attention_kernel.cuh
    fused_gqa_kernel.cuh
    fused_mla_kernel.cuh
    fused_rope_flash_attention_kernel.cuh
    async_flash_decode_kernel.cuh
    fused_transformer_megablock.cuh
    fused_microscaled_attention_kernel.cuh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuda_llm
)
