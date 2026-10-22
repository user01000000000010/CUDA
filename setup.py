import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
]

setup(
    name="fused_online_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_online_softmax_cuda",
            sources=[
                "fused_softmax_binding.cpp",
                "fused_softmax_kernel.cu",
                "flash_attention_v2_instantiations.cu",
                "paged_attention_instantiations.cu",
                "fused_rope_flash_attention_instantiations.cu",
                "fused_gqa_instantiations.cu",
                "async_flash_decode_instantiations.cu",
                "flash_attention_bwd_instantiations.cu",
                "fused_mla_instantiations.cu",
                "flash_attention_v3_hopper_instantiations.cu",
                "fused_transformer_megablock_instantiations.cu",
                "fp8_paged_attention_instantiations.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": cuda_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
