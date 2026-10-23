"""
Modular, resource-conscious build for the CUDA kernel library.

Environment variables
---------------------
KERNELS       Comma-separated list of kernels to build (default: "softmax").
              Use "all" to build everything (NOT recommended on small machines).
              Available: softmax, fa2, paged, rope, gqa, decode, bwd, mla,
                         fa3, megablock, fp8_paged
CUDA_ARCHS    Comma-separated target archs (default: "80").
              Examples: "80", "80,90a", "90a", "100a".
              An "a" suffix enables arch-specific features (wgmma, TMA,
              setmaxnreg on 90a; tcgen05/TMEM on 100a).
MAX_JOBS      Parallel compile jobs (default: 2). Keep low to avoid OOM.
BUILD_DIR     Directory for intermediate build files. Defaults to a folder
              OUTSIDE Google Drive (~/.cache/cuda_kernels_build) so Drive does
              not try to sync hundreds of MB of object files.
PTXAS_VERBOSE Set to 1 to print register / shared memory / spill usage.
LINEINFO      Set to 1 to embed line info for Nsight Compute (default: 1).

Examples
--------
    KERNELS=fa2 CUDA_ARCHS=80 python setup.py build_ext --inplace
    KERNELS=fa3 CUDA_ARCHS=90a PTXAS_VERBOSE=1 python setup.py build_ext --inplace
"""
import os
import sys

from setuptools import setup

# ----------------------------------------------------------------------------
# Resource guard rails – set BEFORE importing torch.utils.cpp_extension,
# which reads MAX_JOBS when spawning ninja.
# ----------------------------------------------------------------------------
os.environ.setdefault("MAX_JOBS", "2")

from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

# kernel key -> (nvcc source file, binding macro)
KERNEL_REGISTRY = {
    "softmax":   ("fused_softmax_kernel.cu",                       "WITH_SOFTMAX"),
    "fa2":       ("flash_attention_v2_instantiations.cu",          "WITH_FA2"),
    "paged":     ("paged_attention_instantiations.cu",             "WITH_PAGED"),
    "rope":      ("fused_rope_flash_attention_instantiations.cu",  "WITH_ROPE"),
    "gqa":       ("fused_gqa_instantiations.cu",                   "WITH_GQA"),
    "decode":    ("async_flash_decode_instantiations.cu",          "WITH_DECODE"),
    "bwd":       ("flash_attention_bwd_instantiations.cu",         "WITH_BWD"),
    "mla":       ("fused_mla_instantiations.cu",                   "WITH_MLA"),
    "fa3":       ("flash_attention_v3_hopper_instantiations.cu",   "WITH_FA3"),
    "megablock": ("fused_transformer_megablock_instantiations.cu", "WITH_MEGABLOCK"),
    "fp8_paged": ("fp8_paged_attention_instantiations.cu",         "WITH_FP8_PAGED"),
}

# Kernels that require arch-specific (sm_90a+) features once Phase 4 lands.
REQUIRES_SM90A = {"fa3"}


def _parse_kernels():
    raw = os.environ.get("KERNELS", "softmax").strip().lower()
    if raw == "all":
        return list(KERNEL_REGISTRY)
    selected = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in selected if k not in KERNEL_REGISTRY]
    if unknown:
        sys.exit(f"[setup.py] Unknown kernel(s): {unknown}. "
                 f"Available: {sorted(KERNEL_REGISTRY)}")
    return selected


def _gencode_flags(archs):
    """Explicit -gencode flags. Passing any 'arch' flag makes PyTorch skip its
    own auto-detected TORCH_CUDA_ARCH_LIST flags, so we fully control targets."""
    flags = []
    for arch in archs:
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return flags


def _parse_archs():
    raw = os.environ.get("CUDA_ARCHS", "80")
    archs = [a.strip().lower() for a in raw.split(",") if a.strip()]
    for a in archs:
        if not a.rstrip("a").isdigit():
            sys.exit(f"[setup.py] Invalid arch '{a}'. Use e.g. 80, 89, 90a, 100a.")
    return archs


kernels = _parse_kernels()
archs = _parse_archs()

if REQUIRES_SM90A & set(kernels) and not any(a in ("90a", "100a") for a in archs):
    print("[setup.py] WARNING: 'fa3' is intended for sm_90a. "
          "Add CUDA_ARCHS=90a once Phase 4 lands.")

sources = ["fused_softmax_binding.cpp"] + [KERNEL_REGISTRY[k][0] for k in kernels]
macros = [(KERNEL_REGISTRY[k][1], None) for k in kernels]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--threads=1",  # one ptxas per arch at a time -> lower peak RAM
] + _gencode_flags(archs)

if os.environ.get("LINEINFO", "1") == "1":
    nvcc_flags.append("-lineinfo")
if os.environ.get("PTXAS_VERBOSE", "0") == "1":
    nvcc_flags.append("-Xptxas=-v")

build_dir = os.environ.get(
    "BUILD_DIR", os.path.join(os.path.expanduser("~"), ".cache", "cuda_kernels_build")
)

print("=" * 72)
print(f"[setup.py] Kernels   : {', '.join(kernels)}")
print(f"[setup.py] Archs     : {', '.join(archs)}")
print(f"[setup.py] MAX_JOBS  : {os.environ['MAX_JOBS']}")
print(f"[setup.py] Build dir : {build_dir}")
print("=" * 72)

setup(
    name="fused_online_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_online_softmax_cuda",
            sources=sources,
            define_macros=macros,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"build": {"build_base": build_dir}},
)
