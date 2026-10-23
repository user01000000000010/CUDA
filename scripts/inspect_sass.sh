#!/usr/bin/env bash
# ============================================================================
# Prove which hardware instructions the compiled kernels actually use.
#
# Usage:  scripts/inspect_sass.sh [path/to/extension.so]
#
# Answers the falsification questions from the audit, e.g.
#   "HMMA.16816 vs HGMMA (WARPGROUP.MMA)?"  "LDSM?"  "TMA (UTMALDG)?"
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SO="${1:-$(ls "${ROOT}"/fused_online_softmax_cuda*.so 2>/dev/null | head -n1 || true)}"
[[ -f "${SO}" ]] || { echo "No extension .so found. Build first (scripts/build.sh)."; exit 1; }
command -v cuobjdump >/dev/null || { echo "cuobjdump not found (needs CUDA toolkit)."; exit 1; }

OUT_DIR="${ROOT}/sass_reports"
mkdir -p "${OUT_DIR}"
SASS="${OUT_DIR}/$(basename "${SO}" .so).sass"
cuobjdump -sass "${SO}" > "${SASS}"
echo "Full SASS: ${SASS}"
echo

count() { grep -cE "$1" "${SASS}" || true; }

printf "%-34s %8s   %s\n" "Instruction class" "Count" "Meaning"
printf "%-34s %8s   %s\n" "----------------------------------" "--------" "-------"
printf "%-34s %8s   %s\n" "FFMA (scalar FP32 FMA)"        "$(count '\bFFMA\b')"        "CUDA-core math (no tensor cores)"
printf "%-34s %8s   %s\n" "HMMA (mma.sync, sm_80+)"       "$(count '\bHMMA\.')"        "Ampere-style tensor cores"
printf "%-34s %8s   %s\n" "HGMMA (wgmma, sm_90a)"         "$(count '\bHGMMA\.')"       "Hopper warpgroup MMA"
printf "%-34s %8s   %s\n" "UTC*MMA (tcgen05, sm_100a)"    "$(count '\bUTC[A-Z]*MMA')"  "Blackwell tensor cores"
printf "%-34s %8s   %s\n" "LDSM (ldmatrix)"               "$(count '\bLDSM\b|\bLDSM\.')" "SMEM->fragment loads"
printf "%-34s %8s   %s\n" "LDGSTS (cp.async)"             "$(count '\bLDGSTS')"        "Async global->SMEM copy"
printf "%-34s %8s   %s\n" "UTMALDG (TMA load)"            "$(count '\bUTMALDG')"       "Tensor Memory Accelerator"
printf "%-34s %8s   %s\n" "SYNCS (mbarrier)"              "$(count '\bSYNCS\.')"       "Async transaction barriers"
printf "%-34s %8s   %s\n" "USETMAXREG (setmaxnreg)"       "$(count '\bUSETMAXREG')"    "Register reallocation"
printf "%-34s %8s   %s\n" "STL/LDL (local memory)"        "$(count '\b(STL|LDL)(\.|\b)')" "Register spills / local arrays"

echo
echo "Per-kernel summary (function -> FFMA / HMMA / HGMMA / LDL+STL):"
awk '
  /Function :/ { if (fn) printf "  %-70s %6d %6d %6d %6d\n", fn, f, h, g, l; fn=$3; f=h=g=l=0; next }
  /[ \t]FFMA[ .]/      { f++ }
  /HMMA\./             { h++ }
  /HGMMA\./            { g++ }
  /[ \t](STL|LDL)[ .]/ { l++ }
  END { if (fn) printf "  %-70s %6d %6d %6d %6d\n", fn, f, h, g, l }
' "${SASS}" | c++filt | cut -c1-110
