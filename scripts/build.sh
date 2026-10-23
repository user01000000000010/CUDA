#!/usr/bin/env bash
# ============================================================================
# Resource-safe build + test of ONE (or a few) kernels.
# Run this on the GPU machine, NOT on the Mac.
#
# Usage:
#   scripts/build.sh fa2            # build FA-2 for sm_80, run its tests
#   scripts/build.sh fa3 90a        # build FA-3 for sm_90a
#   scripts/build.sh "paged,decode" 80,90a
#
# Env overrides: MAX_JOBS (default 2), BUILD_DIR, PTXAS_VERBOSE (default 1)
# ============================================================================
set -euo pipefail

KERNELS="${1:?usage: scripts/build.sh <kernels> [archs]}"
CUDA_ARCHS="${2:-80}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/sass_reports"
mkdir -p "${LOG_DIR}"
TAG="$(echo "${KERNELS}" | tr ',' '_')_sm$(echo "${CUDA_ARCHS}" | tr ',' '_')"
BUILD_LOG="${LOG_DIR}/build_${TAG}.log"

export KERNELS CUDA_ARCHS
export MAX_JOBS="${MAX_JOBS:-2}"
export PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

echo "==> Building kernels='${KERNELS}' archs='${CUDA_ARCHS}' MAX_JOBS=${MAX_JOBS}"
cd "${ROOT}"
# nice: lower CPU priority so the machine stays responsive during compilation
nice -n 10 python setup.py build_ext --inplace 2>&1 | tee "${BUILD_LOG}"

echo
echo "==> ptxas resource usage (registers / spills / smem)"
grep -E "Used [0-9]+ registers|spill stores|spill loads|bytes smem" "${BUILD_LOG}" \
  | sed 's/^ *ptxas info *: *//' | sort -u || echo "(no ptxas info – set PTXAS_VERBOSE=1)"

if grep -qE "[1-9][0-9]* bytes spill (stores|loads)" "${BUILD_LOG}"; then
  echo "!! WARNING: register spills detected (see ${BUILD_LOG})"
fi

echo
echo "==> Running correctness tests"
python -m pytest -q tests/ -rs
