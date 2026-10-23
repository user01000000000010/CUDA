#!/usr/bin/env bash
set -euo pipefail

# Konfiguration
GPU_ID=${GPU_ID:-0}
LOCK_SM_CLOCK=${LOCK_SM_CLOCK:-1500} # Lås SM-klockan (MHz) för stabila och reproducerbara resultat
REPORT_FILE="CUDA_AUDIT_REPORT.md"

echo "========================================================"
echo "  CUDA 10/10 Verification & Hardware Profiling Suite"
echo "========================================================"

# 1. Kontrollera att NVIDIAs verktyg finns i PATH
for cmd in compute-sanitizer ncu nvidia-smi; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "[FEL] Verktyget '$cmd' hittades inte i PATH. Kontrollera din CUDA Toolkit-installation."
        exit 1
    fi
done

# 2. Lås GPU-klockor om skriptet körs med sudo/root-rättigheter
cleanup() {
    if [ "$EUID" -eq 0 ]; then
        echo "[*] Återställer GPU-klockfrekvenser..."
        nvidia-smi -i "$GPU_ID" -rgc &> /dev/null || true
    fi
}
trap cleanup EXIT

if [ "$EUID" -eq 0 ]; then
    echo "[*] Låser GPU $GPU_ID SM-klocka för deterministiska mätningar..."
    nvidia-smi -i "$GPU_ID" -pm 1 &> /dev/null || true
    nvidia-smi -i "$GPU_ID" -lgc "$LOCK_SM_CLOCK","$LOCK_SM_CLOCK" &> /dev/null || true
else
    echo "[i] Körs utan root: GPU-klockor låses endast via ncu (--clock-control base)."
    echo "    Tips: Vid 'ERR_NVGPUCTRPERM', kör med sudo eller sätt NVreg_RestrictProfilingToAdminUsers=0."
fi

# 3. Sätt miljövariabler för maximal synkronisering under felsökning
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TORCH_SHOW_CPP_STACKTRACES=1

# 4. Exekvera Python-orkestreraren
python3 cuda_audit_engine.py \
    --bench-script "benchmark_pytorch.py" \
    --launch-skip 5 \
    --launch-count 10 \
    --report "$REPORT_FILE" "$@"

echo "========================================================"
echo "  Färdig! Öppna $REPORT_FILE för att granska rapporten."
echo "========================================================"
