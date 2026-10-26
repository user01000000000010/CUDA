#!/usr/bin/env python3
"""
Automatiserad GPU Verification & Profiling Engine.
Kör Compute Sanitizer (memcheck, racecheck, synccheck) och Nsight Compute (ncu),
samt genererar en prestandarapport i Markdown.
"""

import argparse
import csv
import datetime
import io
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List

# Mätetal (metrics) i Nsight Compute för Ampere / Ada / Hopper
NCU_METRICS = [
    "gpu__time_duration.sum",                                         # Exekveringstid
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",               # Compute SOL (%)
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",# Memory SOL (%)
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", # Tensor Core-aktivitet (%)
    "sm__warps_active.avg.pct_of_peak_sustained_active",              # Achieved Occupancy (%)
    "launch__registers_per_thread",                                   # Register per tråd
    "local_memory_spilling_requests.sum",                             # Register spilling (måste vara 0)
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",       # SMEM Bank Conflicts (Load)
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",       # SMEM Bank Conflicts (Store)
]

# Regex för att enbart profilera dina egna CUDA-kärnor (skippar PyTorch ATen/cuBLAS warmup)
DEFAULT_KERNEL_REGEX = "regex:.*(flash|paged|mla|fused|rope|gqa|megablock|softmax).*"


@dataclass
class SanitizerResult:
    tool: str
    passed: bool
    error_summary: str
    duration_sec: float


@dataclass
class KernelProfile:
    name: str
    duration_us: float = 0.0
    compute_sol_pct: float = 0.0
    memory_sol_pct: float = 0.0
    tensor_core_pct: float = 0.0
    occupancy_pct: float = 0.0
    registers_per_thread: int = 0
    spills: int = 0
    bank_conflicts_ld: int = 0
    bank_conflicts_st: int = 0

    @property
    def max_sol(self) -> float:
        return max(self.compute_sol_pct, self.memory_sol_pct)

    @property
    def total_bank_conflicts(self) -> int:
        return self.bank_conflicts_ld + self.bank_conflicts_st

    def evaluate_grade(self) -> str:
        """Utvärderar kärnan mot 100/100 och 10/10 silicon-effektivitetskriterier."""
        if self.spills > 0:
            return "VARNING (Register Spilling detekterat)"
        if self.max_sol >= 80.0 and self.total_bank_conflicts == 0:
            return "100/100 (10/10 - Optimal Speed-of-Light & Fulländad Silicon-effektivitet)"
        if self.max_sol >= 70.0:
            return "90/100 (9/10 - Hög SOL)"
        if self.max_sol >= 50.0:
            return "80/100 (8/10 - Godkänd)"
        return "Under 80/100 (Flaskhals detekterad)"


def run_compute_sanitizer(target_cmd: List[str], tools: List[str]) -> List[SanitizerResult]:
    results = []
    for tool in tools:
        print(f"[*] Kör compute-sanitizer --tool {tool} ...")
        cmd = [
            "compute-sanitizer",
            "--tool", tool,
            "--leak-check=full" if tool == "memcheck" else "--print-limit=10",
            "--error-exitcode=1",
            *target_cmd,
        ]
        start = datetime.datetime.now()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = (datetime.datetime.now() - start).total_seconds()

        passed = (proc.returncode == 0)
        # Extrahera ERROR SUMMARY-raden från compute-sanitizer
        summary_lines = [
            line.strip() for line in (proc.stdout + "\n" + proc.stderr).splitlines()
            if "ERROR SUMMARY:" in line
        ]
        summary = summary_lines[-1] if summary_lines else ("0 errors" if passed else "Okänt fel")
        results.append(SanitizerResult(tool=tool, passed=passed, error_summary=summary, duration_sec=elapsed))
        status_str = "PASS" if passed else "FAIL"
        print(f"    -> [{status_str}] {summary} ({elapsed:.1f}s)")

    return results


def parse_float(val: str) -> float:
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def run_ncu_profiler(target_cmd: List[str], kernel_regex: str, launch_skip: int, launch_count: int, export_rep: str = "") -> List[KernelProfile]:
    print(f"[*] Kör Nsight Compute (ncu) med {len(NCU_METRICS)} hårdvarumätetal ...")
    cmd = [
        "ncu",
        "--csv",
        "--kernel-name-base", "demangled",
        "--kernel-name", kernel_regex,
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(NCU_METRICS),
        "--clock-control", "base",
    ]
    if export_rep:
        os.makedirs(os.path.dirname(export_rep) or ".", exist_ok=True)
        cmd.extend(["--export", export_rep, "--force-overwrite"])
    cmd.extend(target_cmd)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    raw_output = proc.stdout

    # Filtrera bort "==PROF==" och eventuella PyTorch-utskrifter före CSV-headern
    csv_lines = []
    in_csv = False
    for line in raw_output.splitlines():
        if line.startswith('"ID"'):
            in_csv = True
        if in_csv and not line.startswith("=="):
            csv_lines.append(line)

    if not csv_lines:
        print("[!] Varning: Ingen CSV-data returnerades från ncu. Kontrollera rättigheter eller kernel-regex.")
        if proc.stderr:
            print(f"    ncu stderr: {proc.stderr.strip()[:300]}")
        return []

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    kernels: Dict[str, KernelProfile] = {}

    for row in reader:
        k_name = row.get("Kernel Name", "Unknown_Kernel")
        # Korta ner extremt långa C++ template-namn för läsbarhet
        short_name = k_name.split("(")[0][:75]
        metric_name = row.get("Metric Name", "")
        metric_unit = row.get("Metric Unit", "")
        metric_val = parse_float(row.get("Metric Value", "0"))

        if short_name not in kernels:
            kernels[short_name] = KernelProfile(name=short_name)

        kp = kernels[short_name]

        if metric_name == "gpu__time_duration.sum":
            # Konvertera till mikrosekunder (us)
            if metric_unit == "ns":
                kp.duration_us = metric_val / 1000.0
            elif metric_unit == "ms":
                kp.duration_us = metric_val * 1000.0
            else:
                kp.duration_us = metric_val
        elif metric_name == "sm__throughput.avg.pct_of_peak_sustained_elapsed":
            kp.compute_sol_pct = metric_val
        elif metric_name == "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed":
            kp.memory_sol_pct = metric_val
        elif metric_name == "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active":
            kp.tensor_core_pct = metric_val
        elif metric_name == "sm__warps_active.avg.pct_of_peak_sustained_active":
            kp.occupancy_pct = metric_val
        elif metric_name == "launch__registers_per_thread":
            kp.registers_per_thread = int(metric_val)
        elif metric_name == "local_memory_spilling_requests.sum":
            kp.spills = int(metric_val)
        elif metric_name == "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum":
            kp.bank_conflicts_ld = int(metric_val)
        elif metric_name == "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum":
            kp.bank_conflicts_st = int(metric_val)

    return list(kernels.values())


def generate_markdown_report(
    sanitizer_results: List[SanitizerResult],
    profiles: List[KernelProfile],
    output_path: str,
) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# CUDA Verification & Hardware Performance Report",
        f"**Genererad:** `{now}`\n",
        "---",
        "## 1. Compute Sanitizer (Minnes- och Trådsäkerhet)",
        "| Verktyg | Status | Tid (s) | Sammanfattning |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for sr in sanitizer_results:
        badge = "GODKÄND" if sr.passed else "MISSLYCKAD"
        lines.append(f"| `{sr.tool}` | **{badge}** | {sr.duration_sec:.1f}s | `{sr.error_summary}` |")

    lines.extend([
        "\n---",
        "## 2. Nsight Compute (Hårdvaruprofilering & Speed-of-Light)",
        "| Kernel | Tid (us) | Compute SOL | Memory SOL | Tensor Core | Occupancy | Reg/Tråd | Bank Conflicts | Bedömning |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ])

    if not profiles:
        lines.append("| *Inga kärnor matchade filtret* | - | - | - | - | - | - | - | - |")
    else:
        for kp in profiles:
            lines.append(
                f"| `{kp.name}` | {kp.duration_us:.2f} | {kp.compute_sol_pct:.1f}% | "
                f"{kp.memory_sol_pct:.1f}% | {kp.tensor_core_pct:.1f}% | {kp.occupancy_pct:.1f}% | "
                f"{kp.registers_per_thread} | {kp.total_bank_conflicts} | **{kp.evaluate_grade()}** |"
            )

    lines.extend([
        "\n---",
        "## 3. Diagnostiska Rekommendationer (Vägen till 10/10)",
    ])

    for kp in profiles:
        recs = []
        if kp.spills > 0:
            recs.append(f"- **Register Spilling ({kp.spills} st):** Justera `__launch_bounds__` eller minska lokala arrayer i `{kp.name}`.")
        if kp.total_bank_conflicts > 0:
            recs.append(f"- **SMEM Bank Conflicts ({kp.total_bank_conflicts} st):** Kontrollera swizzle-mönstret (`XOR`-adressering) i delat minne för `{kp.name}`.")
        if kp.max_sol < 75.0:
            if kp.memory_sol_pct > kp.compute_sol_pct:
                recs.append(f"- **Minnesflaskhals ({kp.memory_sol_pct:.1f}% Memory SOL):** Öka antalet asynkrona pipeline-steg (`cp.async` / TMA) för att dölja HBM-latens.")
            else:
                recs.append(f"- **Beräkningsflaskhals ({kp.compute_sol_pct:.1f}% Compute SOL):** Undersök warp-stalls (t.ex. `mbarrier`-väntetider) och överväg Warp Specialization.")
        if not recs:
            recs.append("- Inga uppenbara flaskhalsar; kärnan opererar nära hårdvarans teoretiska gräns.")

        lines.append(f"### `{kp.name}`")
        lines.extend(recs)
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[+] Rapport sparad till: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Kör CUDA Sanitizer & NCU Profiling Suite.")
    parser.add_argument("--test-script", default="tests/run_all_tests.py", help="Skript för sanitizer-tester")
    parser.add_argument("--bench-script", default="benchmark_pytorch.py", help="Skript för NCU-profilering")
    parser.add_argument("--kernel-regex", default=DEFAULT_KERNEL_REGEX, help="Regex för kärnfiltrering i ncu")
    parser.add_argument("--launch-skip", type=int, default=5, help="Antal warmup-launches att hoppa över per kärna")
    parser.add_argument("--launch-count", type=int, default=10, help="Antal kärnstarter att profilera")
    parser.add_argument("--report", default="CUDA_AUDIT_REPORT.md", help="Sökväg för genererad Markdown-rapport")
    parser.add_argument("--export-ncu", default="ncu_reports/cuda_audit_profile", help="Exportera .ncu-rep rapport för Nsight Compute GUI")
    parser.add_argument("--skip-sanitizer", action="store_true", help="Hoppa över compute-sanitizer")
    args = parser.parse_args()

    sanitizer_results = []
    if not args.skip_sanitizer:
        target_test = [sys.executable, args.test_script] if os.path.exists(args.test_script) else [sys.executable, args.bench_script]
        sanitizer_results = run_compute_sanitizer(
            target_cmd=target_test,
            tools=["memcheck", "racecheck", "synccheck"],
        )

    profiles = run_ncu_profiler(
        target_cmd=[sys.executable, args.bench_script],
        kernel_regex=args.kernel_regex,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        export_rep=args.export_ncu,
    )

    generate_markdown_report(sanitizer_results, profiles, args.report)

    # Returnera felkod 1 om någon sanitizer-körning misslyckades (för CI/CD-integration)
    if any(not sr.passed for sr in sanitizer_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
