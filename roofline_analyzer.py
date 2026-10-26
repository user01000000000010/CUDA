#!/usr/bin/env python3
"""
GPU Roofline Model Analyzer & Visualization Engine.
Beräknar aritmetisk intensitet (FLOPs / Byte) för Attention- och Softmax-kärnor,
plottar hårdvarans teoretiska gränser (Roofline SOL) och markerar uppmätt prestanda.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# Kända GPU-specifikationer (Peak TFLOPs & Peak Memory Bandwidth i GB/s)
GPU_DATABASE = {
    "H100_SXM": {"fp16_tflops": 989.0, "fp8_tflops": 1978.0, "hbm_bw_gbs": 3350.0},
    "A100_SXM": {"fp16_tflops": 312.0, "fp8_tflops": 0.0,    "hbm_bw_gbs": 2039.0},
    "RTX_4090": {"fp16_tflops": 165.2, "fp8_tflops": 330.0,  "hbm_bw_gbs": 1008.0},
    "RTX_3090": {"fp16_tflops": 71.0,  "fp8_tflops": 0.0,    "hbm_bw_gbs": 936.0},
    "DEFAULT":  {"fp16_tflops": 312.0, "fp8_tflops": 624.0,  "hbm_bw_gbs": 2000.0},
}


@dataclass
class KernelMeasurement:
    name: str
    arithmetic_intensity: float  # FLOPs / Byte
    achieved_tflops: float       # TFLOPs/s
    achieved_bw_gbs: float       # GB/s
    precision: str = "fp16"


def get_gpu_specs(name: str = "DEFAULT") -> Dict[str, float]:
    for key in GPU_DATABASE:
        if key.lower() in name.lower():
            return GPU_DATABASE[key]
    return GPU_DATABASE["DEFAULT"]


def generate_ascii_roofline(
    specs: Dict[str, float],
    measurements: List[KernelMeasurement],
    precision: str = "fp16"
) -> str:
    """Genererar en textbaserad Roofline-översikt för terminalen och loggar."""
    peak_flops = specs.get(f"{precision}_tflops", specs["fp16_tflops"])
    peak_bw = specs["hbm_bw_gbs"]
    knee_point = peak_flops * 1000.0 / peak_bw  # FLOP/Byte där taket nås

    lines = [
        "================================================================================",
        f"  ROOFLINE ANALYSIS ({precision.upper()}) | Knee Point: {knee_point:.2f} FLOPs/Byte",
        f"  Peak Compute: {peak_flops:.1f} TFLOPs/s | Peak Memory BW: {peak_bw:.1f} GB/s",
        "================================================================================",
        f"{'Kernel Name':<35} | {'Intens. (FL/B)':<14} | {'TFLOPs':<10} | {'SOL %':<8} | {'Regim':<15}",
        "-" * 80,
    ]

    for m in measurements:
        theoretical_max = min(peak_flops, (m.arithmetic_intensity * peak_bw) / 1000.0)
        sol_pct = (m.achieved_tflops / theoretical_max * 100.0) if theoretical_max > 0 else 0.0
        regime = "Memory-Bound" if m.arithmetic_intensity < knee_point else "Compute-Bound"
        
        lines.append(
            f"{m.name:<35} | {m.arithmetic_intensity:<14.2f} | {m.achieved_tflops:<10.2f} | "
            f"{sol_pct:<7.1f}% | {regime:<15}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def plot_roofline_html(
    specs: Dict[str, float],
    measurements: List[KernelMeasurement],
    output_html: str,
    precision: str = "fp16"
) -> None:
    """Genererar en interaktiv HTML/Canvas Roofline-graf utan externa beroenden."""
    peak_flops = specs.get(f"{precision}_tflops", specs["fp16_tflops"])
    peak_bw = specs["hbm_bw_gbs"]
    knee = peak_flops * 1000.0 / peak_bw

    data_points = []
    for m in measurements:
        theoretical_max = min(peak_flops, (m.arithmetic_intensity * peak_bw) / 1000.0)
        sol_pct = (m.achieved_tflops / theoretical_max * 100.0) if theoretical_max > 0 else 0.0
        data_points.append({
            "name": m.name,
            "ai": m.arithmetic_intensity,
            "tflops": m.achieved_tflops,
            "sol_pct": round(sol_pct, 1),
            "regime": "Memory-Bound" if m.arithmetic_intensity < knee else "Compute-Bound"
        })

    html_content = f"""<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <title>CUDA Roofline Model Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); max-width: 1000px; margin: auto; }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
        .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
        .stat-box {{ background: #334155; padding: 12px 18px; border-radius: 8px; flex: 1; }}
        .stat-box .title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}
        .stat-box .val {{ font-size: 20px; font-weight: bold; color: #f1f5f9; }}
        canvas {{ background: #0b0f19; border-radius: 8px; width: 100%; height: 500px; display: block; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
        th {{ color: #94a3b8; }}
        .tag-10 {{ color: #4ade80; font-weight: bold; }}
        .tag-warn {{ color: #facc15; font-weight: bold; }}
    </style>
</head>
<body>
<div class="card">
    <h1>🚀 GPU Roofline Model Analysis</h1>
    <div class="stats">
        <div class="stat-box"><div class="title">Peak Compute</div><div class="val">{peak_flops} TFLOPs</div></div>
        <div class="stat-box"><div class="title">Peak Memory Bandwidth</div><div class="val">{peak_bw} GB/s</div></div>
        <div class="stat-box"><div class="title">Knee Point (Ridge)</div><div class="val">{knee:.2f} FLOP/Byte</div></div>
    </div>
    <canvas id="rooflineCanvas" width="950" height="500"></canvas>
    <table>
        <thead>
            <tr><th>Kärna</th><th>Aritmetisk Intensitet</th><th>Uppmätt Prestanda</th><th>Speed-of-Light (SOL)</th><th>Klassning</th></tr>
        </thead>
        <tbody>
"""
    for dp in data_points:
        tag_class = "tag-10" if dp["sol_pct"] >= 80.0 else "tag-warn"
        html_content += f"""
            <tr>
                <td><code>{dp['name']}</code></td>
                <td>{dp['ai']:.2f} FLOPs/Byte</td>
                <td>{dp['tflops']:.2f} TFLOPs/s</td>
                <td class="{tag_class}">{dp['sol_pct']}%</td>
                <td>{dp['regime']}</td>
            </tr>
"""

    html_content += f"""
        </tbody>
    </table>
</div>

<script>
    const canvas = document.getElementById('rooflineCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pad = 60;

    // Log-skalor: AI från 0.1 till 1000, TFLOPs från 1 till 2000
    const minAI = 0.1, maxAI = 1000;
    const minG = 1, maxG = {peak_flops * 1.5};

    function xToPx(x) {{ return pad + (Math.log10(x) - Math.log10(minAI)) / (Math.log10(maxAI) - Math.log10(minAI)) * (W - 2 * pad); }}
    function yToPx(y) {{ return H - pad - (Math.log10(y) - Math.log10(minG)) / (Math.log10(maxG) - Math.log10(minG)) * (H - 2 * pad); }}

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    [0.1, 1, 10, 100, 1000].forEach(ai => {{
        let x = xToPx(ai);
        ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.fillText(ai, x - 10, H - pad + 20);
    }});
    [1, 10, 100, 1000].forEach(g => {{
        let y = yToPx(g);
        ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.fillText(g, 15, y + 4);
    }});

    // Roofline boundary (Taket)
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(xToPx(minAI), yToPx((minAI * {peak_bw}) / 1000.0));
    ctx.lineTo(xToPx({knee}), yToPx({peak_flops}));
    ctx.lineTo(xToPx(maxAI), yToPx({peak_flops}));
    ctx.stroke();

    // Data points
    const points = {json.dumps(data_points)};
    points.forEach(p => {{
        let px = xToPx(p.ai);
        let py = yToPx(Math.max(p.tflops, 1));
        ctx.fillStyle = p.sol_pct >= 80.0 ? '#4ade80' : '#38bdf8';
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#f8fafc';
        ctx.font = '12px sans-serif';
        ctx.fillText(p.name, px + 10, py + 4);
    }});
</script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Interaktiv Roofline-analys genererad: {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Roofline Model Generator för CUDA Attention Kernels")
    parser.add_argument("--gpu", default="DEFAULT", help="GPU modell (H100_SXM, A100_SXM, RTX_4090, RTX_3090)")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp8"], help="Beräkningsprecision")
    parser.add_argument("--output-html", default="roofline_report.html", help="HTML-fil för visuell rapport")
    args = parser.parse_args()

    specs = get_gpu_specs(args.gpu)

    # Exempelberäkningar baserade på standarddimensioner för dina kernels (B=4, H=32, S=4096, D=128)
    # H100 SXM (peak_bw = 3350 GB/s): theoretical_max = (intensity * 3350) / 1000
    sample_measurements = [
        KernelMeasurement(name="flash_attention_v2_fwd", arithmetic_intensity=64.0, achieved_tflops=202.0, achieved_bw_gbs=3156.0),
        KernelMeasurement(name="flash_attention_v3_hopper", arithmetic_intensity=85.3, achieved_tflops=278.0, achieved_bw_gbs=3259.0),
        KernelMeasurement(name="fused_online_softmax", arithmetic_intensity=2.0, achieved_tflops=3.8, achieved_bw_gbs=1900.0),
        KernelMeasurement(name="paged_attention_fp8", arithmetic_intensity=16.0, achieved_tflops=48.5, achieved_bw_gbs=3031.0),
        KernelMeasurement(name="fused_mla_kernel", arithmetic_intensity=42.6, achieved_tflops=135.0, achieved_bw_gbs=3169.0),
    ]

    print(generate_ascii_roofline(specs, sample_measurements, args.precision))
    plot_roofline_html(specs, sample_measurements, args.output_html, args.precision)


if __name__ == "__main__":
    main()
