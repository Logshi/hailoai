#!/usr/bin/env python3
"""
Benchmark Analysis: CPU (ONNXRuntime) vs Hailo-8 (HailoRT)
Generates charts (A), summary table (B), evaluation (C), and article text (D).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── File paths ──────────────────────────────────────────────────────────────
CPU_CSV  = "benchmark_cpu_20260130_144113.csv"
HAILO_CSV = "benchmark_hailo_20260130_134534.csv"
OUT_DIR  = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & prepare ──────────────────────────────────────────────────────────
def load(path, label):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["label"] = label
    # throttle binary flag
    df["throttled"] = df["throttle_status"].apply(
        lambda s: 0 if str(s).strip().upper() in ("OK", "0X0", "0X00000", "0X00") else 1
    )
    return df

cpu_raw   = load(CPU_CSV, "CPU (YOLOv8n ONNX)")
hailo_raw = load(HAILO_CSV, "Hailo-8 (YOLOv8m HEF)")

# Exclude first row (warm-up)
cpu   = cpu_raw.iloc[1:].reset_index(drop=True)
hailo = hailo_raw.iloc[1:].reset_index(drop=True)

# ── Style ───────────────────────────────────────────────────────────────────
COLOR_CPU   = "#e74c3c"
COLOR_HAILO = "#2ecc71"
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════
# A) CHARTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== A) Generating charts ===\n")

# ── A1: pipeline_fps vs time ────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(cpu["elapsed_s"], cpu["pipeline_fps"], color=COLOR_CPU, alpha=0.7, linewidth=0.8, label="CPU")
ax.plot(hailo["elapsed_s"], hailo["pipeline_fps"], color=COLOR_HAILO, alpha=0.7, linewidth=0.8, label="Hailo-8")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pipeline FPS")
ax.set_title("Pipeline FPS over Time")
ax.legend()
save(fig, "a1_fps_vs_time.png")

# ── A2: cpu_percent vs time ─────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(cpu["elapsed_s"], cpu["cpu_percent"], color=COLOR_CPU, alpha=0.7, linewidth=0.8, label="CPU")
ax.plot(hailo["elapsed_s"], hailo["cpu_percent"], color=COLOR_HAILO, alpha=0.7, linewidth=0.8, label="Hailo-8")
ax.set_xlabel("Time (s)")
ax.set_ylabel("CPU Usage (%)")
ax.set_title("CPU Usage over Time")
ax.legend()
save(fig, "a2_cpu_percent_vs_time.png")

# ── A3: cpu_temp_c vs time ──────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(cpu["elapsed_s"], cpu["cpu_temp_c"], color=COLOR_CPU, alpha=0.7, linewidth=0.8, label="CPU")
ax.plot(hailo["elapsed_s"], hailo["cpu_temp_c"], color=COLOR_HAILO, alpha=0.7, linewidth=0.8, label="Hailo-8")
ax.axhline(y=85, color="orange", linestyle="--", alpha=0.6, label="Thermal throttle (85°C)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("CPU Temperature (°C)")
ax.set_title("CPU Temperature over Time")
ax.legend()
save(fig, "a3_temp_vs_time.png")

# ── A4: e2e_ms vs time ─────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(cpu["elapsed_s"], cpu["e2e_ms"], color=COLOR_CPU, alpha=0.7, linewidth=0.8, label="CPU")
ax.plot(hailo["elapsed_s"], hailo["e2e_ms"], color=COLOR_HAILO, alpha=0.7, linewidth=0.8, label="Hailo-8")
ax.set_xlabel("Time (s)")
ax.set_ylabel("End-to-End Latency (ms)")
ax.set_title("End-to-End Latency over Time")
ax.legend()
save(fig, "a4_e2e_vs_time.png")

# ── A5: inference_ms vs time ────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.plot(cpu["elapsed_s"], cpu["inference_ms"], color=COLOR_CPU, alpha=0.7, linewidth=0.8, label="CPU")
ax.plot(hailo["elapsed_s"], hailo["inference_ms"], color=COLOR_HAILO, alpha=0.7, linewidth=0.8, label="Hailo-8")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Inference Latency (ms)")
ax.set_title("Model Inference Latency over Time")
ax.legend()
save(fig, "a5_inference_vs_time.png")

# ── A6: FPS histogram ──────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.hist(cpu["pipeline_fps"], bins=30, alpha=0.6, color=COLOR_CPU, label="CPU", edgecolor="white")
ax.hist(hailo["pipeline_fps"], bins=30, alpha=0.6, color=COLOR_HAILO, label="Hailo-8", edgecolor="white")
ax.set_xlabel("Pipeline FPS")
ax.set_ylabel("Count")
ax.set_title("Pipeline FPS Distribution")
ax.legend()
save(fig, "a6_fps_histogram.png")

# ── A7: FPS boxplot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(
    [cpu["pipeline_fps"], hailo["pipeline_fps"]],
    tick_labels=["CPU (YOLOv8n)", "Hailo-8 (YOLOv8m)"],
    patch_artist=True,
    widths=0.5,
)
bp["boxes"][0].set_facecolor(COLOR_CPU)
bp["boxes"][1].set_facecolor(COLOR_HAILO)
for box in bp["boxes"]:
    box.set_alpha(0.7)
ax.set_ylabel("Pipeline FPS")
ax.set_title("Pipeline FPS — CPU vs Hailo-8")
save(fig, "a7_fps_boxplot.png")

# ── A8: scatter cpu_percent vs fps ──────────────────────────────────────────
fig, ax = plt.subplots()
ax.scatter(cpu["cpu_percent"], cpu["pipeline_fps"], c=COLOR_CPU, alpha=0.3, s=12, label="CPU")
ax.scatter(hailo["cpu_percent"], hailo["pipeline_fps"], c=COLOR_HAILO, alpha=0.3, s=12, label="Hailo-8")
ax.set_xlabel("CPU Usage (%)")
ax.set_ylabel("Pipeline FPS")
ax.set_title("CPU Usage vs Pipeline FPS")
ax.legend()
save(fig, "a8_cpu_vs_fps.png")

# ── A9: scatter cpu_temp vs fps ─────────────────────────────────────────────
fig, ax = plt.subplots()
ax.scatter(cpu["cpu_temp_c"], cpu["pipeline_fps"], c=COLOR_CPU, alpha=0.3, s=12, label="CPU")
ax.scatter(hailo["cpu_temp_c"], hailo["pipeline_fps"], c=COLOR_HAILO, alpha=0.3, s=12, label="Hailo-8")
ax.set_xlabel("CPU Temperature (°C)")
ax.set_ylabel("Pipeline FPS")
ax.set_title("CPU Temperature vs Pipeline FPS")
ax.legend()
save(fig, "a9_temp_vs_fps.png")

# ═══════════════════════════════════════════════════════════════════════════
# B) SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== B) Summary Table (steady-state, warm-up excluded) ===\n")

def stats(df, label):
    fps = df["pipeline_fps"]
    return {
        "Mode": label,
        "Samples": len(df),
        "FPS mean": f"{fps.mean():.2f}",
        "FPS median": f"{fps.median():.2f}",
        "FPS p01": f"{fps.quantile(0.01):.2f}",
        "FPS min": f"{fps.min():.2f}",
        "FPS max": f"{fps.max():.2f}",
        "e2e_ms mean": f"{df['e2e_ms'].mean():.1f}",
        "inference_ms mean": f"{df['inference_ms'].mean():.1f}",
        "cpu% mean": f"{df['cpu_percent'].mean():.1f}",
        "temp mean": f"{df['cpu_temp_c'].mean():.1f}",
        "temp max": f"{df['cpu_temp_c'].max():.1f}",
        "throttle_ratio": f"{df['throttled'].mean()*100:.1f}%",
    }

summary = pd.DataFrame([stats(cpu, "CPU (YOLOv8n)"), stats(hailo, "Hailo-8 (YOLOv8m)")])
summary = summary.set_index("Mode").T
print(summary.to_string())

# Save as CSV too
summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"))
print(f"\n  saved: {OUT_DIR}/summary_table.csv")

# ═══════════════════════════════════════════════════════════════════════════
# C) EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== C) Evaluation ===\n")

fps_cpu = cpu["pipeline_fps"].mean()
fps_hailo = hailo["pipeline_fps"].mean()
cpu_pct_cpu = cpu["cpu_percent"].mean()
cpu_pct_hailo = hailo["cpu_percent"].mean()
temp_cpu = cpu["cpu_temp_c"].mean()
temp_hailo = hailo["cpu_temp_c"].mean()
e2e_cpu = cpu["e2e_ms"].mean()
e2e_hailo = hailo["e2e_ms"].mean()
inf_cpu = cpu["inference_ms"].mean()
inf_hailo = hailo["inference_ms"].mean()
throttle_cpu = cpu["throttled"].mean() * 100
throttle_hailo = hailo["throttled"].mean() * 100

fps_improvement = fps_hailo / fps_cpu
cpu_decrease = ((cpu_pct_cpu - cpu_pct_hailo) / cpu_pct_cpu) * 100
temp_diff = temp_cpu - temp_hailo
e2e_improvement = e2e_cpu / e2e_hailo

overhead_cpu = e2e_cpu - inf_cpu
overhead_hailo = e2e_hailo - inf_hailo

eval_text = f"""
1. FPS Improvement:
   Hailo-8 delivers {fps_hailo:.1f} FPS vs CPU's {fps_cpu:.1f} FPS.
   That is a {fps_improvement:.1f}x throughput improvement.

2. CPU Load Reduction:
   CPU mode uses {cpu_pct_cpu:.1f}% of all cores vs Hailo's {cpu_pct_hailo:.1f}%.
   CPU load decreased by {cpu_decrease:.0f}%.

3. Temperature & Throttling:
   CPU mode: mean {temp_cpu:.1f}°C, max {cpu['cpu_temp_c'].max():.1f}°C, throttled {throttle_cpu:.0f}% of the time.
   Hailo mode: mean {temp_hailo:.1f}°C, max {hailo['cpu_temp_c'].max():.1f}°C, throttled {throttle_hailo:.0f}% of the time.
   Temperature difference: {temp_diff:.1f}°C lower with Hailo.

4. End-to-End Latency:
   CPU: {e2e_cpu:.1f} ms mean -> Hailo: {e2e_hailo:.1f} ms mean.
   Hailo is {e2e_improvement:.1f}x faster end-to-end.
   Inference only: CPU {inf_cpu:.1f} ms vs Hailo {inf_hailo:.1f} ms ({inf_cpu/inf_hailo:.1f}x).

5. RTSP Decode Overhead (e2e - inference):
   CPU:   {overhead_cpu:.1f} ms overhead (pre/post-processing + decode)
   Hailo: {overhead_hailo:.1f} ms overhead
   The overhead is small relative to inference in CPU mode ({overhead_cpu/e2e_cpu*100:.0f}% of e2e).
   In Hailo mode the overhead is {overhead_hailo/e2e_hailo*100:.0f}% of e2e, suggesting
   frame decode + pre/post-processing is NOT a bottleneck yet, but would become one
   if inference were further accelerated (e.g. batching or a faster model).
"""
print(eval_text)

# ═══════════════════════════════════════════════════════════════════════════
# D) MEDIUM ARTICLE SECTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== D) Test Results — Medium Article Section ===\n")

article = f"""
## Test Results

### Setup
- **Board:** Raspberry Pi 5 (4 GB RAM, Broadcom BCM2712, 4x Cortex-A76)
- **OS:** Debian (aarch64), Python 3.10+
- **Camera:** IP camera via RTSP, 640×640 input resolution
- **CPU mode:** YOLOv8n (nano) exported to ONNX, ONNXRuntime with 4 threads
- **Hailo mode:** YOLOv8m (medium) compiled to HEF, Hailo-8 M.2 AI accelerator via PCIe
- **Duration:** ~{len(cpu)}s (CPU), ~{len(hailo)}s (Hailo) of continuous inference

### Key Findings

- **{fps_improvement:.0f}× throughput gain:** The Hailo-8 achieved **{fps_hailo:.1f} FPS** (YOLOv8m) compared to the CPU's **{fps_cpu:.1f} FPS** (YOLOv8n) — a {fps_improvement:.1f}× improvement while running a *larger* model.
- **{cpu_decrease:.0f}% CPU reduction:** CPU utilization dropped from **{cpu_pct_cpu:.0f}%** (all cores saturated) to just **{cpu_pct_hailo:.0f}%**, freeing compute for other tasks.
- **{temp_diff:.0f}°C cooler operation:** Average SoC temperature fell from **{temp_cpu:.0f}°C** to **{temp_hailo:.0f}°C**. CPU mode triggered thermal throttling in **{throttle_cpu:.0f}%** of samples; Hailo mode experienced **{throttle_hailo:.0f}%** throttling.
- **{e2e_improvement:.0f}× lower latency:** Mean end-to-end frame latency dropped from **{e2e_cpu:.0f} ms** to **{e2e_hailo:.0f} ms**, making real-time alerting feasible.
- **Inference vs. overhead:** In Hailo mode, model inference accounts for ~{inf_hailo/e2e_hailo*100:.0f}% of the frame pipeline ({inf_hailo:.0f} ms out of {e2e_hailo:.0f} ms). The remaining ~{overhead_hailo:.0f} ms is RTSP decode and pre/post-processing — not yet a bottleneck, but the next optimization target as inference latency decreases.

### Conclusion

Offloading YOLOv8 inference to the Hailo-8 AI accelerator transforms the Raspberry Pi 5 from a barely-capable edge device (~{fps_cpu:.0f} FPS, thermally throttled) into a viable real-time detection platform (~{fps_hailo:.0f} FPS, thermally stable). The CPU is freed for application logic, networking, and multi-stream handling. For production deployments requiring sustained inference at the edge, a dedicated AI accelerator is not optional — it is essential.
"""
print(article)

# Save article text
with open(os.path.join(OUT_DIR, "article_test_results.md"), "w") as f:
    f.write(article)
print(f"  saved: {OUT_DIR}/article_test_results.md")

print("\n✓ Analysis complete. All outputs in ./{OUT_DIR}/")
