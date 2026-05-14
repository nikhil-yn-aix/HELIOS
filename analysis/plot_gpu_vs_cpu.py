import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.linewidth": 1.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    }
)

df = pd.read_csv("results/gpu_vs_cpu/data/helios_final_detailed_runs.csv")

df["MP_Success"] = df["MP_Success"].astype(str).map({"True": 1, "False": 0})
df["AS_Success"] = df["AS_Success"].astype(str).map({"True": 1, "False": 0})

df["MP_Fitness_Safe"] = np.maximum(df["MP_Fitness"], 1e-60)
df["AS_Fitness_Safe"] = np.maximum(df["AS_Fitness"], 1e-60)
df["MP_Log"] = np.log10(df["MP_Fitness_Safe"])
df["AS_Log"] = np.log10(df["AS_Fitness_Safe"])
df["Quality_Advantage"] = df["MP_Log"] - df["AS_Log"]
df["MP_Efficiency"] = df["MP_Evals"] / df["MP_Time"]
df["AS_Efficiency"] = df["AS_Evals"] / df["AS_Time"]

summary = (
    df.groupby("Function")
    .agg(
        {
            "Speedup": "mean",
            "MP_Success": "mean",
            "AS_Success": "mean",
            "Quality_Advantage": "mean",
            "MP_Efficiency": "mean",
            "AS_Efficiency": "mean",
        }
    )
    .reset_index()
)

color_map = {
    "speedup": "#FF1493",
    "mp": "#00CED1",
    "as": "#FF4500",
    "quality_good": "#32CD32",
    "quality_bad": "#FFD700",
    "efficiency": "#8A2BE2",
}

fig = plt.figure(figsize=(17, 14))
gs = fig.add_gridspec(
    2, 2, hspace=0.45, wspace=0.3, top=0.88, bottom=0.1, left=0.08, right=0.95
)

ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh(
    summary["Function"],
    summary["Speedup"],
    color=color_map["speedup"],
    alpha=0.85,
    edgecolor="black",
    linewidth=1.5,
)
ax1.set_xlabel("Speedup Factor (×)", fontweight="bold", fontsize=13)
ax1.set_title("Average Speedup by Function", fontweight="bold", pad=20, fontsize=15)
ax1.grid(True, axis="x", alpha=0.4)
ax1.set_xlim(0, summary["Speedup"].max() * 1.15)
for i, (bar, val) in enumerate(zip(bars1, summary["Speedup"])):
    ax1.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}×",
        va="center",
        fontweight="bold",
        fontsize=11,
        color="black",
    )

ax2 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(summary["Function"]))
width = 0.35
mp_bars = ax2.bar(
    x_pos - width / 2,
    summary["MP_Success"] * 100,
    width,
    label="Helios-MP",
    color=color_map["mp"],
    alpha=0.85,
    edgecolor="black",
    linewidth=1.5,
)
as_bars = ax2.bar(
    x_pos + width / 2,
    summary["AS_Success"] * 100,
    width,
    label="Helios-AS",
    color=color_map["as"],
    alpha=0.85,
    edgecolor="black",
    linewidth=1.5,
)
ax2.set_ylabel("Success Rate (%)", fontweight="bold", fontsize=13)
ax2.set_xlabel("Benchmark Functions", fontweight="bold", fontsize=13)
ax2.set_title("Success Rate Comparison", fontweight="bold", pad=20, fontsize=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(summary["Function"], rotation=45, ha="right", fontsize=11)
ax2.legend(loc="upper right", framealpha=0.95, fontsize=12)
ax2.set_ylim(0, 105)
ax2.grid(True, axis="y", alpha=0.4)

ax3 = fig.add_subplot(gs[1, 0])
quality_colors = [
    color_map["quality_good"] if x < 0 else color_map["quality_bad"]
    for x in summary["Quality_Advantage"]
]
bars3 = ax3.bar(
    summary["Function"],
    summary["Quality_Advantage"],
    color=quality_colors,
    alpha=0.85,
    edgecolor="black",
    linewidth=1.5,
)
ax3.axhline(y=0, color="red", linestyle="--", alpha=0.8, linewidth=2.5)
ax3.set_ylabel(
    "Quality Advantage\n(Log10 MP - Log10 AS)", fontweight="bold", fontsize=13
)
ax3.set_xlabel("Benchmark Functions", fontweight="bold", fontsize=13)
ax3.set_title("Solution Quality Comparison", fontweight="bold", pad=20, fontsize=15)
ax3.tick_params(axis="x", rotation=45, labelsize=11)
ax3.grid(True, axis="y", alpha=0.4)
ax3.text(
    0.02,
    0.95,
    "Equal Quality Reference",
    transform=ax3.transAxes,
    verticalalignment="top",
    fontsize=10,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

ax4 = fig.add_subplot(gs[1, 1])
efficiency_ratio = summary["MP_Efficiency"] / summary["AS_Efficiency"]
bars4 = ax4.bar(
    summary["Function"],
    efficiency_ratio,
    color=color_map["efficiency"],
    alpha=0.85,
    edgecolor="black",
    linewidth=1.5,
)
ax4.axhline(y=1, color="red", linestyle="--", alpha=0.8, linewidth=2.5)
ax4.set_ylabel("Efficiency Ratio (MP/AS)", fontweight="bold", fontsize=13)
ax4.set_xlabel("Benchmark Functions", fontweight="bold", fontsize=13)
ax4.set_title("Evaluation Efficiency Ratio", fontweight="bold", pad=20, fontsize=15)
ax4.set_xticklabels(summary["Function"], rotation=45, ha="right", fontsize=11)
ax4.grid(True, axis="y", alpha=0.4)
ax4.set_yscale("log")
ax4.text(
    0.02,
    0.05,
    "Equal Efficiency Reference",
    transform=ax4.transAxes,
    verticalalignment="bottom",
    fontsize=10,
    style="italic",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

fig.suptitle(
    "Helios Algorithm Performance Analysis: Multiprocessing vs Asynchronous",
    fontsize=20,
    fontweight="bold",
    y=0.94,
    color="#2C3E50",
)

plt.savefig("helios_performance_analysis_vibrant.png", dpi=300, bbox_inches="tight")
plt.show()
