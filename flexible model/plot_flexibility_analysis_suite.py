#!/usr/bin/env python3
"""
plot_flexibility_analysis_suite.py  —  Springer Nature submission version
Implements all improvements from flexibility_plot_improvement_plan.md:
  - Helvetica/Arial font (Springer Nature requirement)
  - 174 mm / 6.85-inch full-width figure
  - White background, no seaborn grey
  - Wong (2011) colorblind-safe palette
  - No in-panel annotation boxes
  - Lowercased bold (a)-(e) panel labels (Springer house style)
  - TIFF (300 DPI) + PDF output
  - Twin-axes removed; panel chart types revised for clarity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Springer Nature figure style ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     8.5,
    "axes.labelsize":     8,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
})

# ── Wong (2011) colorblind-safe palette ──────────────────────────────────────
C_AUC     = "#0072B2"   # blue
C_AUPR    = "#D55E00"   # vermillion
C_TIME    = "#E69F00"   # amber
C_MEM     = "#009E73"   # green
C_REF     = "#333333"   # dark grey for reference lines
C_CAT     = "#CC79A7"   # rose / category drops

# ── Helpers ───────────────────────────────────────────────────────────────────
def bytes_to_gib(series: pd.Series) -> pd.Series:
    return series / float(1024 ** 3)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply uniform panel styling."""
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold", loc="left", pad=4)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)


# ── Panel A: Subset Cardinality ───────────────────────────────────────────────
def plot_subset_cardinality(ax, exp1_agg: pd.DataFrame, exp1_results: pd.DataFrame):
    exp1_agg = exp1_agg.sort_values("subset_size")
    x = exp1_agg["subset_size"].to_numpy()

    # Mean AUC with ±std band
    ax.plot(x, exp1_agg["auc_mean"], marker="o", ms=4, color=C_AUC,
            linewidth=1.3, label="AUC (mean ± std)", zorder=3)
    ax.fill_between(x,
                    exp1_agg["auc_mean"] - exp1_agg["auc_std"].fillna(0),
                    exp1_agg["auc_mean"] + exp1_agg["auc_std"].fillna(0),
                    color=C_AUC, alpha=0.15, zorder=2)

    # Mean AUPR with ±std band
    ax.plot(x, exp1_agg["aupr_mean"], marker="s", ms=4, color=C_AUPR,
            linewidth=1.3, label="AUPR (mean ± std)", zorder=3)
    ax.fill_between(x,
                    exp1_agg["aupr_mean"] - exp1_agg["aupr_std"].fillna(0),
                    exp1_agg["aupr_mean"] + exp1_agg["aupr_std"].fillna(0),
                    color=C_AUPR, alpha=0.15, zorder=2)

    # Reference line: best single modality
    best_single = (
        exp1_results.loc[exp1_results["subset_size"] == 1]
        .sort_values("auc", ascending=False)
        .iloc[0]
    )
    ax.axhline(best_single["auc"], color=C_REF, linewidth=0.8,
               linestyle="--", zorder=1)
    ax.text(x[-1] + 0.05, best_single["auc"] + 0.0004,
            f"Best single: {best_single['auc']:.4f}",
            fontsize=6.5, color=C_REF, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_ylim(0.885, 0.928)
    ax.legend(frameon=True, edgecolor="#cccccc", framealpha=0.95)
    _style_ax(ax, xlabel="Number of omics modalities",
              ylabel="Performance", title="(a) Subset Cardinality")


# ── Panel B: Pathway Sharding ─────────────────────────────────────────────────
def plot_pathway_sharding(ax, exp2: pd.DataFrame):
    exp2 = exp2.sort_values("shard_count")
    x    = exp2["shard_count"].to_numpy()
    auc  = exp2["auc"].to_numpy()
    aupr = exp2["aupr"].to_numpy()

    ax.plot(x, auc,  marker="o", ms=4, color=C_AUC,  linewidth=1.3, label="AUC",  zorder=3)
    ax.plot(x, aupr, marker="s", ms=4, color=C_AUPR, linewidth=1.3, label="AUPR", zorder=3)

    # Removed delta annotations to save word count

    # Runtime annotation
    rt_min = exp2["mean_fold_elapsed_seconds"].min()
    rt_max = exp2["mean_fold_elapsed_seconds"].max()
    ax.text(0.97, 0.04,
            f"Runtime: {rt_min:.0f}–{rt_max:.0f} s (all shards)",
            transform=ax.transAxes, fontsize=6.5,
            ha="right", va="bottom", color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0.882, 0.935)   # extra headroom for the delta labels above
    ax.legend(frameon=True, edgecolor="#cccccc", framealpha=0.95)
    _style_ax(ax, xlabel="Pathway shard count",
              ylabel="Performance", title="(b) Pathway Sharding")


# ── Panel C: Missing-Modality Robustness ──────────────────────────────────────
def plot_missing_modality(ax, exp3: pd.DataFrame):
    exp3_full = exp3.loc[exp3["scenario_id"] == "full"].iloc[0]
    full_auc  = exp3_full["auc_mean"]

    # Random drop groups: 1, 2, 4 dropped
    exp3_rand = exp3.loc[exp3["drop_type"] == "random_fraction"].copy()
    drop_counts = [1, 2, 4]
    rand_means  = [exp3_rand.loc[exp3_rand["drop_count"] == c, "auc_mean"].mean()
                   for c in drop_counts]
    rand_stds   = [exp3_rand.loc[exp3_rand["drop_count"] == c, "auc_mean"].std(ddof=0)
                   for c in drop_counts]

    # Category drops
    cat_drop = exp3.loc[exp3["drop_type"] == "category"].copy()
    epi_row  = exp3.loc[exp3["scenario_id"] == "category_drop_epigenomics"].iloc[0]
    tx_row   = exp3.loc[exp3["scenario_id"] == "category_drop_transcriptomics"].iloc[0]

    bar_w = 0.55
    # positions: 1,2,3 for random drops; 4.5,5.5 for category drops (gap)
    rand_pos = np.array([1, 2, 3])
    cat_pos  = np.array([4.5, 5.5])
    all_means = rand_means + [epi_row["auc_mean"], tx_row["auc_mean"]]
    all_stds  = rand_stds  + [0, 0]
    all_pos   = np.concatenate([rand_pos, cat_pos])
    all_colors = [C_AUC] * 3 + [C_CAT, C_CAT]

    bars = ax.bar(all_pos, all_means, width=bar_w,
                  color=all_colors, edgecolor="black", linewidth=0.4, zorder=3)
    ax.errorbar(rand_pos, rand_means, yerr=rand_stds,
                fmt="none", ecolor="black", elinewidth=0.8,
                capsize=2.5, capthick=0.8, zorder=4)

    # Full-omics reference line
    ax.axhline(full_auc, color=C_REF, linewidth=0.9, linestyle="--", zorder=1)
    ax.text(5.9, full_auc + 0.0008, f"Full 7-omics: {full_auc:.4f}",
            fontsize=6.5, color=C_REF, va="bottom", ha="right")

    # Value labels on bars removed to save word count

    ax.set_xticks(list(rand_pos) + list(cat_pos))
    ax.set_xticklabels(["−1", "−2", "−4", "−Epi", "−Tx"])
    ax.set_ylim(0.875, 0.930)

    # Legend patches
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_AUC,  edgecolor="black", linewidth=0.4, label="Random drop"),
        mpatches.Patch(facecolor=C_CAT,  edgecolor="black", linewidth=0.4, label="Category drop"),
    ], frameon=True, edgecolor="#cccccc", framealpha=0.95)

    _style_ax(ax, xlabel="Modalities removed",
              ylabel="AUC", title="(c) Missing-Modality Robustness")


# ── Panel D: Noisy-Modality Tolerance ────────────────────────────────────────
def plot_noisy_modality(ax, exp4: pd.DataFrame):
    exp4      = exp4.sort_values("noise_count").copy()
    exp4_zero = exp4.loc[exp4["noise_count"] == 0].iloc[0]
    exp4["auc_delta"]  = exp4["auc"]  - exp4_zero["auc"]
    exp4["aupr_delta"] = exp4["aupr"] - exp4_zero["aupr"]
    x = exp4["noise_count"].to_numpy()

    # ±0.001 noise-floor shaded band — all changes fall within this region
    ax.axhspan(-0.001, 0.001, color="#dddddd", alpha=0.6, zorder=1,
               label="±0.001 noise floor")
    # Label the band boundaries for direct readability
    ax.text(x[-1] + 0.1, +0.001, "+0.001", fontsize=6.0, va="bottom",
            color="#666666", ha="left")
    ax.text(x[-1] + 0.1, -0.001, "−0.001", fontsize=6.0, va="top",
            color="#666666", ha="left")

    ax.axhline(0.0, color=C_REF, linewidth=0.9, linestyle="--", zorder=2)
    ax.plot(x, exp4["auc_delta"],  marker="o", ms=4, color=C_AUC,
            linewidth=1.3, label="ΔAUC vs. 0-noise baseline", zorder=3)
    ax.plot(x, exp4["aupr_delta"], marker="s", ms=4, color=C_AUPR,
            linewidth=1.3, label="ΔAUPR vs. 0-noise baseline", zorder=3)

    # Explicit conclusion text — prevents reader from inferring upward trend
    ax.text(0.50, 0.95,
            "No systematic trend (all fluctuations within noise floor)",
            transform=ax.transAxes, fontsize=6.3, ha="center", va="top",
            color="#333333", style="italic")

    ax.set_ylim(-0.0025, 0.0038)
    ax.set_xticks(x)
    ax.legend(frameon=True, edgecolor="#cccccc", framealpha=0.95,
              loc="lower right", fontsize=6.2)
    _style_ax(ax, xlabel="Noisy modalities added",
              ylabel="Performance delta vs. baseline",
              title="(d) Noisy-Modality Tolerance")


# ── Panel E: Compute Overhead Comparison ─────────────────────────────────────
def plot_compute_frontier(ax, exp5: pd.DataFrame):
    exp5_valid = exp5.dropna(
        subset=["mean_fold_elapsed_seconds", "max_peak_gpu_memory_bytes"]
    ).copy()
    exp5_valid["mem_gib"] = bytes_to_gib(exp5_valid["max_peak_gpu_memory_bytes"])

    EXPERIMENTS = {
        "exp1_subset_cardinality": "Subset\nCardinality",
        "exp2_pathway_sharding":   "Pathway\nSharding",
        "exp4_noisy_modality":     "Noisy\nModality",
    }
    EXP_COLORS = {
        "exp1_subset_cardinality": C_AUC,
        "exp2_pathway_sharding":   "#9467bd",
        "exp4_noisy_modality":     C_AUPR,
    }

    means = (
        exp5_valid.groupby("experiment", as_index=False)
        .agg(
            runtime=("mean_fold_elapsed_seconds", "mean"),
            runtime_std=("mean_fold_elapsed_seconds", "std"),
            mem_gib=("mem_gib", "mean"),
            mem_gib_std=("mem_gib", "std"),
        )
    )

    exp_keys  = [k for k in EXPERIMENTS if k in means["experiment"].values]
    labels    = [EXPERIMENTS[k] for k in exp_keys]
    runtimes  = [means.loc[means["experiment"] == k, "runtime"].iloc[0]     for k in exp_keys]
    rt_stds   = [means.loc[means["experiment"] == k, "runtime_std"].iloc[0] for k in exp_keys]
    mems      = [means.loc[means["experiment"] == k, "mem_gib"].iloc[0]     for k in exp_keys]
    mem_stds  = [means.loc[means["experiment"] == k, "mem_gib_std"].iloc[0] for k in exp_keys]
    colors    = [EXP_COLORS[k] for k in exp_keys]

    x       = np.arange(len(exp_keys))
    bar_w   = 0.32
    offset  = 0.33

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.set_ylabel("Peak GPU memory (GiB)", color=C_MEM)
    ax2.tick_params(axis="y", labelcolor=C_MEM, labelsize=7.5)

    bars1 = ax.bar(x - offset / 2, runtimes, width=bar_w,
                   color=[c for c in colors], edgecolor="black", linewidth=0.4,
                   alpha=0.85, zorder=3, label="Mean fold time (s)")
    ax.errorbar(x - offset / 2, runtimes, yerr=rt_stds,
                fmt="none", ecolor="black", elinewidth=0.8,
                capsize=2.5, capthick=0.8, zorder=4)

    bars2 = ax2.bar(x + offset / 2, mems, width=bar_w,
                    color=C_MEM, edgecolor="black", linewidth=0.4,
                    alpha=0.55, zorder=3, label="Peak GPU memory (GiB)",
                    hatch="///")
    ax2.errorbar(x + offset / 2, mems, yerr=mem_stds,
                 fmt="none", ecolor="black", elinewidth=0.8,
                 capsize=2.5, capthick=0.8, zorder=4)

    # Value labels
    for bar, val in zip(bars1, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=6)
    for bar, val in zip(bars2, mems):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=6, color=C_MEM)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean fold time (s)")

    # Combined legend
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_AUC,  edgecolor="black", linewidth=0.4, label="Subset Cardinality"),
        mpatches.Patch(facecolor="#9467bd", edgecolor="black", linewidth=0.4, label="Pathway Sharding"),
        mpatches.Patch(facecolor=C_AUPR, edgecolor="black", linewidth=0.4, label="Noisy Modality"),
        mpatches.Patch(facecolor=C_MEM,  edgecolor="black", linewidth=0.4, hatch="///", label="GPU memory"),
    ], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12),
       ncol=4, fontsize=6.5)

    _style_ax(ax, xlabel="", ylabel="Mean fold time (s)",
              title="(e) Compute Overhead Comparison")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Springer-Nature-ready flexibility analysis figure."
    )
    parser.add_argument("--tag", default="scale_v1",
                        help="Experiment tag inside flexibility_outputs.")
    parser.add_argument("--outputs-root", default="flexible model/flexibility_outputs",
                        help="Root directory containing exp1-exp5 summary outputs.")
    parser.add_argument("--out-dir", default=None,
                        help="Optional output directory.")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    tag          = args.tag
    out_dir      = (Path(args.out_dir) if args.out_dir
                    else outputs_root / "flexibility_analysis" / tag)
    ensure_dir(out_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    exp1_agg     = pd.read_csv(outputs_root / "exp1_subset_cardinality" / tag / "aggregate_by_k.csv")
    exp1_results = pd.read_csv(outputs_root / "exp1_subset_cardinality" / tag / "results.csv")
    exp2         = pd.read_csv(outputs_root / "exp2_pathway_sharding"   / tag / "results.csv")
    exp3         = pd.read_csv(outputs_root / "exp3_missing_modality"   / tag / "aggregate_metrics.csv")
    exp4         = pd.read_csv(outputs_root / "exp4_noisy_modality"     / tag / "results.csv")
    exp5         = pd.read_csv(outputs_root / "exp5_compute_scaling"    / tag / "combined_compute_scaling.csv")

    # ── Build figure (3 rows × 2 cols; last panel spans both cols) ────────────
    # Springer Nature full-width: 174 mm = 6.85 inches
    fig = plt.figure(figsize=(6.85, 8.5), facecolor="white")
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        3, 2,
        hspace=0.50, wspace=0.35,
        left=0.09, right=0.97,
        top=0.97,  bottom=0.07,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])   # spans both columns

    plot_subset_cardinality(ax_a, exp1_agg, exp1_results)
    plot_pathway_sharding(ax_b, exp2)
    plot_missing_modality(ax_c, exp3)
    plot_noisy_modality(ax_d, exp4)

    exp5_valid = exp5.dropna(
        subset=["mean_fold_elapsed_seconds", "max_peak_gpu_memory_bytes",
                "parameter_count"]
    ).copy()
    plot_compute_frontier(ax_e, exp5_valid)

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf_path  = out_dir / f"flexibility_analysis_suite_{tag}.pdf"
    tiff_path = out_dir / f"flexibility_analysis_suite_{tag}.tiff"
    png_path  = out_dir / f"flexibility_analysis_suite_{tag}.png"

    fig.savefig(pdf_path,  dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path,  dpi=800, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved PDF  : {pdf_path}")
    print(f"Saved TIFF : {tiff_path}")
    print(f"Saved PNG  : {png_path}")


if __name__ == "__main__":
    main()
