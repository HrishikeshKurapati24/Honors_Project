#!/usr/bin/env python3
"""
plot_dataset_claim_analysis_suite.py — Springer Nature submission version
Implements:
  - Helvetica/Arial font (Springer Nature requirement)
  - 174 mm / 6.85-inch full-width figure
  - White background, no seaborn grey
  - Wong (2011) colorblind-safe palette
  - No in-panel annotation boxes
  - Lowercased bold (a)-(e) panel labels
  - TIFF (300 DPI) + PDF output
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
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
})

# ── Wong (2011) colorblind-safe palette ──────────────────────────────────────
C_AUC     = "#0072B2"   # blue
C_AUPR    = "#D55E00"   # vermillion
C_POS     = "#009E73"   # green (positive delta)
C_NEG     = "#D55E00"   # vermillion (negative delta)
C_REF     = "#333333"   # dark grey for reference lines
C_FULL    = "#CC79A7"   # rose

MODALITY_ORDER = [
    "genomics_mutation",
    "epigenomics_chromatin",
    "epigenomics_methylation",
    "transcriptomics_expression",
    "proteomics_reverse_phase",
    "metabolomics_profile",
    "pathway",
]

MODALITY_LABELS = {
    "genomics_mutation": "Mut",
    "epigenomics_chromatin": "Chrom",
    "epigenomics_methylation": "Meth",
    "transcriptomics_expression": "Expr",
    "proteomics_reverse_phase": "Prot",
    "metabolomics_profile": "Metab",
    "pathway": "Path",
}

CONFIG_LABELS = {
    "pathway_exp_prot": "Path+Expr+Prot",
    "exp_prot": "Expr+Prot",
    "prot_metab": "Prot+Metab",
    "expression_only": "Expr only",
    "mut_meth_exp": "Mut+Meth+Expr",
    "meth_exp": "Meth+Expr",
    "mut_exp": "Mut+Expr",
    "full_7omics": "Full 7-omics",
    "mut_chrom_exp": "Mut+Chrom+Expr",
    "pathway": "Path only",
}

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def count_modalities(stems: str) -> int:
    return len(str(stems).split("|"))

def nice_config(name: str) -> str:
    return CONFIG_LABELS.get(name, name.replace("_", "+"))

def nice_modality(name: str) -> str:
    return MODALITY_LABELS.get(name, name)

def _style_ax(ax, xlabel="", ylabel="", title=""):
    """Apply uniform panel styling."""
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold", loc="left", pad=4)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)

def build_analysis(
    dc1: pd.DataFrame,
    dc2: pd.DataFrame,
    dc3_matrix_auc: pd.DataFrame,
    dc3_matrix_aupr: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> str:
    dc1_sorted = dc1.sort_values("auc", ascending=False).reset_index(drop=True)
    top_cfg = dc1_sorted.iloc[0]
    full_cfg = dc1.loc[dc1["config_name"] == "full_7omics"].iloc[0]
    expr_only = dc1.loc[dc1["config_name"] == "expression_only"].iloc[0]

    dc2_nonbaseline = dc2.loc[dc2["removed_stem"].notna()].copy()
    best_remove = dc2_nonbaseline.sort_values("delta_auc_vs_full", ascending=False).iloc[0]
    worst_remove = dc2_nonbaseline.sort_values("delta_auc_vs_full", ascending=True).iloc[0]

    dc3_long = (
        dc3_matrix_auc.stack()
        .rename("delta_auc")
        .reset_index()
        .rename(columns={"level_0": "base_name", "level_1": "added_stem"})
        .merge(
            dc3_matrix_aupr.stack().rename("delta_aupr").reset_index().rename(
                columns={"level_0": "base_name", "level_1": "added_stem"}
            ),
            on=["base_name", "added_stem"],
            how="left",
        )
    )
    best_add = dc3_long.sort_values("delta_auc", ascending=False).iloc[0]
    worst_add = dc3_long.sort_values("delta_auc", ascending=True).iloc[0]

    top_pair = pairwise.iloc[0]
    second_pair = pairwise.iloc[1]
    third_pair = pairwise.iloc[2]

    return f"""### Dataset Redundancy and Noise Analysis

Figure Y summarizes complementary views showing that the current GDSC multi-omics regime contains substantial redundancy and only selective cross-modality complementarity. In the curated configuration benchmark, the best-performing compact subset, {nice_config(top_cfg['config_name'])}, reached {top_cfg['auc']:.4f} AUC and {top_cfg['aupr']:.4f} AUPR, outperforming the full seven-omics configuration ({full_cfg['auc']:.4f} AUC, {full_cfg['aupr']:.4f} AUPR). Even the single-modality expression-only model achieved {expr_only['auc']:.4f} AUC, which already exceeded the broader full-omics setting. This result directly indicates that additional modalities in the present dataset do not consistently provide additive predictive signal.

The leave-one-out analysis reinforces this interpretation by showing that several modalities can be removed from the full seven-omics model without loss, and in some cases with slight improvement. The largest gain came from removing {nice_modality(best_remove['removed_stem'])}, which improved AUC by {best_remove['delta_auc_vs_full']:+.4f} and AUPR by {best_remove['delta_aupr_vs_full']:+.4f} relative to the full model. Removing {nice_modality(worst_remove['removed_stem'])} produced the strongest degradation, with {worst_remove['delta_auc_vs_full']:+.4f} AUC and {worst_remove['delta_aupr_vs_full']:+.4f} AUPR change, suggesting that this modality contributes the clearest unique signal. The overall pattern is therefore asymmetrical: only a small subset of modalities appears indispensable, while others are at least partially redundant.

The add-one-in experiment shows that complementarity is selective rather than generic. The strongest improvement was obtained by adding {nice_modality(best_add['added_stem'])} to the {nice_config(best_add['base_name'])} base, yielding {best_add['delta_auc']:+.4f} AUC and {best_add['delta_aupr']:+.4f} AUPR. In contrast, the weakest addition was {nice_modality(worst_add['added_stem'])} on top of {nice_config(worst_add['base_name'])}, which changed AUC by {worst_add['delta_auc']:+.4f}. This supports the claim that useful multi-omics fusion on the current dataset arises from a few targeted combinations rather than from naive modality accumulation.

Finally, the structural redundancy analysis shows that multiple omics views induce overlapping cell-cell geometry. The strongest pairwise redundancy was observed for {nice_modality(top_pair['modality_a'])}-{nice_modality(top_pair['modality_b'])} (matrix correlation {top_pair['similarity_matrix_correlation']:.3f}, neighbor overlap {top_pair['average_neighbor_overlap']:.3f}), followed by {nice_modality(second_pair['modality_a'])}-{nice_modality(second_pair['modality_b'])} and {nice_modality(third_pair['modality_a'])}-{nice_modality(third_pair['modality_b'])}. These same modalities also appear repeatedly in the strongest predictive compact subsets, which links the static data geometry to the observed predictive redundancy. Taken together, the results support the conclusion that the present GDSC omics benchmark is constrained more by redundancy and weak unique signal across modalities than by the capacity of the encoder to ingest additional data sources.
"""

def main() -> None:
    parser = argparse.ArgumentParser(description="Create single-page dataset-claim figure suite.")
    parser.add_argument("--tag", default="dataset_claim_v1", help="Experiment tag inside dataset_claim_outputs.")
    parser.add_argument(
        "--outputs-root",
        default="flexible model/dataset_claim_outputs",
        help="Root directory containing dc1-dc5 outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    tag = args.tag
    out_dir = Path(args.out_dir) if args.out_dir else outputs_root / "dataset_claim_analysis" / tag
    ensure_dir(out_dir)

    dc1 = pd.read_csv(outputs_root / "dc1_curated_configs" / tag / "leaderboard.csv")
    dc2 = pd.read_csv(outputs_root / "dc2_leave_one_out" / tag / "leave_one_out_deltas.csv")
    dc3 = pd.read_csv(outputs_root / "dc3_add_one_in" / tag / "results.csv")
    corr = pd.read_csv(outputs_root / "dc5_redundancy_analysis" / tag / "similarity_matrix_correlation.csv").rename(
        columns={"Unnamed: 0": "modality"}
    )
    overlap = pd.read_csv(outputs_root / "dc5_redundancy_analysis" / tag / "neighbor_overlap.csv").rename(
        columns={"Unnamed: 0": "modality"}
    )
    pairwise = pd.read_csv(outputs_root / "dc5_redundancy_analysis" / tag / "pairwise_redundancy.csv")

    # Precompute dc3 deltas.
    dc3_base = dc3.loc[dc3["added_stem"].isna(), ["base_name", "auc", "aupr"]].rename(
        columns={"auc": "base_auc", "aupr": "base_aupr"}
    )
    dc3_delta = dc3.merge(dc3_base, on="base_name", how="left")
    dc3_delta["delta_auc"] = dc3_delta["auc"] - dc3_delta["base_auc"]
    dc3_delta["delta_aupr"] = dc3_delta["aupr"] - dc3_delta["base_aupr"]
    dc3_delta = dc3_delta.loc[dc3_delta["added_stem"].notna()].copy()

    base_order = ["pathway", "exp_prot", "mut_meth_exp"]
    add_order = [
        "genomics_mutation",
        "epigenomics_chromatin",
        "epigenomics_methylation",
        "transcriptomics_expression",
        "proteomics_reverse_phase",
        "metabolomics_profile",
        "pathway",
    ]

    dc3_auc_matrix = dc3_delta.pivot(index="base_name", columns="added_stem", values="delta_auc").reindex(
        index=base_order, columns=add_order
    )
    dc3_aupr_matrix = dc3_delta.pivot(index="base_name", columns="added_stem", values="delta_aupr").reindex(
        index=base_order, columns=add_order
    )

    # DC5 matrices.
    corr = corr.set_index("modality").reindex(index=MODALITY_ORDER, columns=MODALITY_ORDER)
    overlap = overlap.set_index("modality").reindex(index=MODALITY_ORDER, columns=MODALITY_ORDER)

    # ── Figure Layout ────────────────────────────────────────────────────────
    # Springer Nature full width = 174 mm = 6.85 inches
    fig = plt.figure(figsize=(6.85, 8.5), facecolor="white")
    
    gs = fig.add_gridspec(
        2, 2,
        hspace=0.18, wspace=0.35,
        left=0.08, right=0.97,
        top=0.94, bottom=0.13,
    )
    
    ax2 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 0])

    # ── Abbreviations Legend ──────────────────────────────────────────────────
    # Place it cleanly in the empty gs[1, 1] slot
    ax1 = fig.add_subplot(gs[1, 1])
    ax1.axis("off")
    # Abbreviations box moved to LaTeX caption to save PDF word count.

    # ── Panel B: Leave-one-out deltas ─────────────────────────────────────────
    dc2_plot = dc2.loc[dc2["removed_stem"].notna()].sort_values("delta_auc_vs_full", ascending=True)
    y2 = np.arange(len(dc2_plot))
    
    ax2.axvline(0.0, color=C_REF, linestyle="--", linewidth=1.0, zorder=2)
    ax2.hlines(y2, 0, dc2_plot["delta_auc_vs_full"], color=np.where(dc2_plot["delta_auc_vs_full"] >= 0, C_POS, C_NEG), linewidth=1.5, alpha=0.85, zorder=3)
    ax2.scatter(dc2_plot["delta_auc_vs_full"], y2, color=np.where(dc2_plot["delta_auc_vs_full"] >= 0, C_POS, C_NEG), s=40, zorder=4, label="ΔAUC")
    ax2.scatter(dc2_plot["delta_aupr_vs_full"], y2, color=C_AUC, marker="s", s=30, zorder=4, label="ΔAUPR")
    
    ax2.set_yticks(y2)
    ax2.set_yticklabels([nice_modality(x) for x in dc2_plot["removed_stem"]], fontsize=7)
    
    # Value annotations on the points removed for brevity

    ax2.legend(loc="lower right", fontsize=6.5, frameon=True, edgecolor="#cccccc", facecolor="white", framealpha=0.9)
    _style_ax(ax2, xlabel="Delta vs. full 7-omics", title="(a) Leave-One-Out Effect")

    # ── Panel C: Add-one-in heatmap ───────────────────────────────────────────
    vmax = np.nanmax(np.abs(dc3_auc_matrix.to_numpy()))
    # Use pcolormesh for vectorized output instead of imshow
    data_dc3 = dc3_auc_matrix.to_numpy()
    rows_dc3, cols_dc3 = data_dc3.shape
    im = ax3.pcolormesh(
        np.arange(cols_dc3 + 1) - 0.5,
        np.arange(rows_dc3 + 1) - 0.5,
        data_dc3,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        shading='flat', edgecolors='none', antialiased=False
    )
    ax3.set_xlim(-0.5, cols_dc3 - 0.5)
    ax3.set_ylim(rows_dc3 - 0.5, -0.5) # Flip y-axis to match imshow behavior
    ax3.set_aspect("auto")
    ax3.set_xticks(np.arange(len(add_order)))
    ax3.set_xticklabels([nice_modality(x) for x in add_order], rotation=35, ha="right", fontsize=7.5)
    ax3.set_yticks(np.arange(len(base_order)))
    ax3.set_yticklabels([nice_config(x) for x in base_order], fontsize=7.5)
    
    for i in range(len(base_order)):
        for j in range(len(add_order)):
            auc_val = dc3_auc_matrix.iloc[i, j]
            aupr_val = dc3_aupr_matrix.iloc[i, j]
            if pd.notna(auc_val):
                ax3.text(j, i, f"{auc_val:+.3f}\n{aupr_val:+.3f}", ha="center", va="center", fontsize=6, color="black")
                
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("ΔAUC", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.solids.set_rasterized(False)
    
    ax3.set_title("(b) Add-One-In Complementarity", fontweight="bold", loc="left", pad=4)
    for spine in ("top", "right", "left", "bottom"):
        ax3.spines[spine].set_visible(False)
    ax3.grid(False)

    # Block for Permutation Sensivity removed

    # ── Panel E: Structural redundancy ────────────────────────────────────────
    corr_vals = corr.to_numpy()
    overlap_vals = overlap.to_numpy()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
    
    # Plot correlation background
    # Use pcolormesh for vectorized output instead of imshow
    n = len(MODALITY_ORDER)
    heat = ax5.pcolormesh(
        np.arange(n + 1) - 0.5,
        np.arange(n + 1) - 0.5,
        corr_vals,
        cmap="Blues", vmin=0.0, vmax=np.nanmax(corr_vals),
        shading='flat', edgecolors='none', antialiased=False
    )
    ax5.set_xlim(-0.5, n - 0.5)
    ax5.set_ylim(n - 0.5, -0.5) # Match imshow upper origin
    ax5.set_aspect("equal")
    
    n = len(MODALITY_ORDER)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax5.text(j, i, "1.00", ha="center", va="center", fontsize=6, color="black")
            else:
                # Use text for correlation value
                color = "white" if corr_vals[i, j] > 0.6 else "black"
                ax5.text(j, i, f"{corr_vals[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax5.set_xticks(np.arange(n))
    ax5.set_yticks(np.arange(n))
    ax5.set_xticklabels([nice_modality(m) for m in MODALITY_ORDER], rotation=35, ha="right", fontsize=7.5)
    ax5.set_yticklabels([nice_modality(m) for m in MODALITY_ORDER], fontsize=7.5)
    
    cbar2 = fig.colorbar(heat, ax=ax5, fraction=0.028, pad=0.02)
    cbar2.set_label("Similarity Matrix Correlation", fontsize=7)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.solids.set_rasterized(False)
    
    ax5.set_title("(c) Structural Redundancy", fontweight="bold", loc="left", pad=4)
    for spine in ("top", "right", "left", "bottom"):
        ax5.spines[spine].set_visible(False)
    ax5.grid(False)


    # ── Output ────────────────────────────────────────────────────────────────
    pdf_path = out_dir / f"dataset_claim_analysis_suite_{tag}.pdf"
    tiff_path = out_dir / f"dataset_claim_analysis_suite_{tag}.tiff"
    png_path = out_dir / f"dataset_claim_analysis_suite_{tag}.png"
    md_path = out_dir / f"dataset_claim_analysis_section_{tag}.md"

    fig.savefig(pdf_path, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff_path, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=800, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    analysis = build_analysis(dc1, dc2, dc3_auc_matrix, dc3_aupr_matrix, pairwise)
    md_path.write_text(analysis)

    print(f"Saved PDF  : {pdf_path}")
    print(f"Saved TIFF : {tiff_path}")
    print(f"Saved PNG  : {png_path}")
    print(f"Saved analysis markdown: {md_path}")


if __name__ == "__main__":
    main()
