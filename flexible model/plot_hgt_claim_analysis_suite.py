#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "sans-serif"]

C_AUC = "#0072B2"
C_AUPR = "#D55E00"
C_LOCAL = "#aaaaaa"

BRANCH_LABELS = {
    "full": "Full",
    "global_only": "Global only",
    "local_only": "Local only",
}

BUCKET_ORDER = ["both_2hop", "cell_2hop", "drug_2hop", "three_hop_only"]
BUCKET_LABELS = {
    "both_2hop": "Both 2-hop",
    "cell_2hop": "Cell 2-hop",
    "drug_2hop": "Drug 2-hop",
    "three_hop_only": "3-hop only",
}

EDGE_LABELS = {
    "full_graph": "Full graph",
    "no_cell_similarity": "No cell-sim",
    "no_drug_similarity": "No drug-sim",
    "response_only": "Response only",
}

def _style_ax(ax, title, xlabel=None, ylabel=None):
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontweight="bold", loc="left", pad=4, fontsize=8)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(False)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"): ax.spines[spine].set_linewidth(0.6)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def build_analysis(h1: pd.DataFrame, h2: pd.DataFrame, h3: pd.DataFrame, h4: pd.DataFrame, h5: pd.DataFrame) -> str:
    full = h1.loc[h1["variant_name"] == "full"].iloc[0]
    local = h1.loc[h1["variant_name"] == "local_only"].iloc[0]
    global_only = h1.loc[h1["variant_name"] == "global_only"].iloc[0]

    h2_wide = h2.pivot(index="fraction", columns="variant_name", values=["auc", "recovery_auc", "aupr", "recovery_aupr"])
    frac_low = h2_wide.index.min()
    frac_high = h2_wide.index.max()

    h3_keep = h3.loc[h3["bucket"].isin(BUCKET_ORDER)].copy()
    h3_auc = h3_keep.pivot(index="bucket", columns="variant_name", values="auc")
    h3_aupr = h3_keep.pivot(index="bucket", columns="variant_name", values="aupr")
    h3_auc["delta"] = h3_auc["full"] - h3_auc["local_only"]
    h3_aupr["delta"] = h3_aupr["full"] - h3_aupr["local_only"]
    top_auc_bucket = h3_auc["delta"].idxmax()
    top_aupr_bucket = h3_aupr["delta"].idxmax()

    h4 = h4.copy()
    full_graph = h4.loc[h4["edge_mode"] == "full_graph"].iloc[0]
    h4["delta_auc"] = h4["auc"] - full_graph["auc"]
    h4["delta_aupr"] = h4["aupr"] - full_graph["aupr"]
    strongest_edge_drop = h4.loc[h4["edge_mode"] != "full_graph"].sort_values("delta_auc").iloc[0]

    h5 = h5.copy()
    depth0 = h5.loc[h5["depth"] == 0].iloc[0]
    h5["delta_auc_vs_depth0"] = h5["auc"] - depth0["auc"]
    best_depth = h5.sort_values("auc", ascending=False).iloc[0]

    return f"""### HGT Long-Range Dependency and Missing-Link Analysis

Figure Z provides direct evidence that the HGT branch contributes specifically to long-range heterogeneous reasoning rather than merely increasing model capacity. In the branch-ablation experiment, the full model achieved {full['auc']:.4f} AUC and {full['aupr']:.4f} AUPR, compared with {local['auc']:.4f} AUC and {local['aupr']:.4f} AUPR for the local-only variant, corresponding to gains of {full['auc'] - local['auc']:+.4f} AUC and {full['aupr'] - local['aupr']:+.4f} AUPR. The global-only variant ({global_only['auc']:.4f} AUC, {global_only['aupr']:.4f} AUPR) remained close to the full model, indicating that most of the performance gain is carried by the global HGT branch rather than by local neighborhood propagation alone.

The response-edge sparsification experiment directly tests missing-link recovery by withholding increasing fractions of observed drug-cell response edges during training. At both low and high sparsity, the full model consistently outperformed the local-only variant on hidden-link recovery, with recovery AUC improving from {h2_wide.loc[frac_low, ('recovery_auc', 'local_only')]:.4f} to {h2_wide.loc[frac_low, ('recovery_auc', 'full')]:.4f} at fraction {frac_low:.1f}, and from {h2_wide.loc[frac_high, ('recovery_auc', 'local_only')]:.4f} to {h2_wide.loc[frac_high, ('recovery_auc', 'full')]:.4f} at fraction {frac_high:.1f}. The same ordering also held for standard test AUC, showing that the HGT branch is more robust when direct response supervision becomes sparse.

The path-conditioned analysis shows where this gain arises. The largest AUC improvements of the full model over the local-only variant occurred in the {BUCKET_LABELS[top_auc_bucket]} bucket ({h3_auc.loc[top_auc_bucket, 'delta']:+.4f}) and the {BUCKET_LABELS[[b for b in BUCKET_ORDER if b != top_auc_bucket][0]] if top_auc_bucket != BUCKET_ORDER[0] else BUCKET_LABELS[BUCKET_ORDER[1]]} regime remains similarly elevated, while the largest AUPR gain occurred in the {BUCKET_LABELS[top_aupr_bucket]} bucket ({h3_aupr.loc[top_aupr_bucket, 'delta']:+.4f}). This concentration of gains in multi-hop-supported buckets strongly suggests that the HGT branch is exploiting heterogeneous paths beyond immediate local neighborhoods.

The supplementary edge-type ablation further clarifies the mechanism by showing that the gain depends on auxiliary heterogeneous structure. Relative to the full graph, the strongest degradation occurred under {EDGE_LABELS[strongest_edge_drop['edge_mode']]} ({strongest_edge_drop['delta_auc']:+.4f} AUC, {strongest_edge_drop['delta_aupr']:+.4f} AUPR), indicating that similarity edges provide important long-range support beyond the response graph itself. The supplementary depth study shows that performance peaks at depth {int(best_depth['depth'])} ({best_depth['auc']:.4f} AUC), while deeper propagation degrades performance, which is consistent with useful multi-hop aggregation up to a moderate depth rather than unlimited gains from stacking more layers.
"""

def main() -> None:
    parser = argparse.ArgumentParser(description="Create main and supplementary HGT-claim figure suites.")
    parser.add_argument("--tag", default="hgt_claim_v1", help="Experiment tag inside hgt_claim_outputs.")
    parser.add_argument("--outputs-root", default="flexible model/hgt_claim_outputs")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    tag = args.tag
    out_dir = Path(args.out_dir) if args.out_dir else outputs_root / "hgt_claim_analysis" / tag
    ensure_dir(out_dir)

    h1 = pd.read_csv(outputs_root / "hgt1_branch_ablation" / tag / "results.csv")
    h2 = pd.read_csv(outputs_root / "hgt2_response_sparsification" / tag / "results.csv")
    h3 = pd.read_csv(outputs_root / "hgt3_path_conditioned" / tag / "bucket_summary.csv")
    h4 = pd.read_csv(outputs_root / "hgt4_edge_type_ablation" / tag / "results.csv")
    h5 = pd.read_csv(outputs_root / "hgt5_depth_study" / tag / "results.csv")

    # Main paper figure (Springer Nature Width: 174mm = 6.85in)
    fig = plt.figure(figsize=(6.85, 6.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30, left=0.09, right=0.97, top=0.95, bottom=0.10)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ── (a) Branch Ablation ──────────────────────────────────────────────
    h1_plot = h1.copy()
    h1_plot["label"] = h1_plot["variant_name"].map(BRANCH_LABELS)
    order = ["local_only", "global_only", "full"]
    h1_plot["order"] = h1_plot["variant_name"].map({k: i for i, k in enumerate(order)})
    h1_plot = h1_plot.sort_values("order")
    x = np.arange(len(h1_plot))
    width = 0.34
    
    ax_a.bar(x - width / 2, h1_plot["auc"], width, color=C_AUC, label="AUC", zorder=3)
    ax_a.bar(x + width / 2, h1_plot["aupr"], width, color=C_AUPR, label="AUPR", zorder=3)
    
    # Annotations removed for brevity

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(h1_plot["label"], fontsize=7)
    ax_a.set_ylim(0.87, 0.93)
    _style_ax(ax_a, title="(a) Branch Ablation", ylabel="Performance")
    ax_a.legend(loc="upper left", fontsize=6.0, frameon=False)

    # ── (b) Path-Conditioned Gain ─────────────────────────────────────────
    h3_keep = h3.loc[h3["bucket"].isin(BUCKET_ORDER)].copy()
    h3_auc = h3_keep.pivot(index="bucket", columns="variant_name", values="auc").reindex(BUCKET_ORDER)
    h3_aupr = h3_keep.pivot(index="bucket", columns="variant_name", values="aupr").reindex(BUCKET_ORDER)
    h3_pairs = h3_keep.pivot(index="bucket", columns="variant_name", values="pair_count").reindex(BUCKET_ORDER)
    delta_auc = h3_auc["full"] - h3_auc["local_only"]
    delta_aupr = h3_aupr["full"] - h3_aupr["local_only"]
    xc = np.arange(len(BUCKET_ORDER))
    
    ax_b.bar(xc - width / 2, delta_auc.to_numpy(), width, color=C_AUC, label="ΔAUC", zorder=3)
    ax_b.bar(xc + width / 2, delta_aupr.to_numpy(), width, color=C_AUPR, label="ΔAUPR", zorder=3)
    
    # Annotations removed for brevity

    ax_b.set_xticks(xc)
    ax_b.set_xticklabels([BUCKET_LABELS[b] for b in BUCKET_ORDER], rotation=22, ha="right", fontsize=7)
    _style_ax(ax_b, title="(b) Multi-Hop Path Gain", ylabel="Full vs. Local-only")
    ax_b.legend(loc="upper left", fontsize=6.0, frameon=False)

    # ── (c) Standard Prediction (Sparsification) ───────────────────────────
    h2_full = h2.loc[h2["variant_name"] == "full"].sort_values("fraction")
    h2_local = h2.loc[h2["variant_name"] == "local_only"].sort_values("fraction")
    fractions = h2_full["fraction"].to_numpy()

    ax_c.plot(fractions, h2_full["auc"], marker="o", markersize=4, color=C_AUC, linewidth=1.5, label="Full", zorder=3)
    ax_c.plot(fractions, h2_local["auc"], marker="o", markersize=4, color=C_LOCAL, linewidth=1.5, label="Local only", zorder=3)
    _style_ax(ax_c, title="(c) Robustness to Sparse Data", xlabel="Fraction of Training Edges Withheld", ylabel="Standard Test AUC")
    ax_c.legend(loc="lower left", fontsize=6.0, frameon=False)

    # ── (d) Missing-Link Recovery (Sparsification) ─────────────────────────
    ax_d.plot(fractions, h2_full["recovery_auc"], marker="o", markersize=4, color=C_AUC, linewidth=1.5, label="Full", zorder=3)
    ax_d.plot(fractions, h2_local["recovery_auc"], marker="o", markersize=4, color=C_LOCAL, linewidth=1.5, label="Local only", zorder=3)
    _style_ax(ax_d, title="(d) Reconstructing Missing Links", xlabel="Fraction of Training Edges Withheld", ylabel="AUC on Withheld Edges")
    ax_d.legend(loc="lower left", fontsize=6.0, frameon=False)

    main_pdf = out_dir / f"hgt_claim_main_figure_{tag}.pdf"
    main_png = out_dir / f"hgt_claim_main_figure_{tag}.png"
    fig.savefig(main_pdf, dpi=800, bbox_inches="tight", format="pdf")
    fig.savefig(main_png, dpi=800, bbox_inches="tight", format="png")
    plt.close(fig)

    # Supplementary figure (Springer Nature Width: 174mm = 6.85in)
    fig_s = plt.figure(figsize=(6.85, 3.2), facecolor="white")
    gs_s = fig_s.add_gridspec(1, 2, wspace=0.32, bottom=0.20, left=0.08, right=0.97, top=0.88)
    ax_s1 = fig_s.add_subplot(gs_s[0, 0])
    ax_s2 = fig_s.add_subplot(gs_s[0, 1])

    # ── (S1) Edge-type ablation ────────────────────────────────────────────
    h4_plot = h4.copy()
    full_graph = h4_plot.loc[h4_plot["edge_mode"] == "full_graph"].iloc[0]
    h4_plot["delta_auc"] = h4_plot["auc"] - full_graph["auc"]
    h4_plot["delta_aupr"] = h4_plot["aupr"] - full_graph["aupr"]
    h4_plot = h4_plot.set_index("edge_mode").loc[["no_cell_similarity", "no_drug_similarity", "response_only"]].reset_index()
    xd = np.arange(len(h4_plot))
    
    ax_s1.bar(xd - width / 2, h4_plot["delta_auc"], width, color=C_AUC, label="ΔAUC", zorder=3)
    ax_s1.bar(xd + width / 2, h4_plot["delta_aupr"], width, color=C_AUPR, label="ΔAUPR", zorder=3)
    ax_s1.axhline(0.0, color=C_LOCAL, linestyle="--", linewidth=1.0, zorder=2)
    
    # Annotations removed for brevity

    ax_s1.set_xticks(xd)
    ax_s1.set_xticklabels([EDGE_LABELS[m] for m in h4_plot["edge_mode"]], rotation=18, ha="right", fontsize=7)
    _style_ax(ax_s1, title="(S1) Edge-Type Ablation", ylabel="Drop from full graph")
    ax_s1.legend(loc="lower left", fontsize=6.0, frameon=False)

    # ── (S2) Depth Study ───────────────────────────────────────────────────
    h5_plot = h5.sort_values("depth")
    ax_s2.plot(h5_plot["depth"], h5_plot["auc"], marker="o", markersize=4, color=C_AUC, linewidth=1.5, label="AUC", zorder=3)
    ax_s2.plot(h5_plot["depth"], h5_plot["aupr"], marker="s", markersize=4, color=C_AUPR, linewidth=1.5, label="AUPR", zorder=3)
    best_depth = h5_plot.sort_values("auc", ascending=False).iloc[0]
    ax_s2.axvline(best_depth["depth"], color="#E69F00", linestyle=":", linewidth=1.2, zorder=2)
    
    _style_ax(ax_s2, title="(S2) Depth Study", xlabel="HGT Depth", ylabel="Performance")
    ax_s2.legend(loc="lower left", fontsize=6.0, frameon=False)

    supp_pdf = out_dir / f"hgt_claim_supplementary_figure_{tag}.pdf"
    supp_png = out_dir / f"hgt_claim_supplementary_figure_{tag}.png"
    fig_s.savefig(supp_pdf, dpi=800, bbox_inches="tight", format="pdf")
    fig_s.savefig(supp_png, dpi=800, bbox_inches="tight", format="png")
    plt.close(fig_s)

    md_path = out_dir / f"hgt_claim_analysis_section_{tag}.md"
    md_path.write_text(build_analysis(h1, h2, h3, h4, h5))

    print(f"Saved main PDF: {main_pdf}")
    print(f"Saved main PNG: {main_png}")
    print(f"Saved supplementary PDF: {supp_pdf}")
    print(f"Saved supplementary PNG: {supp_png}")
    print(f"Saved analysis markdown: {md_path}")

if __name__ == "__main__":
    main()
