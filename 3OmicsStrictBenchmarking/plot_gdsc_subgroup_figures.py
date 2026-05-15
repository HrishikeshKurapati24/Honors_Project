import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_MODEL_ORDER = [
    "FUSECDR",
    "GraphCDR",
    "RedCDR",
    "GADRP",
    "DeepTTC",
    "GraphDRP",
]

DEFAULT_MODEL_COLORS = {
    "FUSECDR": "#111111",
    "GraphCDR": "#3B6CE1",
    "RedCDR": "#E41A1C",
    "GADRP": "#2E8B57",
    "DeepTTC": "#7AA6D9",
    "GraphDRP": "#F39C12",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create GDSC subgroup figures from 3OmicsStrictBenchmarking random/dataset-2 results."
    )
    parser.add_argument(
        "--results-root",
        default="results/random/dataset-2",
        help="Model result root containing per-model fold_*_predictions.csv files.",
    )
    parser.add_argument(
        "--prepared-response-pairs",
        default="prepared/dataset-2/response_pairs.csv",
        help="Prepared dataset-2 response_pairs.csv used to identify the most common drugs.",
    )
    parser.add_argument(
        "--cell-annotations",
        default="../data/CCLE/Processed data/Cell_lines_annotations.csv",
        help="CCLE annotation file mapping depMapID to CCLE_ID.",
    )
    parser.add_argument(
        "--drug-name-map",
        default="../data/GDSC/Initial data/pubchem_cids_1.csv",
        help="CSV mapping PUBCHEM_CID to Drug Name.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/gdsc_random_dataset2",
        help="Directory where plots and subgroup metric CSVs will be written.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=["GraTransDRP"],
        help="Models to exclude from the figures.",
    )
    parser.add_argument(
        "--top-drugs",
        type=int,
        default=10,
        help="Number of most common drugs to include in the drug-level figure.",
    )
    parser.add_argument(
        "--top-tissues",
        type=int,
        default=5,
        help="Number of most common tissues to include in the radar plot.",
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def nice_tissue_name(raw_name: str) -> str:
    if pd.isna(raw_name):
        return "Unknown"
    special = {
        "HAEMATOPOIETIC_AND_LYMPHOID_TISSUE": "Haematopoietic\n/lymphoid",
        "LARGE_INTESTINE": "Large intestine",
        "CENTRAL_NERVOUS_SYSTEM": "Central nervous\nsystem",
        "UPPER_AERODIGESTIVE_TRACT": "Upper aerodigestive\ntract",
        "AUTONOMIC_GANGLIA": "Autonomic\nganglia",
        "URINARY_TRACT": "Urinary\ntract",
        "SOFT_TISSUE": "Soft tissue",
    }
    if raw_name in special:
        return special[raw_name]
    return raw_name.replace("_", " ").title()


def compute_binary_metrics(df: pd.DataFrame) -> tuple[float, float]:
    labels = df["label"].to_numpy(dtype=np.int64)
    preds = df["prediction"].to_numpy(dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(labels, preds), average_precision_score(labels, preds)


def load_model_predictions(model_dir: Path) -> pd.DataFrame:
    fold_frames = []
    for fold_idx in range(1, 6):
        pred_path = model_dir / f"fold_{fold_idx}_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing fold predictions: {pred_path}")
        fold_df = pd.read_csv(pred_path)
        fold_df["fold"] = fold_idx
        fold_frames.append(fold_df)
    merged = pd.concat(fold_frames, ignore_index=True)
    merged["cell_id"] = merged["cell_id"].astype(str)
    merged["drug_id"] = merged["drug_id"].astype(str)
    return merged


def prepare_reference_tables(
    annotations_path: Path,
    drug_name_map_path: Path,
    prepared_response_pairs_path: Path,
):
    annotations = pd.read_csv(annotations_path, usecols=["CCLE_ID", "depMapID"]).dropna()
    annotations["depMapID"] = annotations["depMapID"].astype(str)
    annotations["tissue_raw"] = annotations["CCLE_ID"].str.split("_", n=1).str[1]

    drug_map = pd.read_csv(drug_name_map_path)
    drug_map["PUBCHEM_CID"] = drug_map["PUBCHEM_CID"].astype(str)
    drug_map = drug_map.rename(columns={"PUBCHEM_CID": "drug_id", "Drug Name": "drug_name"})

    response_pairs = pd.read_csv(prepared_response_pairs_path)
    response_pairs["drug_id"] = response_pairs["drug_id"].astype(str)
    return annotations, drug_map, response_pairs


def collect_subgroup_metrics(
    results_root: Path,
    annotations: pd.DataFrame,
    drug_map: pd.DataFrame,
    response_pairs: pd.DataFrame,
    exclude_models: set[str],
    top_tissues: int,
    top_drugs: int,
):
    available_models = []
    for d in results_root.iterdir():
        if d.is_dir() and d.name not in exclude_models:
            name = d.name
            if name == "FUSECDR":
                name = "FUSECDR"
            available_models.append(name)
    available_models.sort()

    model_order = [m for m in DEFAULT_MODEL_ORDER if m in available_models]
    remaining = [m for m in available_models if m not in model_order]
    model_order.extend(remaining)

    reference_model = model_order[0]
    actual_ref_dir = "FUSECDR" if reference_model == "FUSECDR" else reference_model
    reference_predictions = load_model_predictions(results_root / actual_ref_dir)
    reference_predictions = reference_predictions.merge(
        annotations[["depMapID", "CCLE_ID", "tissue_raw"]],
        left_on="cell_id",
        right_on="depMapID",
        how="left",
    )

    top_tissue_ids = (
        reference_predictions["tissue_raw"].value_counts().head(top_tissues).index.tolist()
    )
    top_drug_ids = response_pairs["drug_id"].value_counts().head(top_drugs).index.tolist()

    tissue_rows = []
    drug_rows = []

    for model_name in model_order:
        actual_dir = "FUSECDR" if model_name == "FUSECDR" else model_name
        pred_df = load_model_predictions(results_root / actual_dir)
        pred_df = pred_df.merge(
            annotations[["depMapID", "CCLE_ID", "tissue_raw"]],
            left_on="cell_id",
            right_on="depMapID",
            how="left",
        )

        for tissue_id in top_tissue_ids:
            tissue_df = pred_df[pred_df["tissue_raw"] == tissue_id]
            auc, aupr = compute_binary_metrics(tissue_df)
            tissue_rows.append(
                {
                    "model": model_name,
                    "tissue_raw": tissue_id,
                    "tissue_display": nice_tissue_name(tissue_id),
                    "pair_count": len(tissue_df),
                    "auc": auc,
                    "aupr": aupr,
                }
            )

        for drug_id in top_drug_ids:
            drug_df = pred_df[pred_df["drug_id"] == drug_id]
            auc, aupr = compute_binary_metrics(drug_df)
            drug_name_row = drug_map[drug_map["drug_id"] == drug_id]
            drug_name = drug_name_row["drug_name"].iloc[0] if not drug_name_row.empty else drug_id
            drug_rows.append(
                {
                    "model": model_name,
                    "drug_id": drug_id,
                    "drug_name": drug_name,
                    "pair_count": len(drug_df),
                    "auc": auc,
                    "aupr": aupr,
                }
            )

    tissue_metrics = pd.DataFrame(tissue_rows)
    drug_metrics = pd.DataFrame(drug_rows)
    return model_order, top_tissue_ids, top_drug_ids, tissue_metrics, drug_metrics


def build_radar_plot(
    tissue_metrics: pd.DataFrame,
    model_order: list[str],
    top_tissue_ids: list[str],
    output_dir: Path,
):
    tissue_labels = [
        tissue_metrics.loc[tissue_metrics["tissue_raw"] == tissue_id, "tissue_display"].iloc[0]
        for tissue_id in top_tissue_ids
    ]
    metrics = ["auc", "aupr"]
    titles = {"auc": "AUC", "aupr": "AUPR"}

    angles = np.linspace(0, 2 * np.pi, len(top_tissue_ids), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )

    for ax, metric in zip(axes, metrics):
        pivot = (
            tissue_metrics.pivot(index="model", columns="tissue_raw", values=metric)
            .reindex(index=model_order, columns=top_tissue_ids)
            * 100.0
        )
        finite_values = pivot.to_numpy().ravel()
        finite_values = finite_values[np.isfinite(finite_values)]
        lower = math.floor((finite_values.min() - 1.0) / 5.0) * 5.0
        upper = math.ceil((finite_values.max() + 1.0) / 5.0) * 5.0
        ticks = np.linspace(lower, upper, 4)

        for model_name in model_order:
            values = pivot.loc[model_name].tolist()
            values += values[:1]
            ax.plot(
                angles,
                values,
                linewidth=2.4,
                color=DEFAULT_MODEL_COLORS.get(model_name),
                label=model_name,
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tissue_labels, fontsize=14)
        ax.set_ylim(lower, upper)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.0f}" for tick in ticks], fontsize=11)
        ax.set_title(titles[metric], fontsize=26, pad=16)
        ax.grid(alpha=0.45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=16,
        bbox_to_anchor=(0.5, -0.05),
        handlelength=3.2,
    )
    fig.suptitle(
        "GDSC Tissue-Level Performance by Model",
        fontsize=22,
        y=1.05,
    )

    png_path = output_dir / "gdsc_tissue_radar_random_dataset2.png"
    pdf_path = output_dir / "gdsc_tissue_radar_random_dataset2.pdf"
    fig.savefig(png_path, dpi=800, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(ax, matrix: pd.DataFrame, title: str):
    values = matrix.to_numpy(dtype=float) * 100.0
    # Use pcolormesh for vectorized output instead of imshow
    rows, cols = values.shape
    im = ax.pcolormesh(
        np.arange(cols + 1) - 0.5,
        np.arange(rows + 1) - 0.5,
        values,
        cmap="YlGnBu",
        shading='flat', edgecolors='none', antialiased=False
    )
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5) # Match imshow upper origin
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=12)
    ax.set_title(title, fontsize=20, pad=12)

    return im


def build_drug_heatmaps(
    drug_metrics: pd.DataFrame,
    model_order: list[str],
    top_drug_ids: list[str],
    output_dir: Path,
):
    ordered_names = []
    for drug_id in top_drug_ids:
        ordered_names.append(
            drug_metrics.loc[drug_metrics["drug_id"] == drug_id, "drug_name"].iloc[0]
        )

    auc_matrix = (
        drug_metrics.assign(drug_name=pd.Categorical(drug_metrics["drug_name"], ordered_names))
        .pivot(index="drug_name", columns="model", values="auc")
        .reindex(index=ordered_names, columns=model_order)
    )
    aupr_matrix = (
        drug_metrics.assign(drug_name=pd.Categorical(drug_metrics["drug_name"], ordered_names))
        .pivot(index="drug_name", columns="model", values="aupr")
        .reindex(index=ordered_names, columns=model_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    im_auc = draw_heatmap(axes[0], auc_matrix, "Drug-Level AUC")
    im_aupr = draw_heatmap(axes[1], aupr_matrix, "Drug-Level AUPR")
    cbar_auc = fig.colorbar(im_auc, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_auc.set_label("Percentage", fontsize=12)
    cbar_auc.solids.set_rasterized(False)
    cbar_aupr = fig.colorbar(im_aupr, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_aupr.set_label("Percentage", fontsize=12)
    cbar_aupr.solids.set_rasterized(False)
    fig.suptitle(
        "GDSC Top-10 Most Common Drugs by Model",
        fontsize=22,
    )

    png_path = output_dir / "gdsc_top10_drug_heatmaps_random_dataset2.png"
    pdf_path = output_dir / "gdsc_top10_drug_heatmaps_random_dataset2.pdf"
    fig.savefig(png_path, dpi=800, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    results_root = resolve_path(base_dir, args.results_root)
    prepared_response_pairs_path = resolve_path(base_dir, args.prepared_response_pairs)
    annotations_path = resolve_path(base_dir, args.cell_annotations)
    drug_name_map_path = resolve_path(base_dir, args.drug_name_map)
    output_dir = resolve_path(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations, drug_map, response_pairs = prepare_reference_tables(
        annotations_path,
        drug_name_map_path,
        prepared_response_pairs_path,
    )

    model_order, top_tissue_ids, top_drug_ids, tissue_metrics, drug_metrics = collect_subgroup_metrics(
        results_root=results_root,
        annotations=annotations,
        drug_map=drug_map,
        response_pairs=response_pairs,
        exclude_models=set(args.exclude_models),
        top_tissues=args.top_tissues,
        top_drugs=args.top_drugs,
    )

    tissue_metrics.to_csv(output_dir / "gdsc_tissue_metrics_random_dataset2.csv", index=False)
    drug_metrics.to_csv(output_dir / "gdsc_top10_drug_metrics_random_dataset2.csv", index=False)

    build_radar_plot(tissue_metrics, model_order, top_tissue_ids, output_dir)
    build_drug_heatmaps(drug_metrics, model_order, top_drug_ids, output_dir)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
