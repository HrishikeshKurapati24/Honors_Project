import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from data_flexible import dataload_flexible
from hgt_claim_utils import (
    HGT_CLAIM_OUTPUTS_DIR,
    compute_path_support_rows,
    ensure_hgt_claim_split_dir,
    load_experiment_rows,
    load_fold_tables,
    load_prediction_rows,
    save_experiment_progress,
    save_prediction_rows,
    sorted_entity_ids,
)
from main_flexible import metrics_graph
from flexibility_utils import FINAL_DATASET_DIR


VARIANTS = ["full", "local_only"]


def parse_args():
    parser = argparse.ArgumentParser(description="HGT-claim experiment 3: path-conditioned evaluation.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--branch-ablation-dir", type=str, default=None, help="Path to completed hgt1_branch_ablation/<tag> directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = HGT_CLAIM_OUTPUTS_DIR / "hgt3_path_conditioned" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    branch_ablation_dir = (
        Path(args.branch_ablation_dir).resolve()
        if args.branch_ablation_dir
        else HGT_CLAIM_OUTPUTS_DIR / "hgt1_branch_ablation" / tag
    )
    split_dir = ensure_hgt_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    loaded = dataload_flexible(str(dataset_root))
    cell_ids, drug_ids = sorted_entity_ids(loaded.data_new)
    cell_similarity_df = loaded.similarity_feature.loc[cell_ids]
    drug_phys_df = pd.DataFrame.from_dict(
        loaded.physicochemical_feature,
        orient="index",
    ).loc[drug_ids]

    results_rows = load_experiment_rows(experiment_dir / "results.csv")
    completed = {(row["variant_name"], int(row["fold"])) for row in results_rows}
    metadata = {
        "experiment": "hgt3_path_conditioned",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "branch_ablation_dir": str(branch_ablation_dir),
        "top_k": args.top_k,
        "variants": VARIANTS,
    }
    save_experiment_progress(experiment_dir=experiment_dir, rows=results_rows, metadata=metadata)

    resolved_folds = args.fold_ids or sorted(
        int(path.name.split("_", 1)[1])
        for path in Path(split_dir).glob("fold_*")
        if path.is_dir()
    )
    for fold_id in resolved_folds:
        split_tables = load_fold_tables(split_dir, fold_id)
        for variant_name in VARIANTS:
            if (variant_name, fold_id) in completed:
                continue
            prediction_path = branch_ablation_dir / "subruns" / variant_name / f"fold_{fold_id}_predictions.csv"
            prediction_df = pd.read_csv(prediction_path)
            prediction_df["cell_id"] = prediction_df["cell_id"].astype(str)
            prediction_df["drug_id"] = prediction_df["drug_id"].astype(str)
            eval_pairs = prediction_df[["cell_id", "drug_id", "label"]].copy()
            support_df = compute_path_support_rows(
                train_pairs=split_tables["train"],
                eval_pairs=eval_pairs,
                cell_similarity_df=cell_similarity_df,
                drug_similarity_df=drug_phys_df,
                top_k=args.top_k,
            )
            support_df["cell_id"] = support_df["cell_id"].astype(str)
            support_df["drug_id"] = support_df["drug_id"].astype(str)
            merged = prediction_df.merge(
                support_df[["cell_id", "drug_id", "bucket", "drug_2hop", "cell_2hop", "three_hop_only"]],
                on=["cell_id", "drug_id"],
                how="left",
            )
            save_prediction_rows(
                experiment_dir / f"{variant_name}_fold_{fold_id}_bucket_predictions.csv",
                merged.to_dict(orient="records"),
            )

            bucket_rows = []
            for bucket, bucket_df in merged.groupby("bucket"):
                auc, aupr, f1, acc = metrics_graph(
                    bucket_df["label"].to_numpy(dtype=np.int64),
                    bucket_df["prediction"].to_numpy(dtype=np.float32),
                )
                bucket_rows.append(
                    {
                        "variant_name": variant_name,
                        "fold": fold_id,
                        "bucket": bucket,
                        "pair_count": int(len(bucket_df)),
                        "auc": float(auc),
                        "aupr": float(aupr),
                        "f1": float(f1),
                        "acc": float(acc),
                    }
                )
            pd.DataFrame(bucket_rows).to_csv(
                experiment_dir / f"{variant_name}_fold_{fold_id}_bucket_metrics.csv",
                index=False,
            )
            results_rows.extend(bucket_rows)
            completed.add((variant_name, fold_id))
            save_experiment_progress(experiment_dir=experiment_dir, rows=results_rows, metadata=metadata)

    summary_df = pd.DataFrame(results_rows)
    if not summary_df.empty:
        aggregate = (
            summary_df.groupby(["variant_name", "bucket"], as_index=False)
            .agg(
                pair_count=("pair_count", "sum"),
                auc=("auc", "mean"),
                aupr=("aupr", "mean"),
                f1=("f1", "mean"),
                acc=("acc", "mean"),
            )
        )
        aggregate.to_csv(experiment_dir / "bucket_summary.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
