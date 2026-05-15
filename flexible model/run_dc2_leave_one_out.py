import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from dataset_claim_utils import (
    DATASET_CLAIM_OUTPUTS_DIR,
    ensure_dataset_claim_split_dir,
    get_leave_one_out_configs,
    result_row_from_summary,
    run_or_reuse_flexible,
    save_incremental_experiment_artifacts,
    slugify,
)
from flexibility_utils import FINAL_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-claim experiment 2: leave-one-out modality analysis.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = DATASET_CLAIM_OUTPUTS_DIR / "dc2_leave_one_out" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_dataset_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    progress_path = experiment_dir / "results.csv"
    existing = pd.read_csv(progress_path) if progress_path.is_file() else pd.DataFrame()
    completed = set(existing["config_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict(orient="records") if not existing.empty else []

    configs = get_leave_one_out_configs(dataset_root)
    metadata = {
        "experiment": "dc2_leave_one_out",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "configs": configs,
    }
    save_incremental_experiment_artifacts(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    for config in configs:
        if config["name"] in completed:
            continue
        run_name = f"dc2_{tag}_{slugify(config['name'])}"
        summary = run_or_reuse_flexible(
            run_name=run_name,
            dataset_root=dataset_root,
            omics=config["omics"],
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
            split_dir=split_dir,
        )
        row = result_row_from_summary(
            experiment="dc2_leave_one_out",
            config_name=config["name"],
            omics=config["omics"],
            run_name=run_name,
            summary=summary,
            extra_fields={"removed_stem": config.get("removed_stem", "")},
        )
        rows.append(row)
        completed.add(config["name"])
        save_incremental_experiment_artifacts(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    results_df = pd.DataFrame(rows)
    full_row = results_df[results_df["config_name"] == "full_7omics"]
    if not full_row.empty:
        baseline_auc = float(full_row.iloc[0]["auc"])
        baseline_aupr = float(full_row.iloc[0]["aupr"])
        baseline_f1 = float(full_row.iloc[0]["f1"])
        baseline_acc = float(full_row.iloc[0]["acc"])
        results_df["delta_auc_vs_full"] = results_df["auc"] - baseline_auc
        results_df["delta_aupr_vs_full"] = results_df["aupr"] - baseline_aupr
        results_df["delta_f1_vs_full"] = results_df["f1"] - baseline_f1
        results_df["delta_acc_vs_full"] = results_df["acc"] - baseline_acc
    results_df.to_csv(experiment_dir / "leave_one_out_deltas.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
