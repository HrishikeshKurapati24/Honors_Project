import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from hgt_claim_runner import run_hgt_training_experiment
from hgt_claim_utils import (
    HGT_CLAIM_OUTPUTS_DIR,
    ensure_hgt_claim_split_dir,
    load_experiment_rows,
    save_experiment_progress,
)
from flexibility_utils import FINAL_DATASET_DIR


VARIANTS = [
    {"variant_name": "full", "use_local_branch": True, "use_global_branch": True},
    {"variant_name": "local_only", "use_local_branch": True, "use_global_branch": False},
]


def parse_args():
    parser = argparse.ArgumentParser(description="HGT-claim experiment 2: response-edge sparsification.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = HGT_CLAIM_OUTPUTS_DIR / "hgt2_response_sparsification" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_hgt_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    rows = load_experiment_rows(experiment_dir / "results.csv")
    completed = {(row["variant_name"], float(row["fraction"])) for row in rows}
    metadata = {
        "experiment": "hgt2_response_sparsification",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "fractions": args.fractions,
        "variants": VARIANTS,
    }
    save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata, sort_metric="recovery_auc")

    for fraction in args.fractions:
        for variant in VARIANTS:
            key = (variant["variant_name"], float(fraction))
            if key in completed:
                continue
            results_dir = experiment_dir / "subruns" / f"{variant['variant_name']}_frac_{str(fraction).replace('.', 'p')}"
            summary = run_hgt_training_experiment(
                dataset_root=dataset_root,
                split_dir=split_dir,
                results_dir=results_dir,
                variant_name=variant["variant_name"],
                use_local_branch=variant["use_local_branch"],
                use_global_branch=variant["use_global_branch"],
                response_sparsity_fraction=fraction,
                device=args.device,
                seed=args.seed,
                epochs=args.epochs,
                fold_ids=args.fold_ids,
            )
            fold_frame = pd.DataFrame(summary["folds"])
            rows.append(
                {
                    "variant_name": variant["variant_name"],
                    "fraction": float(fraction),
                    "auc": float(summary["mean"].get("test_auc", 0.0)),
                    "aupr": float(summary["mean"].get("test_aupr", 0.0)),
                    "f1": float(summary["mean"].get("test_f1", 0.0)),
                    "acc": float(summary["mean"].get("test_acc", 0.0)),
                    "recovery_auc": float(fold_frame["recovery_auc"].mean()) if "recovery_auc" in fold_frame.columns else 0.0,
                    "recovery_aupr": float(fold_frame["recovery_aupr"].mean()) if "recovery_aupr" in fold_frame.columns else 0.0,
                    "recovery_f1": float(fold_frame["recovery_f1"].mean()) if "recovery_f1" in fold_frame.columns else 0.0,
                    "recovery_acc": float(fold_frame["recovery_acc"].mean()) if "recovery_acc" in fold_frame.columns else 0.0,
                    "subrun_dir": str(results_dir),
                }
            )
            save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata, sort_metric="recovery_auc")

    pd.DataFrame(rows).sort_values(["fraction", "recovery_auc"], ascending=[True, False]).to_csv(
        experiment_dir / "leaderboard.csv",
        index=False,
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
