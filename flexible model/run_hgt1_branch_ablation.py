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
    {"variant_name": "global_only", "use_local_branch": False, "use_global_branch": True},
]


def parse_args():
    parser = argparse.ArgumentParser(description="HGT-claim experiment 1: branch ablation.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = HGT_CLAIM_OUTPUTS_DIR / "hgt1_branch_ablation" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_hgt_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    rows = load_experiment_rows(experiment_dir / "results.csv")
    completed = {row["variant_name"] for row in rows}
    metadata = {
        "experiment": "hgt1_branch_ablation",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "variants": VARIANTS,
    }
    save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    for variant in VARIANTS:
        if variant["variant_name"] in completed:
            continue
        results_dir = experiment_dir / "subruns" / variant["variant_name"]
        summary = run_hgt_training_experiment(
            dataset_root=dataset_root,
            split_dir=split_dir,
            results_dir=results_dir,
            variant_name=variant["variant_name"],
            use_local_branch=variant["use_local_branch"],
            use_global_branch=variant["use_global_branch"],
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            fold_ids=args.fold_ids,
        )
        rows.append(
            {
                "variant_name": variant["variant_name"],
                "use_local_branch": variant["use_local_branch"],
                "use_global_branch": variant["use_global_branch"],
                "auc": float(summary["mean"].get("test_auc", 0.0)),
                "aupr": float(summary["mean"].get("test_aupr", 0.0)),
                "f1": float(summary["mean"].get("test_f1", 0.0)),
                "acc": float(summary["mean"].get("test_acc", 0.0)),
                "subrun_dir": str(results_dir),
            }
        )
        save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    pd.DataFrame(rows).sort_values("auc", ascending=False).to_csv(experiment_dir / "leaderboard.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
