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


DEPTH_VARIANTS = [
    {
        "depth": 0,
        "variant_name": "depth_0_local_only",
        "use_local_branch": True,
        "use_global_branch": False,
        "num_global_layers": 0,
    },
    {
        "depth": 1,
        "variant_name": "depth_1",
        "use_local_branch": True,
        "use_global_branch": True,
        "num_global_layers": 1,
    },
    {
        "depth": 2,
        "variant_name": "depth_2",
        "use_local_branch": True,
        "use_global_branch": True,
        "num_global_layers": 2,
    },
    {
        "depth": 3,
        "variant_name": "depth_3",
        "use_local_branch": True,
        "use_global_branch": True,
        "num_global_layers": 3,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="HGT-claim experiment 5: HGT depth study.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--fold-ids", nargs="+", type=int, default=None)
    parser.add_argument("--num-local-layers", type=int, default=2)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = HGT_CLAIM_OUTPUTS_DIR / "hgt5_depth_study" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_hgt_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    rows = load_experiment_rows(experiment_dir / "results.csv")
    completed = {int(row["depth"]) for row in rows}
    metadata = {
        "experiment": "hgt5_depth_study",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "num_local_layers": args.num_local_layers,
        "depth_variants": DEPTH_VARIANTS,
    }
    save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    for variant in DEPTH_VARIANTS:
        if variant["depth"] in completed:
            continue
        results_dir = experiment_dir / "subruns" / variant["variant_name"]
        summary = run_hgt_training_experiment(
            dataset_root=dataset_root,
            split_dir=split_dir,
            results_dir=results_dir,
            variant_name=variant["variant_name"],
            use_local_branch=variant["use_local_branch"],
            use_global_branch=variant["use_global_branch"],
            num_local_layers=args.num_local_layers,
            num_global_layers=variant["num_global_layers"],
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            fold_ids=args.fold_ids,
        )
        rows.append(
            {
                "depth": int(variant["depth"]),
                "variant_name": variant["variant_name"],
                "use_local_branch": variant["use_local_branch"],
                "use_global_branch": variant["use_global_branch"],
                "num_local_layers": int(args.num_local_layers),
                "num_global_layers": int(variant["num_global_layers"]),
                "auc": float(summary["mean"].get("test_auc", 0.0)),
                "aupr": float(summary["mean"].get("test_aupr", 0.0)),
                "f1": float(summary["mean"].get("test_f1", 0.0)),
                "acc": float(summary["mean"].get("test_acc", 0.0)),
                "subrun_dir": str(results_dir),
            }
        )
        completed.add(int(variant["depth"]))
        save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    pd.DataFrame(rows).sort_values("depth", ascending=True).to_csv(experiment_dir / "leaderboard.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
