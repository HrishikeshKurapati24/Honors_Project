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


EDGE_MODES = [
    "full_graph",
    "no_drug_similarity",
    "no_cell_similarity",
    "response_only",
]


def parse_args():
    parser = argparse.ArgumentParser(description="HGT-claim experiment 4: edge-type ablation.")
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
    experiment_dir = HGT_CLAIM_OUTPUTS_DIR / "hgt4_edge_type_ablation" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_hgt_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    rows = load_experiment_rows(experiment_dir / "results.csv")
    completed = {row["edge_mode"] for row in rows}
    metadata = {
        "experiment": "hgt4_edge_type_ablation",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "edge_modes": EDGE_MODES,
    }
    save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    for edge_mode in EDGE_MODES:
        if edge_mode in completed:
            continue
        results_dir = experiment_dir / "subruns" / edge_mode
        summary = run_hgt_training_experiment(
            dataset_root=dataset_root,
            split_dir=split_dir,
            results_dir=results_dir,
            variant_name="full",
            use_local_branch=True,
            use_global_branch=True,
            edge_mode=edge_mode,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            fold_ids=args.fold_ids,
        )
        rows.append(
            {
                "edge_mode": edge_mode,
                "auc": float(summary["mean"].get("test_auc", 0.0)),
                "aupr": float(summary["mean"].get("test_aupr", 0.0)),
                "f1": float(summary["mean"].get("test_f1", 0.0)),
                "acc": float(summary["mean"].get("test_acc", 0.0)),
                "subrun_dir": str(results_dir),
            }
        )
        completed.add(edge_mode)
        save_experiment_progress(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    pd.DataFrame(rows).sort_values("auc", ascending=False).to_csv(experiment_dir / "leaderboard.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
