import argparse
import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from flexibility_utils import (
    FINAL_DATASET_DIR,
    FLEXIBILITY_OUTPUTS_DIR,
    list_real_omics_stems,
    run_main_flexible,
    summarize_flexible_summary,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: random real-omics subset cardinality study.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--subsets-per-k", type=int, default=3)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    omics_pool = list_real_omics_stems(dataset_root)
    if not omics_pool:
        raise ValueError(f"No real omics stems found in {dataset_root}")

    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = FLEXIBILITY_OUTPUTS_DIR / "exp1_subset_cardinality" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for subset_size in range(1, len(omics_pool) + 1):
        candidates = list(itertools.combinations(omics_pool, subset_size))
        rng.shuffle(candidates)
        selected_subsets = candidates[: min(args.subsets_per_k, len(candidates))]
        for subset_index, subset in enumerate(selected_subsets, start=1):
            subset_stems = list(subset)
            run_name = f"exp1_{tag}_k{subset_size}_subset{subset_index}"
            summary = run_main_flexible(
                run_name=run_name,
                dataset_root=dataset_root,
                omics=subset_stems,
                epoch=args.epochs,
                k_fold=args.k_fold,
                seed=args.seed,
                device=args.device,
            )
            metrics = summarize_flexible_summary(summary)
            rows.append(
                {
                    "subset_size": subset_size,
                    "subset_index": subset_index,
                    "omics_stems": "|".join(subset_stems),
                    "run_name": run_name,
                    "run_dir": summary["run_dir"],
                    **metrics,
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(experiment_dir / "results.csv", index=False)
    aggregate_df = (
        results_df.groupby("subset_size", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_std=("aupr", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            elapsed_mean=("elapsed_seconds", "mean"),
            fold_elapsed_mean=("mean_fold_elapsed_seconds", "mean"),
            peak_gpu_memory_max=("max_peak_gpu_memory_bytes", "max"),
            parameter_count_max=("parameter_count", "max"),
        )
    )
    aggregate_df.to_csv(experiment_dir / "aggregate_by_k.csv", index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "exp1_subset_cardinality",
            "dataset_root": str(dataset_root),
            "omics_pool": omics_pool,
            "subsets_per_k": args.subsets_per_k,
            "epochs": args.epochs,
            "k_fold": args.k_fold,
            "seed": args.seed,
            "device": args.device,
            "results_csv": str(experiment_dir / "results.csv"),
            "aggregate_csv": str(experiment_dir / "aggregate_by_k.csv"),
        },
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
