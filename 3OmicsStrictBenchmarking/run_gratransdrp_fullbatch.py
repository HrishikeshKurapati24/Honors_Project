import argparse
import os
import sys
import time
from typing import Dict, List

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_wrappers import gratransdrp_fullbatch_shared_graph_runner  # noqa: E402
from benchmarking_common import ensure_dir, read_json, write_json  # noqa: E402
from benchmarking_common.results import save_tuning_outputs  # noqa: E402
from benchmarking_common.splits import ensure_protocol_folds  # noqa: E402


DEFAULT_CONFIG = {
    "lr": 1e-4,
    "dropout": 0.5,
    "batch_size": 16,
    "top_k": 10,
}

FULLBATCH_LR_GRID = [5e-5, 1e-4, 5e-4, 1e-3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and tune GraTransDRP full-batch strict training.")
    parser.add_argument("--dataset", default="dataset-2", help="Strict prepared dataset name.")
    parser.add_argument(
        "--protocol",
        default="random",
        choices=["random", "unseen_cells", "unseen_drugs", "unseen_both"],
        help="Protocol to run on.",
    )
    parser.add_argument("--device", default="cuda", help="Runner device, e.g. cuda or cpu.")
    parser.add_argument("--fullbatch-max-epochs", type=int, default=500, help="Maximum epochs for the full-batch run.")
    parser.add_argument("--fullbatch-patience", type=int, default=100, help="Patience for full-batch early stopping.")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1],
        help="Fold ids to run. Default uses fold 1 for a bounded check.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Benchmark seed.")
    parser.add_argument(
        "--results-subdir",
        default="results_gratransdrp_fullbatch",
        help="Results folder under 3OmicsStrictBenchmarking.",
    )
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Create the requested protocol splits if missing.",
    )
    return parser.parse_args()


def _benchmark_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _prepared_dir(dataset_name: str) -> str:
    return os.path.join(_benchmark_dir(), "prepared", dataset_name)


def _split_dir(protocol: str, dataset_name: str) -> str:
    return os.path.join(_benchmark_dir(), "splits", protocol, dataset_name)


def _ensure_requested_splits(dataset_name: str, protocol: str, seed: int) -> str:
    prepared_response_path = os.path.join(_prepared_dir(dataset_name), "response_pairs.csv")
    output_dir = _split_dir(protocol, dataset_name)
    return ensure_protocol_folds(
        response_pairs_path=prepared_response_path,
        output_dir=output_dir,
        protocol=protocol,
        seed=seed,
        n_splits=5,
    )


def _config_from_existing_best(dataset_name: str) -> tuple[Dict, str]:
    best_config_path = os.path.join(
        _benchmark_dir(),
        "results",
        "random",
        dataset_name,
        "GraTransDRP",
        "best_config.json",
    )
    if os.path.isfile(best_config_path):
        payload = read_json(best_config_path)
        return dict(payload.get("config", {})), best_config_path
    return dict(DEFAULT_CONFIG), "default"


def _load_fold_metrics(results_dir: str) -> pd.DataFrame:
    fold_metrics_path = os.path.join(results_dir, "fold_metrics.csv")
    if not os.path.isfile(fold_metrics_path):
        raise FileNotFoundError(f"Missing fold metrics: {fold_metrics_path}")
    return pd.read_csv(fold_metrics_path)


def main() -> None:
    args = _parse_args()
    benchmark_dir = _benchmark_dir()
    prepared_dir = _prepared_dir(args.dataset)
    if not os.path.isdir(prepared_dir):
        raise FileNotFoundError(
            f"Prepared dataset not found: {prepared_dir}. "
            "Run 3OmicsStrictBenchmarking/prepare_data.py first."
        )

    split_dir = _split_dir(args.protocol, args.dataset)
    if args.prepare_splits or not os.path.isdir(split_dir):
        split_dir = _ensure_requested_splits(args.dataset, args.protocol, args.seed)

    config, config_source = _config_from_existing_best(args.dataset)
    results_root = os.path.join(
        benchmark_dir,
        args.results_subdir,
        args.protocol,
        args.dataset,
        f"fullbatch_cap_{args.fullbatch_max_epochs}_patience_{args.fullbatch_patience}",
    )
    ensure_dir(results_root)

    trials = []
    best_trial = None
    for trial_idx, lr in enumerate(FULLBATCH_LR_GRID, start=1):
        candidate = {
            "lr": lr,
            "dropout": float(config.get("dropout", DEFAULT_CONFIG["dropout"])),
            "batch_size": int(config.get("batch_size", DEFAULT_CONFIG["batch_size"])),
            "top_k": int(config.get("top_k", DEFAULT_CONFIG["top_k"])),
        }
        trial_name = f"trial_{trial_idx:02d}"
        trial_results_dir = os.path.join(results_root, "GraTransDRP_fullbatch_tuning", trial_name)
        ensure_dir(trial_results_dir)
        started_at = time.perf_counter()
        summary = gratransdrp_fullbatch_shared_graph_runner.run(
            root_dir=ROOT_DIR,
            prepared_dir=prepared_dir,
            split_dir=split_dir,
            results_dir=trial_results_dir,
            device=args.device,
            seed=args.seed,
            epochs=args.fullbatch_max_epochs,
            patience=args.fullbatch_patience,
            fold_ids=args.folds,
            **candidate,
        )
        elapsed_sec = time.perf_counter() - started_at
        fold_metrics = _load_fold_metrics(trial_results_dir)
        mean_best_val_auc = float(fold_metrics["best_val_auc"].mean())
        mean_best_epoch = float(fold_metrics["best_epoch"].mean()) if "best_epoch" in fold_metrics else 0.0
        mean_epochs_trained = float(fold_metrics["epochs_trained"].mean()) if "epochs_trained" in fold_metrics else float(args.fullbatch_max_epochs)
        trial_payload = {
            "trial": trial_idx,
            "trial_name": trial_name,
            "config": candidate,
            "mean_best_val_auc": mean_best_val_auc,
            "mean_best_epoch": mean_best_epoch,
            "mean_epochs_trained": mean_epochs_trained,
            "elapsed_sec": elapsed_sec,
            "results_dir": trial_results_dir,
            "summary": summary,
        }
        trials.append(trial_payload)
        if best_trial is None or trial_payload["mean_best_val_auc"] > best_trial["mean_best_val_auc"]:
            best_trial = trial_payload

    if best_trial is None:
        raise RuntimeError("GraTransDRP full-batch tuning produced no completed trials")

    tuning_root = os.path.join(results_root, "GraTransDRP_fullbatch_tuning")
    best_config_payload = {
        "config": best_trial["config"],
        "score": best_trial["mean_best_val_auc"],
        "tuned": True,
        "selection_metric": "mean_best_val_auc",
        "max_epochs": args.fullbatch_max_epochs,
        "patience": args.fullbatch_patience,
        "folds": args.folds,
    }
    save_tuning_outputs(
        model_results_dir=tuning_root,
        trials=[
            {
                "trial": row["trial"],
                "trial_name": row["trial_name"],
                "score": row["mean_best_val_auc"],
                "mean_best_epoch": row["mean_best_epoch"],
                "mean_epochs_trained": row["mean_epochs_trained"],
                "elapsed_sec": row["elapsed_sec"],
                **row["config"],
            }
            for row in trials
        ],
        best_config_payload=best_config_payload,
        metadata={
            "mode": "GraTransDRP_fullbatch",
            "selection_metric": "mean_best_val_auc",
            "max_epochs": args.fullbatch_max_epochs,
            "patience": args.fullbatch_patience,
        },
    )

    final_results_dir = os.path.join(results_root, "GraTransDRP_fullbatch")
    ensure_dir(final_results_dir)
    final_payload = {
        "mode": "GraTransDRP_fullbatch",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "device": args.device,
        "folds": args.folds,
        "config_source": config_source,
        "base_config": config,
        "lr_grid": FULLBATCH_LR_GRID,
        "elapsed_sec": best_trial["elapsed_sec"],
        "config": best_trial["config"],
        "summary": best_trial["summary"],
        "results_dir": best_trial["results_dir"],
        "selection_metric": "mean_best_val_auc",
        "tuning": {
            "selection_metric": "mean_best_val_auc",
            "best_trial_name": best_trial["trial_name"],
            "mean_best_val_auc": best_trial["mean_best_val_auc"],
            "mean_best_epoch": best_trial["mean_best_epoch"],
            "mean_epochs_trained": best_trial["mean_epochs_trained"],
            "max_epochs": args.fullbatch_max_epochs,
            "patience": args.fullbatch_patience,
        },
    }
    write_json(os.path.join(final_results_dir, "compare_timing.json"), final_payload)
    write_json(os.path.join(results_root, "run_summary.json"), final_payload)

    print(
        f"[GraTransDRP fullbatch] protocol={args.protocol} dataset={args.dataset} "
        f"fullbatch_cap={args.fullbatch_max_epochs} patience={args.fullbatch_patience} "
        f"folds={args.folds} device={args.device}",
        flush=True,
    )
    print(f"config_source={config_source}", flush=True)
    print(f"fullbatch_mean={best_trial['summary'].get('mean', {})}", flush=True)
    print(f"fullbatch_best_config={best_trial['config']}", flush=True)


if __name__ == "__main__":
    main()
