import argparse
import os
import sys
import time
from typing import Dict, List

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_wrappers import (  # noqa: E402
    deepttc_fullbatch_shared_graph_runner,
    deepttc_largebatch_shared_graph_runner,
    deepttc_shared_graph_runner,
)
from benchmarking_common import ensure_dir, read_json, write_json  # noqa: E402
from benchmarking_common.results import save_tuning_outputs  # noqa: E402
from benchmarking_common.splits import ensure_historical_protocol_folds  # noqa: E402


DEFAULT_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 0.0,
    "batch_size": 64,
    "top_k": 10,
}

FULLBATCH_TUNING_GRID = [
    {"lr": 5e-5, "weight_decay": 0.0},
    {"lr": 1e-4, "weight_decay": 0.0},
    {"lr": 2e-4, "weight_decay": 0.0},
    {"lr": 5e-4, "weight_decay": 0.0},
    {"lr": 1e-3, "weight_decay": 0.0},
    {"lr": 1e-4, "weight_decay": 1e-5},
    {"lr": 2e-4, "weight_decay": 1e-5},
    {"lr": 5e-4, "weight_decay": 1e-5},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DeepTTC original, large-batch, and full-batch training.")
    parser.add_argument("--dataset", default="dataset-2", help="Strict prepared dataset name.")
    parser.add_argument(
        "--protocol",
        default="random",
        choices=["random", "unseen_cells", "unseen_drugs", "unseen_both"],
        help="Protocol to compare on.",
    )
    parser.add_argument("--device", default="cuda", help="Runner device, e.g. cuda or cpu.")
    parser.add_argument(
        "--original-epochs",
        type=int,
        default=80,
        help="Epochs to run for the original mini-batch mode.",
    )
    parser.add_argument(
        "--largebatch-epochs",
        type=int,
        default=80,
        help="Epochs to run for the large-batch mini-batch mode.",
    )
    parser.add_argument(
        "--largebatch-size",
        type=int,
        default=512,
        help="Training batch size for the large-batch mini-batch mode.",
    )
    parser.add_argument(
        "--fullbatch-max-epochs",
        type=int,
        default=500,
        help="Maximum epochs for the tuned full-batch run.",
    )
    parser.add_argument(
        "--fullbatch-patience",
        type=int,
        default=100,
        help="Early-stopping patience for the tuned full-batch run.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Fold ids to run. Use '--folds 1' for a quicker signal.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Benchmark seed.")
    parser.add_argument(
        "--results-subdir",
        default="results_compare_deepttc",
        help="Results folder under 3OmicsStrictBenchmarking.",
    )
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Create the requested protocol splits if missing.",
    )
    parser.add_argument(
        "--reuse-results-root",
        default=None,
        help=(
            "Existing comparison results root to reuse for previously completed modes. "
            "When set, the script can rerun only one mode and recompute the comparison."
        ),
    )
    parser.add_argument(
        "--run-original-only",
        action="store_true",
        help="Run only DeepTTC_original and reuse existing large-batch/full-batch compare payloads.",
    )
    parser.add_argument(
        "--run-largebatch-only",
        action="store_true",
        help="Run only DeepTTC_largebatch and reuse existing original/full-batch compare payloads.",
    )
    parser.add_argument(
        "--run-fullbatch-only",
        action="store_true",
        help="Run only DeepTTC_fullbatch and reuse existing original/large-batch compare payloads.",
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
    return ensure_historical_protocol_folds(
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
        "DeepTTC",
        "best_config.json",
    )
    if os.path.isfile(best_config_path):
        payload = read_json(best_config_path)
        return dict(payload.get("config", {})), best_config_path
    return dict(DEFAULT_CONFIG), "default"


def _run_mode(
    *,
    mode_name: str,
    runner,
    prepared_dir: str,
    split_dir: str,
    results_root: str,
    device: str,
    seed: int,
    epochs: int,
    folds: List[int],
    config: Dict,
) -> Dict:
    results_dir = os.path.join(results_root, mode_name)
    ensure_dir(results_dir)
    started_at = time.perf_counter()
    summary = runner(
        root_dir=ROOT_DIR,
        prepared_dir=prepared_dir,
        split_dir=split_dir,
        results_dir=results_dir,
        device=device,
        seed=seed,
        epochs=epochs,
        fold_ids=folds,
        **config,
    )
    elapsed_sec = time.perf_counter() - started_at
    payload = {
        "mode": mode_name,
        "elapsed_sec": elapsed_sec,
        "config": config,
        "summary": summary,
    }
    write_json(os.path.join(results_dir, "compare_timing.json"), payload)
    return payload


def _load_mode_payload(results_root: str, mode_name: str) -> Dict:
    payload_path = os.path.join(results_root, mode_name, "compare_timing.json")
    if not os.path.isfile(payload_path):
        raise FileNotFoundError(f"Missing existing mode payload: {payload_path}")
    return read_json(payload_path)


def _load_reused_fullbatch_payload(results_root: str, mode_name: str) -> Dict:
    payload = _load_mode_payload(results_root, mode_name)
    tuning_payload = payload.get("tuning", {})
    selection_metric = tuning_payload.get("selection_metric") or payload.get("selection_metric")
    if selection_metric == "mean_best_val_auc" or tuning_payload.get("mean_best_val_auc") is not None:
        return payload
    raise ValueError(
        f"Reused payload for {mode_name} does not record validation-based tuning metadata: "
        f"{os.path.join(results_root, mode_name, 'compare_timing.json')}"
    )


def _fullbatch_candidates(base_config: Dict) -> List[Dict]:
    batch_size = int(base_config.get("batch_size", DEFAULT_CONFIG["batch_size"]))
    top_k = int(base_config.get("top_k", DEFAULT_CONFIG["top_k"]))
    return [
        {
            "lr": candidate["lr"],
            "weight_decay": candidate["weight_decay"],
            "batch_size": batch_size,
            "top_k": top_k,
        }
        for candidate in FULLBATCH_TUNING_GRID
    ]


def _load_fold_metrics(results_dir: str) -> pd.DataFrame:
    fold_metrics_path = os.path.join(results_dir, "fold_metrics.csv")
    if not os.path.isfile(fold_metrics_path):
        raise FileNotFoundError(f"Missing fold metrics: {fold_metrics_path}")
    return pd.read_csv(fold_metrics_path)


def _tune_fullbatch_mode(
    *,
    prepared_dir: str,
    split_dir: str,
    results_root: str,
    device: str,
    seed: int,
    folds: List[int],
    base_config: Dict,
    max_epochs: int,
    patience: int,
) -> Dict:
    tuning_root = os.path.join(results_root, "DeepTTC_fullbatch_tuning")
    ensure_dir(tuning_root)

    trials = []
    best_trial = None
    for trial_idx, candidate in enumerate(_fullbatch_candidates(base_config), start=1):
        trial_name = f"trial_{trial_idx:02d}"
        trial_results_dir = os.path.join(tuning_root, trial_name)
        ensure_dir(trial_results_dir)
        started_at = time.perf_counter()
        summary = deepttc_fullbatch_shared_graph_runner.run(
            root_dir=ROOT_DIR,
            prepared_dir=prepared_dir,
            split_dir=split_dir,
            results_dir=trial_results_dir,
            device=device,
            seed=seed,
            epochs=max_epochs,
            patience=patience,
            fold_ids=folds,
            **candidate,
        )
        elapsed_sec = time.perf_counter() - started_at
        fold_metrics = _load_fold_metrics(trial_results_dir)
        mean_best_val_auc = float(fold_metrics["best_val_auc"].mean())
        mean_best_epoch = float(fold_metrics["best_epoch"].mean()) if "best_epoch" in fold_metrics else 0.0
        mean_epochs_trained = (
            float(fold_metrics["epochs_trained"].mean()) if "epochs_trained" in fold_metrics else float(max_epochs)
        )
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
        raise RuntimeError("DeepTTC full-batch tuning produced no completed trials")

    best_config_payload = {
        "config": best_trial["config"],
        "score": best_trial["mean_best_val_auc"],
        "tuned": True,
        "selection_metric": "mean_best_val_auc",
        "max_epochs": max_epochs,
        "patience": patience,
        "folds": folds,
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
            "mode": "DeepTTC_fullbatch",
            "selection_metric": "mean_best_val_auc",
            "max_epochs": max_epochs,
            "patience": patience,
        },
    )
    return {
        "tuning_root": tuning_root,
        "trials": trials,
        "best_trial": best_trial,
        "best_config_payload": best_config_payload,
    }


def _metric_delta(original: Dict, fullbatch: Dict) -> Dict[str, float]:
    delta = {}
    for metric in ["auc", "aupr", "f1", "acc"]:
        delta[metric] = float(fullbatch["mean"].get(metric, 0.0) - original["mean"].get(metric, 0.0))
    return delta


def main() -> None:
    args = _parse_args()
    mode_only_flags = [
        args.run_original_only,
        args.run_largebatch_only,
        args.run_fullbatch_only,
    ]
    if sum(bool(flag) for flag in mode_only_flags) > 1:
        raise ValueError("Use at most one of --run-original-only, --run-largebatch-only, or --run-fullbatch-only")
    if any(mode_only_flags) and not args.reuse_results_root:
        raise ValueError("Mode-specific reruns require --reuse-results-root so the other modes can be reused")

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
    results_root = args.reuse_results_root or os.path.join(
        benchmark_dir,
        args.results_subdir,
        args.protocol,
        args.dataset,
        (
            f"original_epochs_{args.original_epochs}"
            f"_largebatch_bs_{args.largebatch_size}_epochs_{args.largebatch_epochs}"
            f"_fullbatch_cap_{args.fullbatch_max_epochs}_patience_{args.fullbatch_patience}"
        ),
    )
    ensure_dir(results_root)

    if args.run_largebatch_only or args.run_fullbatch_only:
        original_payload = _load_mode_payload(results_root, "DeepTTC_original")
    else:
        original_payload = _run_mode(
            mode_name="DeepTTC_original",
            runner=deepttc_shared_graph_runner.run,
            prepared_dir=prepared_dir,
            split_dir=split_dir,
            results_root=results_root,
            device=args.device,
            seed=args.seed,
            epochs=args.original_epochs,
            folds=args.folds,
            config=config,
        )

    if args.run_original_only:
        largebatch_payload = _load_mode_payload(results_root, "DeepTTC_largebatch")
        fullbatch_payload = _load_reused_fullbatch_payload(results_root, "DeepTTC_fullbatch")
        fullbatch_tuning = {
            "best_config_payload": fullbatch_payload.get("tuning", {}),
            "best_trial": {
                "trial_name": fullbatch_payload.get("tuning", {}).get("best_trial_name", "reused"),
                "config": fullbatch_payload.get("config", {}),
            },
        }
    else:
        largebatch_config = {
            **config,
            "batch_size": args.largebatch_size,
        }
        if args.run_fullbatch_only:
            largebatch_payload = _load_mode_payload(results_root, "DeepTTC_largebatch")
        else:
            largebatch_payload = _run_mode(
                mode_name="DeepTTC_largebatch",
                runner=deepttc_largebatch_shared_graph_runner.run,
                prepared_dir=prepared_dir,
                split_dir=split_dir,
                results_root=results_root,
                device=args.device,
                seed=args.seed,
                epochs=args.largebatch_epochs,
                folds=args.folds,
                config=largebatch_config,
            )

        if args.run_largebatch_only:
            fullbatch_payload = _load_reused_fullbatch_payload(results_root, "DeepTTC_fullbatch")
            fullbatch_tuning = {
                "best_config_payload": fullbatch_payload.get("tuning", {}),
                "best_trial": {
                    "trial_name": fullbatch_payload.get("tuning", {}).get("best_trial_name", "reused"),
                    "config": fullbatch_payload.get("config", {}),
                },
            }
        else:
            fullbatch_tuning = _tune_fullbatch_mode(
                prepared_dir=prepared_dir,
                split_dir=split_dir,
                results_root=results_root,
                device=args.device,
                seed=args.seed,
                folds=args.folds,
                base_config=config,
                max_epochs=args.fullbatch_max_epochs,
                patience=args.fullbatch_patience,
            )
            best_fullbatch_trial = fullbatch_tuning["best_trial"]
            fullbatch_results_dir = os.path.join(results_root, "DeepTTC_fullbatch")
            ensure_dir(fullbatch_results_dir)
            fullbatch_payload = {
                "mode": "DeepTTC_fullbatch",
                "elapsed_sec": best_fullbatch_trial["elapsed_sec"],
                "config": best_fullbatch_trial["config"],
                "summary": best_fullbatch_trial["summary"],
                "results_dir": best_fullbatch_trial["results_dir"],
                "selection_metric": "mean_best_val_auc",
                "tuning": {
                    "selection_metric": "mean_best_val_auc",
                    "best_trial_name": best_fullbatch_trial["trial_name"],
                    "mean_best_val_auc": best_fullbatch_trial["mean_best_val_auc"],
                    "mean_best_epoch": best_fullbatch_trial["mean_best_epoch"],
                    "mean_epochs_trained": best_fullbatch_trial["mean_epochs_trained"],
                    "max_epochs": args.fullbatch_max_epochs,
                    "patience": args.fullbatch_patience,
                },
            }
            write_json(os.path.join(fullbatch_results_dir, "compare_timing.json"), fullbatch_payload)
    best_fullbatch_trial = fullbatch_tuning["best_trial"]
    largebatch_config = largebatch_payload.get("config", {"batch_size": args.largebatch_size})

    comparison = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "device": args.device,
        "original_epochs": args.original_epochs,
        "largebatch_epochs": args.largebatch_epochs,
        "largebatch_size": args.largebatch_size,
        "fullbatch_max_epochs": args.fullbatch_max_epochs,
        "fullbatch_patience": args.fullbatch_patience,
        "folds": args.folds,
        "config_source": config_source,
        "original_config": config,
        "largebatch_config": largebatch_config,
        "fullbatch_tuning": {
            "grid": FULLBATCH_TUNING_GRID,
            "best_config_payload": fullbatch_tuning["best_config_payload"],
            "best_trial_name": best_fullbatch_trial["trial_name"],
        },
        "original": original_payload,
        "largebatch": largebatch_payload,
        "fullbatch": fullbatch_payload,
        "delta_largebatch_minus_original": _metric_delta(
            original_payload["summary"],
            largebatch_payload["summary"],
        ),
        "delta_fullbatch_minus_original": _metric_delta(
            original_payload["summary"],
            fullbatch_payload["summary"],
        ),
        "delta_fullbatch_minus_largebatch": _metric_delta(
            largebatch_payload["summary"],
            fullbatch_payload["summary"],
        ),
    }
    write_json(os.path.join(results_root, "comparison.json"), comparison)

    print(
        f"[DeepTTC compare] protocol={args.protocol} dataset={args.dataset} "
        f"original_epochs={args.original_epochs} "
        f"largebatch_size={args.largebatch_size} largebatch_epochs={args.largebatch_epochs} "
        f"fullbatch_cap={args.fullbatch_max_epochs} patience={args.fullbatch_patience} "
        f"folds={args.folds} device={args.device} "
        f"original_only={args.run_original_only} "
        f"largebatch_only={args.run_largebatch_only} "
        f"fullbatch_only={args.run_fullbatch_only}",
        flush=True,
    )
    print(f"config_source={config_source}", flush=True)
    print(f"original_mean={original_payload['summary'].get('mean', {})}", flush=True)
    print(f"largebatch_mean={largebatch_payload['summary'].get('mean', {})}", flush=True)
    print(f"fullbatch_mean={fullbatch_payload['summary'].get('mean', {})}", flush=True)
    print(f"fullbatch_best_config={best_fullbatch_trial['config']}", flush=True)
    print(f"delta_largebatch_minus_original={comparison['delta_largebatch_minus_original']}", flush=True)
    print(f"delta_fullbatch_minus_original={comparison['delta_fullbatch_minus_original']}", flush=True)
    print(f"delta_fullbatch_minus_largebatch={comparison['delta_fullbatch_minus_largebatch']}", flush=True)


if __name__ == "__main__":
    main()
