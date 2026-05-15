import csv
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from benchmarking_common import ensure_dir
from benchmarking_common.metrics import summarize_metric_rows


def write_csv(path: str, rows: List[Dict], fieldnames: List[str] | None = None) -> None:
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_predictions(path: str, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    write_csv(path, rows, fieldnames=["cell_id", "drug_id", "label", "prediction"])


def save_fold_result(model_results_dir: str, fold: int, metrics: Dict, predictions: List[Dict]) -> None:
    ensure_dir(model_results_dir)
    # Save predictions for this specific fold
    save_predictions(os.path.join(model_results_dir, f"fold_{fold}_predictions.csv"), predictions)
    
    # Save metrics for this specific fold in a small JSON
    metrics_path = os.path.join(model_results_dir, f"fold_{fold}_metrics.json")
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)


def load_saved_predictions(model_results_dir: str, fold: int) -> List[Dict]:
    predictions_path = os.path.join(model_results_dir, f"fold_{fold}_predictions.csv")
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Missing saved predictions file: {predictions_path}")
    frame = pd.read_csv(predictions_path)
    return frame.to_dict(orient="records")


def load_completed_folds(model_results_dir: str) -> List[Dict]:
    completed = []
    if not os.path.isdir(model_results_dir):
        return completed
    
    for filename in os.listdir(model_results_dir):
        if filename.startswith("fold_") and filename.endswith("_metrics.json"):
            with open(os.path.join(model_results_dir, filename)) as handle:
                completed.append(json.load(handle))
    return sorted(completed, key=lambda x: x["fold"])


def save_model_outputs(
    model_results_dir: str,
    fold_metrics: List[Dict[str, float]],
    prediction_rows_by_fold: Dict[int, List[Dict]],
    metadata: Dict,
) -> Dict:
    ensure_dir(model_results_dir)
    fold_metrics_path = os.path.join(model_results_dir, "fold_metrics.csv")
    write_csv(fold_metrics_path, fold_metrics)

    for fold, rows in prediction_rows_by_fold.items():
        save_predictions(os.path.join(model_results_dir, f"fold_{fold}_predictions.csv"), rows)

    stats = summarize_metric_rows(fold_metrics)
    summary = {
        "mean": {key: value for key, value in stats["mean"].items() if key != "best_val_auc"},
        "std": {key: value for key, value in stats["std"].items() if key != "best_val_auc"},
        "metadata": metadata,
    }
    with open(os.path.join(model_results_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    summary_rows = []
    for metric in ["auc", "aupr", "f1", "acc"]:
        summary_rows.append(
            {
                "metric": metric.upper(),
                "mean": summary["mean"].get(metric, 0.0),
                "std": summary["std"].get(metric, 0.0),
            }
        )
    write_csv(os.path.join(model_results_dir, "summary.csv"), summary_rows)
    return summary


def save_best_config(model_results_dir: str, payload: Dict) -> Dict:
    ensure_dir(model_results_dir)
    with open(os.path.join(model_results_dir, "best_config.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_best_config(model_results_dir: str) -> Dict:
    with open(os.path.join(model_results_dir, "best_config.json")) as handle:
        return json.load(handle)


def save_tuning_outputs(
    model_results_dir: str,
    trials: List[Dict],
    best_config_payload: Dict,
    metadata: Dict | None = None,
) -> Dict:
    ensure_dir(model_results_dir)
    if trials:
        write_csv(os.path.join(model_results_dir, "tuning_trials.csv"), trials)
    else:
        write_csv(os.path.join(model_results_dir, "tuning_trials.csv"), [])

    tuning_summary = {
        "trial_count": len(trials),
        "best_config": best_config_payload,
        "metadata": metadata or {},
    }
    with open(os.path.join(model_results_dir, "tuning_summary.json"), "w") as handle:
        json.dump(tuning_summary, handle, indent=2)
    save_best_config(model_results_dir, best_config_payload)
    return tuning_summary


def load_model_summary(model_results_dir: str) -> Dict:
    with open(os.path.join(model_results_dir, "summary.json")) as handle:
        return json.load(handle)


def load_fold_metrics(model_results_dir: str) -> pd.DataFrame:
    fold_metrics_path = os.path.join(model_results_dir, "fold_metrics.csv")
    if not os.path.isfile(fold_metrics_path):
        raise FileNotFoundError(f"Missing fold metrics file: {fold_metrics_path}")
    return pd.read_csv(fold_metrics_path)


def load_mean_best_val_auc(model_results_dir: str) -> float:
    fold_metrics = load_fold_metrics(model_results_dir)
    if "best_val_auc" not in fold_metrics.columns:
        raise KeyError(f"fold_metrics.csv in {model_results_dir} does not contain 'best_val_auc'")
    best_val_auc = pd.to_numeric(fold_metrics["best_val_auc"], errors="coerce")
    if best_val_auc.isna().all():
        raise ValueError(f"All best_val_auc entries are missing in {model_results_dir}")
    return float(best_val_auc.mean())


def build_comparison_rows(results_root: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(results_root):
        return rows
    for dataset_name in sorted(os.listdir(results_root)):
        dataset_dir = os.path.join(results_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for model_name in sorted(os.listdir(dataset_dir)):
            model_dir = os.path.join(dataset_dir, model_name)
            summary_path = os.path.join(model_dir, "summary.json")
            if not os.path.isfile(summary_path):
                continue
            summary = load_model_summary(model_dir)
            rows.append(
                {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "AUC_mean": summary["mean"].get("auc", 0.0),
                    "AUC_std": summary["std"].get("auc", 0.0),
                    "AUPR_mean": summary["mean"].get("aupr", 0.0),
                    "AUPR_std": summary["std"].get("aupr", 0.0),
                    "F1_mean": summary["mean"].get("f1", 0.0),
                    "F1_std": summary["std"].get("f1", 0.0),
                    "ACC_mean": summary["mean"].get("acc", 0.0),
                    "ACC_std": summary["std"].get("acc", 0.0),
                }
            )
    return rows


def save_comparison_tables(results_root: str, output_prefix: str) -> None:
    rows = build_comparison_rows(results_root)
    if not rows:
        write_csv(f"{output_prefix}_comparison.csv", [])
        write_csv(f"{output_prefix}_comparison_display.csv", [])
        return
    write_csv(f"{output_prefix}_comparison.csv", rows)

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Model": row["Model"],
                "Dataset": row["Dataset"],
                "AUC": f"{row['AUC_mean']:.4f} +/- {row['AUC_std']:.4f}",
                "AUPR": f"{row['AUPR_mean']:.4f} +/- {row['AUPR_std']:.4f}",
                "F1": f"{row['F1_mean']:.4f} +/- {row['F1_std']:.4f}",
                "ACC": f"{row['ACC_mean']:.4f} +/- {row['ACC_std']:.4f}",
            }
        )
    write_csv(f"{output_prefix}_comparison_display.csv", display_rows)
