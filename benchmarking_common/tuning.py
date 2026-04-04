import json
import os
from typing import Dict, List, Tuple

from benchmarking_common.results import load_best_config, load_model_summary, save_tuning_outputs


TUNING_BUDGETS = {
    "SOULCDR": (40, 80),
    "GraphCDR": (60, 120),
    "RedCDR": (60, 120)
}


def should_tune_random(benchmark_name: str, dataset_name: str, model_name: str) -> bool:
    if model_name == "SOULCDR":
        return dataset_name == "dataset-1"

    valid_datasets = {"dataset-1", "dataset-2"}

    if benchmark_name == "3OmicsBenchmarking":
        return dataset_name in valid_datasets and model_name in {"GraphCDR", "RedCDR"}
    return False


def default_fixed_config(benchmark_name: str, dataset_name: str, model_name: str) -> Dict:
    del benchmark_name, dataset_name
    return {}


def tuning_candidates(model_name: str) -> List[Dict]:
    if model_name == "SOULCDR":
        return [
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 512},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 512},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 256},
            {"lr": 1e-3, "hidden_channels": 128, "output_channels": 256, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 512, "output_channels": 64, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 256},
            {"lr": 1e-3, "hidden_channels": 128, "output_channels": 256, "fusion_channels": 256},
        ]
    if model_name == "GraphCDR":
        return [
            {"alpha": 0.2, "beta": 0.2},
            {"alpha": 0.2, "beta": 0.3},
            {"alpha": 0.3, "beta": 0.2},
            {"alpha": 0.3, "beta": 0.3},
            {"alpha": 0.3, "beta": 0.4},
            {"alpha": 0.4, "beta": 0.3},
        ]
    if model_name == "RedCDR":
        return [
            {"lr": 5e-4, "numk": 3, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 5e-4, "numk": 5, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 1e-3, "numk": 5, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 1e-3, "numk": 5, "rd": 0.5, "pd_weight": 1.5},
            {"lr": 1e-3, "numk": 7, "rd": 0.5, "pd_weight": 1.5},
            {"lr": 5e-4, "numk": 7, "rd": 0.5, "pd_weight": 2.0},
        ]
    raise KeyError(f"Unsupported tuning model '{model_name}'")


def random_best_config_path(benchmark_dir: str, dataset_name: str, model_name: str) -> str:
    return os.path.join(benchmark_dir, "results", "random", dataset_name, model_name, "best_config.json")


def load_random_best_config(benchmark_dir: str, dataset_name: str, model_name: str) -> Dict:
    model_results_dir = os.path.join(benchmark_dir, "results", "random", dataset_name, model_name)
    config_path = os.path.join(model_results_dir, "best_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Missing random-split best config for {model_name} on {dataset_name}. "
            f"Expected {config_path}. Run the random benchmark first."
        )
    return load_best_config(model_results_dir)


def _trial_rows_key(candidate: Dict) -> str:
    return json.dumps(candidate, sort_keys=True)


def resolve_random_config(
    runner,
    root_dir: str,
    benchmark_name: str,
    benchmark_dir: str,
    dataset_name: str,
    prepared_dir: str,
    split_dir: str,
    model_name: str,
    model_results_dir: str,
    device: str,
    seed: int,
    enable_tuning: bool = True,
) -> Dict:
    tune = enable_tuning and should_tune_random(benchmark_name, dataset_name, model_name)
    fixed_config = default_fixed_config(benchmark_name, dataset_name, model_name)
    if not tune:
        payload = {
            "model": model_name,
            "benchmark": benchmark_name,
            "dataset": dataset_name,
            "protocol": "random",
            "tuned": False,
            "config": fixed_config,
        }
        save_tuning_outputs(
            model_results_dir=model_results_dir,
            trials=[],
            best_config_payload=payload,
            metadata={"reason": "tuning_skipped_by_policy"},
        )
        return fixed_config

    best_config_path = os.path.join(model_results_dir, "best_config.json")
    if os.path.isfile(best_config_path):
        print(f"> Found existing best config for {model_name} on {dataset_name}. Skipping tuning.")
        payload = load_best_config(model_results_dir)
        return payload.get("config", fixed_config)

    candidates = tuning_candidates(model_name)
    stage_epochs = TUNING_BUDGETS[model_name]
    stage_folds = [[1], [1, 2]]
    survivors: List[Tuple[float, Dict]] = [(float("-inf"), candidate) for candidate in candidates]
    all_trials: List[Dict] = []

    for stage_idx, (fold_ids, epochs) in enumerate(zip(stage_folds, stage_epochs), start=1):
        stage_rows: List[Tuple[float, Dict]] = []
        current_candidates = [candidate for _, candidate in survivors]
        for candidate_idx, candidate in enumerate(current_candidates, start=1):
            trial_dir = os.path.join(model_results_dir, "_tuning", f"stage_{stage_idx}", f"trial_{candidate_idx}")
            summary_path = os.path.join(trial_dir, "summary.json")

            if os.path.isfile(summary_path):
                print(f"> Fold {fold_ids} - Stage {stage_idx} Trial {candidate_idx}/{len(current_candidates)} - Resuming from existing results")
                summary = load_model_summary(trial_dir)
            else:
                summary = runner(
                    root_dir=root_dir,
                    prepared_dir=prepared_dir,
                    split_dir=split_dir,
                    results_dir=trial_dir,
                    device=device,
                    seed=seed,
                    epochs=epochs,
                    fold_ids=fold_ids,
                    **candidate,
                )
            score = float(summary.get("mean", {}).get("auc", 0.0))
            trial_row = {
                "stage": stage_idx,
                "candidate_id": candidate_idx,
                "fold_ids": ",".join(map(str, fold_ids)),
                "epochs": epochs,
                "auc": score,
                "config": _trial_rows_key(candidate),
            }
            all_trials.append(trial_row)
            stage_rows.append((score, candidate))

        stage_rows.sort(key=lambda item: (item[0], _trial_rows_key(item[1])), reverse=True)
        keep_count = 1 if stage_idx == len(stage_epochs) else max(1, (len(stage_rows) + 1) // 2)
        survivors = stage_rows[:keep_count]

    best_score, best_config = survivors[0]
    payload = {
        "model": model_name,
        "benchmark": benchmark_name,
        "dataset": dataset_name,
        "protocol": "random",
        "tuned": True,
        "selection_metric": "val_auc",
        "selected_score": best_score,
        "config": best_config,
    }
    save_tuning_outputs(
        model_results_dir=model_results_dir,
        trials=all_trials,
        best_config_payload=payload,
        metadata={"pilot_folds": stage_folds, "stage_epochs": stage_epochs},
    )
    return best_config
