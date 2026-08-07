import json
import os
from typing import Dict, List, Tuple

from benchmarking_common.results import load_best_config, load_mean_best_val_auc, save_tuning_outputs


TUNING_BUDGETS = {
    "FUSECDR": (40, 80),
    "FUSECDR_minibatch": (40, 80),
    "GraphCDR": (60, 120),
    "RedCDR": (60, 120),
    "GADRP": (40, 80),
    "DeepTTC": (20, 40),
    "GraphDRP": (20, 40),
}


def tuning_stage_epochs(model_name: str, benchmark_name: str) -> Tuple[int, int]:
    if benchmark_name == "3OmicsStrictBenchmarking":
        if model_name == "DeepTTC":
            return (200, 300)
        if model_name == "GraphDRP":
            return (250, 400)
    return TUNING_BUDGETS[model_name]


def should_tune_random(benchmark_name: str, dataset_name: str, model_name: str) -> bool:
    if model_name in {"FUSECDR", "FUSECDR_minibatch"}:
        if benchmark_name == "3OmicsStrictBenchmarking":
            return dataset_name in {"dataset-1", "dataset-2"}
        return dataset_name == "dataset-1"

    valid_datasets = {"dataset-1", "dataset-2"}

    if benchmark_name == "3OmicsBenchmarking":
        return dataset_name in valid_datasets and model_name in {"GraphCDR", "RedCDR", "GADRP"}
    if benchmark_name == "3OmicsStrictBenchmarking":
        return dataset_name in valid_datasets and model_name in {
            "FUSECDR",
            "GraphCDR",
            "RedCDR",
            "GADRP",
            "DeepTTC",
            "GraphDRP",
        }
    if benchmark_name == "DeepTTCBenchmarking":
        return dataset_name == "dataset-1" and model_name in {"FUSECDR", "DeepTTC"}
    if benchmark_name == "GADRPBenchmarking":
        return dataset_name == "dataset-1" and model_name in {"FUSECDR", "GADRP", "RedCDR"}
    if benchmark_name == "GADRPFeatureFairBenchmarking":
        return dataset_name == "dataset-1" and model_name == "GADRP"
    return False


def default_fixed_config(benchmark_name: str, dataset_name: str, model_name: str) -> Dict:
    del dataset_name
    if benchmark_name == "3OmicsStrictBenchmarking" and model_name in {
        "FUSECDR",
        "FUSECDR_minibatch",
        "GraphCDR",
        "RedCDR",
        "GADRP",
        "DeepTTC",
        "GraphDRP",
    }:
        return {"top_k": 10}
    return {}


def tuning_candidates(model_name: str, benchmark_name: str | None = None) -> List[Dict]:
    if model_name in {"FUSECDR", "FUSECDR_minibatch"}:
        candidates = [
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 512},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 512},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 256},
            {"lr": 1e-3, "hidden_channels": 128, "output_channels": 256, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 512, "output_channels": 64, "fusion_channels": 128},
            {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 256},
            {"lr": 1e-3, "hidden_channels": 128, "output_channels": 256, "fusion_channels": 256},
        ]
        if benchmark_name == "3OmicsStrictBenchmarking":
            # Reduced from the full grid using the phase-3 tuning evidence:
            # - architecture shortlist from phase3_hp_summary.txt (top-performing lr=0.001 region)
            # - contrastive shortlist from the phase-3 CL grid logs
            candidates = [
                {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 512},
                {"lr": 1e-3, "hidden_channels": 256, "output_channels": 256, "fusion_channels": 128},
                {"lr": 1e-3, "hidden_channels": 256, "output_channels": 64, "fusion_channels": 256},
            ]
            contrastive_pairs = [
                {"contrastive_weight": 0.005, "temperature": 0.01},
                {"contrastive_weight": 0.01, "temperature": 0.05},
                {"contrastive_weight": 0.005, "temperature": 0.1},
            ]
            expanded = []
            for candidate in candidates:
                for pair in contrastive_pairs:
                    expanded.append(
                        {
                            **candidate,
                            "top_k": 10,
                            **pair,
                        }
                    )
            if model_name == "FUSECDR_minibatch":
                return [
                    {
                        **candidate,
                        "train_drug_batch_size": 64,
                        "eval_drug_batch_size": 0,
                    }
                    for candidate in expanded
                ]
            return expanded
        if model_name == "FUSECDR_minibatch":
            return [
                {
                    **candidate,
                    "train_drug_batch_size": 64,
                    "eval_drug_batch_size": 0,
                }
                for candidate in candidates
            ]
        return candidates
    if model_name == "GraphCDR":
        return [
            {"lr": 5e-4, "alpha": 0.2, "beta": 0.2},
            {"lr": 5e-4, "alpha": 0.3, "beta": 0.3},
            {"lr": 5e-4, "alpha": 0.3, "beta": 0.4},
            {"lr": 1e-3, "alpha": 0.2, "beta": 0.2},
            {"lr": 1e-3, "alpha": 0.3, "beta": 0.3},
            {"lr": 1e-3, "alpha": 0.3, "beta": 0.4},
        ]
    if model_name == "RedCDR":
        if benchmark_name == "3OmicsStrictBenchmarking":
            return [
                {"lr": 5e-4, "rd": 0.25, "pd_weight": 1.0, "dim_feat": 64, "top_k": 10},
                {"lr": 1e-3, "rd": 0.25, "pd_weight": 1.0, "dim_feat": 64, "top_k": 10},
                {"lr": 1e-3, "rd": 0.5, "pd_weight": 1.5, "dim_feat": 64, "top_k": 10},
                {"lr": 5e-4, "rd": 0.5, "pd_weight": 1.5, "dim_feat": 64, "top_k": 10},
                {"lr": 1e-3, "rd": 0.5, "pd_weight": 1.0, "dim_feat": 100, "top_k": 10},
                {"lr": 5e-4, "rd": 0.25, "pd_weight": 1.5, "dim_feat": 100, "top_k": 10},
            ]
        return [
            {"lr": 5e-4, "numk": 3, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 5e-4, "numk": 5, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 1e-3, "numk": 5, "rd": 0.25, "pd_weight": 1.0},
            {"lr": 1e-3, "numk": 5, "rd": 0.5, "pd_weight": 1.5},
            {"lr": 1e-3, "numk": 7, "rd": 0.5, "pd_weight": 1.5},
            {"lr": 5e-4, "numk": 7, "rd": 0.5, "pd_weight": 2.0},
        ]
    if model_name == "GADRP":
        candidates = []
        for lr in (5e-4, 1e-3):
            for irgcn_layers in (3, 5, 7):
                for alpha in (0.1, 0.2):
                    candidates.append(
                        {
                            "lr": lr,
                            "dropout": 0.2,
                            "irgcn_layers": irgcn_layers,
                            "alpha": alpha,
                        }
                    )
        return candidates
    if model_name == "DeepTTC":
        if benchmark_name == "3OmicsStrictBenchmarking":
            return [
                {"lr": 1e-4, "weight_decay": 0.0, "batch_size": 64},
                {"lr": 2e-4, "weight_decay": 0.0, "batch_size": 64},
                {"lr": 5e-4, "weight_decay": 0.0, "batch_size": 64},
                {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 64},
                {"lr": 1e-4, "weight_decay": 1e-5, "batch_size": 128},
                {"lr": 2e-4, "weight_decay": 1e-5, "batch_size": 128},
                {"lr": 5e-4, "weight_decay": 1e-5, "batch_size": 128},
                {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128},
            ]
        return [
            {"lr": 5e-5, "weight_decay": 0.0, "batch_size": 64},
            {"lr": 1e-4, "weight_decay": 0.0, "batch_size": 64},
            {"lr": 2e-4, "weight_decay": 0.0, "batch_size": 64},
            {"lr": 5e-5, "weight_decay": 1e-5, "batch_size": 128},
            {"lr": 1e-4, "weight_decay": 1e-5, "batch_size": 128},
            {"lr": 2e-4, "weight_decay": 1e-5, "batch_size": 128},
        ]
    if model_name == "GraphDRP":
        if benchmark_name == "3OmicsStrictBenchmarking":
            return [
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GCN", "batch_size": 64},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GAT", "batch_size": 64},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GIN", "batch_size": 64},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 64},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GCN", "batch_size": 64},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT", "batch_size": 64},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GIN", "batch_size": 64},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 64},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GCN", "batch_size": 128},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GAT", "batch_size": 128},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GIN", "batch_size": 128},
                {"lr": 2e-4, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 128},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GCN", "batch_size": 128},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT", "batch_size": 128},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GIN", "batch_size": 128},
                {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 128},
            ]
        return [
            {"lr": 5e-4, "dropout": 0.2, "model_type": "GCN", "batch_size": 64},
            {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT", "batch_size": 128},
            {"lr": 5e-4, "dropout": 0.2, "model_type": "GIN", "batch_size": 64},
            {"lr": 5e-4, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 128},
            {"lr": 1e-3, "dropout": 0.2, "model_type": "GCN", "batch_size": 128},
            {"lr": 1e-3, "dropout": 0.2, "model_type": "GAT_GCN", "batch_size": 64},
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


def _config_preview(config: Dict) -> str:
    return json.dumps(config, sort_keys=True)


def tuning_runner_overrides(benchmark_name: str, model_name: str) -> Dict:
    if benchmark_name == "3OmicsStrictBenchmarking" and model_name in {"DeepTTC", "GraphDRP"}:
        return {"patience": 75}
    return {}


def _stale_random_best_config_reason(
    *,
    tune: bool,
    benchmark_name: str,
    model_name: str,
    payload: Dict,
) -> str | None:
    if tune:
        if not payload.get("tuned", False):
            return "legacy non-tuned payload"
        if payload.get("selection_metric") != "mean_best_val_auc":
            return "legacy tuning selection metric"
        if benchmark_name == "3OmicsStrictBenchmarking" and model_name == "GraphDRP":
            config = payload.get("config", {})
            if payload.get("training_regime") != "per_epoch_full_batch" or int(config.get("batch_size", -1)) != 64:
                return "legacy GraphDRP large-batch regime"
    return None


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
    force_rerun_trials = False
    best_config_path = os.path.join(model_results_dir, "best_config.json")
    if os.path.isfile(best_config_path):
        payload = load_best_config(model_results_dir)
        stale_reason = _stale_random_best_config_reason(
            tune=tune,
            benchmark_name=benchmark_name,
            model_name=model_name,
            payload=payload,
        )
        if stale_reason is not None:
            print(
                f"> Existing best config for {model_name} on {dataset_name} uses {stale_reason}. "
                "Re-running random tuning.",
                flush=True,
            )
            force_rerun_trials = True
        else:
            print(f"> Found existing best config for {model_name} on {dataset_name}. Skipping tuning.")
            return payload.get("config", fixed_config)

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

    candidates = tuning_candidates(model_name, benchmark_name=benchmark_name)
    stage_epochs = tuning_stage_epochs(model_name, benchmark_name)
    stage_folds = [[1], [1, 2]]
    survivors: List[Tuple[float, Dict]] = [(float("-inf"), candidate) for candidate in candidates]
    all_trials: List[Dict] = []

    print(
        f"> Tuning start - benchmark={benchmark_name} dataset={dataset_name} model={model_name} "
        f"candidates={len(candidates)} stages={len(stage_epochs)}",
        flush=True,
    )

    for stage_idx, (fold_ids, epochs) in enumerate(zip(stage_folds, stage_epochs), start=1):
        stage_rows: List[Tuple[float, Dict]] = []
        current_candidates = [candidate for _, candidate in survivors]
        runner_overrides = tuning_runner_overrides(benchmark_name, model_name)
        print(
            f"> Tuning stage {stage_idx}/{len(stage_epochs)} - model={model_name} dataset={dataset_name} "
            f"folds={fold_ids} epochs={epochs} candidates={len(current_candidates)}",
            flush=True,
        )
        for candidate_idx, candidate in enumerate(current_candidates, start=1):
            trial_dir = os.path.join(model_results_dir, "_tuning", f"stage_{stage_idx}", f"trial_{candidate_idx}")
            summary_path = os.path.join(trial_dir, "summary.json")
            config_preview = _config_preview(candidate)

            if os.path.isfile(summary_path) and not force_rerun_trials:
                print(
                    f"> Stage {stage_idx} trial {candidate_idx}/{len(current_candidates)} - "
                    f"model={model_name} dataset={dataset_name} folds={fold_ids} resume config={config_preview}",
                    flush=True,
                )
            else:
                print(
                    f"> Stage {stage_idx} trial {candidate_idx}/{len(current_candidates)} - "
                    f"model={model_name} dataset={dataset_name} folds={fold_ids} start config={config_preview}",
                    flush=True,
                )
                runner(
                    root_dir=root_dir,
                    prepared_dir=prepared_dir,
                    split_dir=split_dir,
                    results_dir=trial_dir,
                    device=device,
                    seed=seed,
                    epochs=epochs,
                    fold_ids=fold_ids,
                    **candidate,
                    **runner_overrides,
                )
            score = load_mean_best_val_auc(trial_dir)
            trial_row = {
                "stage": stage_idx,
                "candidate_id": candidate_idx,
                "fold_ids": ",".join(map(str, fold_ids)),
                "epochs": epochs,
                "selection_metric": "mean_best_val_auc",
                "score": score,
                "mean_best_val_auc": score,
                "config": _trial_rows_key(candidate),
            }
            all_trials.append(trial_row)
            stage_rows.append((score, candidate))
            print(
                f"> Stage {stage_idx} trial {candidate_idx}/{len(current_candidates)} complete - "
                f"model={model_name} dataset={dataset_name} mean_best_val_auc={score:.6f}",
                flush=True,
            )

        stage_rows.sort(key=lambda item: (item[0], _trial_rows_key(item[1])), reverse=True)
        keep_count = 1 if stage_idx == len(stage_epochs) else max(1, (len(stage_rows) + 1) // 2)
        survivors = stage_rows[:keep_count]
        print(
            f"> Tuning stage {stage_idx} complete - model={model_name} dataset={dataset_name} "
            f"survivors={len(survivors)} best_mean_best_val_auc={survivors[0][0]:.6f}",
            flush=True,
        )

    best_score, best_config = survivors[0]
    print(
        f"> Tuning complete - benchmark={benchmark_name} dataset={dataset_name} model={model_name} "
        f"best_mean_best_val_auc={best_score:.6f} best_config={_config_preview(best_config)}",
        flush=True,
    )
    payload = {
        "model": model_name,
        "benchmark": benchmark_name,
        "dataset": dataset_name,
        "protocol": "random",
        "tuned": True,
        "selection_metric": "mean_best_val_auc",
        "selected_score": best_score,
        "training_regime": (
            "per_epoch_full_batch"
            if benchmark_name == "3OmicsStrictBenchmarking" and model_name in {"DeepTTC", "GraphDRP"}
            else "benchmark_default"
        ),
        "config": best_config,
    }
    save_tuning_outputs(
        model_results_dir=model_results_dir,
        trials=all_trials,
        best_config_payload=payload,
        metadata={"pilot_folds": stage_folds, "stage_epochs": stage_epochs},
    )
    return best_config
