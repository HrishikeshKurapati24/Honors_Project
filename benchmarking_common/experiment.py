import os
from typing import Callable, Dict, Iterable, List

from benchmarking_common.data_prep import prepare_benchmark_dataset
from benchmarking_common.results import load_best_config, save_best_config
from benchmarking_common.splits import (
    PROTOCOL_RANDOM,
    PROTOCOL_UNSEEN_BOTH,
    ensure_protocol_folds,
)
from benchmarking_common.tuning import load_random_best_config, resolve_random_config


RUNNABLE_PROTOCOL_MODELS = {
    ("3OmicsBenchmarking", PROTOCOL_UNSEEN_BOTH): {"SOULCDR", "GraphCDR", "RedCDR"},
}


def allowed_models_for_protocol(
    benchmark_name: str,
    protocol: str,
    requested_models: Iterable[str],
) -> List[str]:
    allowed = RUNNABLE_PROTOCOL_MODELS.get((benchmark_name, protocol))
    models = list(requested_models)
    if allowed is None:
        return models
    return [model_name for model_name in models if model_name in allowed]


def split_root_for_protocol(benchmark_dir: str, protocol: str, dataset_name: str) -> str:
    return os.path.join(benchmark_dir, "splits", protocol, dataset_name)


def results_root_for_protocol(benchmark_dir: str, protocol: str, dataset_name: str, model_name: str) -> str:
    return os.path.join(benchmark_dir, "results", protocol, dataset_name, model_name)


def run_protocol_benchmarks(
    *,
    root_dir: str,
    benchmark_name: str,
    benchmark_dir: str,
    protocol: str,
    runners: Dict[str, Callable],
    datasets: Iterable[str],
    models: Iterable[str],
    device: str,
    prepare: bool,
    seed: int = 0,
    enable_tuning: bool = True,
) -> None:
    selected_models = allowed_models_for_protocol(benchmark_name, protocol, models)

    for dataset_name in datasets:
        prepared_dir = os.path.join(benchmark_dir, "prepared", dataset_name)
        if prepare or not os.path.isdir(prepared_dir):
            prepare_benchmark_dataset(root_dir, benchmark_name, dataset_name)

        split_dir = ensure_protocol_folds(
            response_pairs_path=os.path.join(prepared_dir, "response_pairs.csv"),
            output_dir=split_root_for_protocol(benchmark_dir, protocol, dataset_name),
            protocol=protocol,
            seed=seed,
            n_splits=5,
        )

        for model_name in selected_models:
            results_dir = results_root_for_protocol(benchmark_dir, protocol, dataset_name, model_name)
            runner = runners[model_name]

            if protocol == PROTOCOL_RANDOM:
                config = resolve_random_config(
                    runner=runner,
                    root_dir=root_dir,
                    benchmark_name=benchmark_name,
                    benchmark_dir=benchmark_dir,
                    dataset_name=dataset_name,
                    prepared_dir=prepared_dir,
                    split_dir=split_dir,
                    model_name=model_name,
                    model_results_dir=results_dir,
                    device=device,
                    seed=seed,
                    enable_tuning=enable_tuning,
                )
            else:
                random_payload = load_random_best_config(benchmark_dir, dataset_name, model_name)
                config = random_payload.get("config", {})
                save_best_config(
                    results_dir,
                    {
                        "model": model_name,
                        "benchmark": benchmark_name,
                        "dataset": dataset_name,
                        "protocol": protocol,
                        "tuned": False,
                        "reused_from_protocol": PROTOCOL_RANDOM,
                        "config": config,
                    },
                )

            runner(
                root_dir=root_dir,
                prepared_dir=prepared_dir,
                split_dir=split_dir,
                results_dir=results_dir,
                device=device,
                seed=seed,
                **config,
            )
