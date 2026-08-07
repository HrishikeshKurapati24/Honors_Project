import argparse
import json
import os
import sys
import time
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_wrappers import (  # noqa: E402
    deepttc_shared_graph_runner,
    graphdrp_shared_graph_runner,
)
from benchmarking_common import ensure_dir, write_json  # noqa: E402
from benchmarking_common.splits import ensure_historical_protocol_folds  # noqa: E402


RUNNERS = {
    "DeepTTC": deepttc_shared_graph_runner.run,
    "GraphDRP": graphdrp_shared_graph_runner.run,
}

MODEL_DEFAULTS: Dict[str, Dict] = {
    "DeepTTC": {
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "top_k": 10,
    },
    "GraphDRP": {
        "batch_size": 64,
        "lr": 5e-4,
        "dropout": 0.2,
        "model_type": "GAT_GCN",
        "top_k": 10,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time 1-epoch strict benchmark smoke runs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DeepTTC", "GraphDRP"],
        choices=sorted(RUNNERS.keys()),
        help="Models to time.",
    )
    parser.add_argument(
        "--dataset",
        default="dataset-1",
        help="Prepared strict dataset name.",
    )
    parser.add_argument(
        "--protocol",
        default="random",
        choices=["random", "unseen_cells", "unseen_drugs", "unseen_both"],
        help="Benchmark protocol split to use.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Runner device. Use cuda in Colab, cpu on laptop if needed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epoch count for the smoke timing.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Single fold to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Benchmark seed.",
    )
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Generate the requested protocol splits if they do not already exist.",
    )
    parser.add_argument(
        "--results-subdir",
        default="results_smoke_timing",
        help="Results folder under 3OmicsStrictBenchmarking.",
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


def _timing_payload(
    *,
    model_name: str,
    dataset_name: str,
    protocol: str,
    device: str,
    epochs: int,
    fold: int,
    elapsed_sec: float,
    summary: Dict,
    config: Dict,
) -> Dict:
    return {
        "model": model_name,
        "dataset": dataset_name,
        "protocol": protocol,
        "device": device,
        "epochs": epochs,
        "fold": fold,
        "elapsed_sec": elapsed_sec,
        "config": config,
        "summary": summary,
    }


def _print_result(payload: Dict) -> None:
    print(
        f"[{payload['model']}] protocol={payload['protocol']} dataset={payload['dataset']} "
        f"fold={payload['fold']} epochs={payload['epochs']} device={payload['device']} "
        f"elapsed_sec={payload['elapsed_sec']:.2f}",
        flush=True,
    )
    print(json.dumps(payload["summary"].get("mean", {}), indent=2), flush=True)


def main() -> None:
    args = _parse_args()
    benchmark_dir = _benchmark_dir()
    prepared_dir = _prepared_dir(args.dataset)

    if not os.path.isdir(prepared_dir):
        raise FileNotFoundError(
            f"Prepared dataset not found: {prepared_dir}. "
            f"Run 3OmicsStrictBenchmarking/prepare_data.py first."
        )

    split_dir = _split_dir(args.protocol, args.dataset)
    if args.prepare_splits or not os.path.isdir(split_dir):
        split_dir = _ensure_requested_splits(args.dataset, args.protocol, args.seed)

    aggregate_rows: List[Dict] = []
    for model_name in args.models:
        runner = RUNNERS[model_name]
        model_config = dict(MODEL_DEFAULTS[model_name])
        results_dir = os.path.join(
            benchmark_dir,
            args.results_subdir,
            args.protocol,
            args.dataset,
            model_name,
        )
        ensure_dir(results_dir)

        started_at = time.perf_counter()
        summary = runner(
            root_dir=ROOT_DIR,
            prepared_dir=prepared_dir,
            split_dir=split_dir,
            results_dir=results_dir,
            device=args.device,
            seed=args.seed,
            epochs=args.epochs,
            fold_ids=[args.fold],
            **model_config,
        )
        elapsed_sec = time.perf_counter() - started_at

        payload = _timing_payload(
            model_name=model_name,
            dataset_name=args.dataset,
            protocol=args.protocol,
            device=args.device,
            epochs=args.epochs,
            fold=args.fold,
            elapsed_sec=elapsed_sec,
            summary=summary,
            config=model_config,
        )
        write_json(os.path.join(results_dir, "timing.json"), payload)
        aggregate_rows.append(payload)
        _print_result(payload)

    aggregate_path = os.path.join(
        benchmark_dir,
        args.results_subdir,
        args.protocol,
        args.dataset,
        "timing_summary.json",
    )
    write_json(
        aggregate_path,
        {
            "dataset": args.dataset,
            "protocol": args.protocol,
            "device": args.device,
            "epochs": args.epochs,
            "fold": args.fold,
            "rows": aggregate_rows,
        },
    )
    print(f"Saved timing summary to {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
