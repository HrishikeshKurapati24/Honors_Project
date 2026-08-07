import argparse
import os
import sys
import time
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_wrappers import graphdrp_largebatch_shared_graph_runner  # noqa: E402
from benchmarking_common import ensure_dir, read_json, write_json  # noqa: E402
from benchmarking_common.splits import ensure_historical_protocol_folds  # noqa: E402


DEFAULT_CONFIG = {
    "lr": 5e-4,
    "dropout": 0.2,
    "model_type": "GAT_GCN",
    "batch_size": 768,
    "top_k": 10,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphDRP with a larger mini-batch size in 3OmicsStrictBenchmarking.")
    parser.add_argument("--dataset", default="dataset-2", help="Strict prepared dataset name.")
    parser.add_argument(
        "--protocol",
        default="random",
        choices=["random", "unseen_cells", "unseen_drugs", "unseen_both"],
        help="Protocol to run on.",
    )
    parser.add_argument("--device", default="cuda", help="Runner device, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=120, help="Epochs to run.")
    parser.add_argument("--batch-size", type=int, default=768, help="Large pair batch size to use.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate. Defaults to the existing random best config or fallback.")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout. Defaults to the existing random best config or fallback.")
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["GCN", "GAT", "GIN", "GAT_GCN"],
        help="Override GraphDRP drug encoder type. Defaults to the existing random best config or fallback.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override shared-graph top-k. Defaults to the existing random best config or fallback.")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1],
        help="Fold ids to run. Default uses fold 1.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Benchmark seed.")
    parser.add_argument(
        "--results-subdir",
        default="results_graphdrp_largebatch",
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
        "GraphDRP",
        "best_config.json",
    )
    if os.path.isfile(best_config_path):
        payload = read_json(best_config_path)
        return dict(payload.get("config", {})), best_config_path
    return dict(DEFAULT_CONFIG), "default"


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

    base_config, config_source = _config_from_existing_best(args.dataset)
    config = {
        "lr": args.lr if args.lr is not None else float(base_config.get("lr", DEFAULT_CONFIG["lr"])),
        "dropout": args.dropout if args.dropout is not None else float(base_config.get("dropout", DEFAULT_CONFIG["dropout"])),
        "model_type": args.model_type if args.model_type is not None else str(base_config.get("model_type", DEFAULT_CONFIG["model_type"])),
        "batch_size": int(args.batch_size),
        "top_k": args.top_k if args.top_k is not None else int(base_config.get("top_k", DEFAULT_CONFIG["top_k"])),
    }

    results_dir = os.path.join(
        benchmark_dir,
        args.results_subdir,
        args.protocol,
        args.dataset,
        f"bs_{config['batch_size']}_epochs_{args.epochs}",
    )
    ensure_dir(results_dir)

    started_at = time.perf_counter()
    summary = graphdrp_largebatch_shared_graph_runner.run(
        root_dir=ROOT_DIR,
        prepared_dir=prepared_dir,
        split_dir=split_dir,
        results_dir=results_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        fold_ids=args.folds,
        **config,
    )
    elapsed_sec = time.perf_counter() - started_at

    payload = {
        "mode": "GraphDRP_largebatch",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "device": args.device,
        "epochs": args.epochs,
        "folds": args.folds,
        "elapsed_sec": elapsed_sec,
        "config_source": config_source,
        "config": config,
        "summary": summary,
    }
    write_json(os.path.join(results_dir, "run_summary.json"), payload)

    print(
        f"[GraphDRP_largebatch] protocol={args.protocol} dataset={args.dataset} "
        f"epochs={args.epochs} folds={args.folds} batch_size={config['batch_size']} "
        f"device={args.device} elapsed_sec={elapsed_sec:.2f}",
        flush=True,
    )
    print(f"config_source={config_source}", flush=True)
    print(f"config={config}", flush=True)
    print(f"mean={summary.get('mean', {})}", flush=True)


if __name__ == "__main__":
    main()
