import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.experiment import run_protocol_benchmarks
from benchmark_wrappers import (
    deepttc_fullbatch_shared_graph_runner,
    gadrp_shared_graph_runner,
    graphcdr_shared_graph_runner,
    graphdrp_fullbatch_shared_graph_runner,
    redcdr_shared_graph_runner,
    fusecdr_minibatch_strict_runner,
    fusecdr_strict_runner,
)


RUNNERS = {
    "FUSECDR": fusecdr_strict_runner.run,
    "FUSECDR_minibatch": fusecdr_minibatch_strict_runner.run,
    "GraphCDR": graphcdr_shared_graph_runner.run,
    "RedCDR": redcdr_shared_graph_runner.run,
    "GADRP": gadrp_shared_graph_runner.run,
    "DeepTTC": deepttc_fullbatch_shared_graph_runner.run,
    "GraphDRP": graphdrp_fullbatch_shared_graph_runner.run,
}

DEFAULT_MODELS = ["FUSECDR", "GraphCDR", "RedCDR", "GADRP", "DeepTTC", "GraphDRP"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["dataset-1"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_name = os.path.basename(benchmark_dir)
    run_protocol_benchmarks(
        root_dir=ROOT_DIR,
        benchmark_name=benchmark_name,
        benchmark_dir=benchmark_dir,
        protocol="unseen_cells",
        runners=RUNNERS,
        datasets=args.datasets,
        models=args.models,
        device=args.device,
        prepare=args.prepare,
        seed=0,
        enable_tuning=False,
    )


if __name__ == "__main__":
    main()
