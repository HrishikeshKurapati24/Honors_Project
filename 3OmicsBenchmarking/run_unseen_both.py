import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.experiment import run_protocol_benchmarks
from benchmark_wrappers import graphcdr_runner, redcdr_runner, soulcdr_runner


RUNNERS = {
    "SOULCDR": soulcdr_runner.run,
    "GraphCDR": graphcdr_runner.run,
    "RedCDR": redcdr_runner.run,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["dataset-1", "dataset-2"])
    parser.add_argument("--models", nargs="+", default=list(RUNNERS.keys()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_name = os.path.basename(benchmark_dir)
    run_protocol_benchmarks(
        root_dir=ROOT_DIR,
        benchmark_name=benchmark_name,
        benchmark_dir=benchmark_dir,
        protocol="unseen_both",
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
