import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.data_prep import prepare_benchmark_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["dataset-1", "dataset-2"])
    args = parser.parse_args()

    benchmark_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    root_dir = ROOT_DIR
    for dataset_name in args.datasets:
        metadata = prepare_benchmark_dataset(root_dir, benchmark_name, dataset_name)
        print(f"Prepared {benchmark_name}/{dataset_name}: {metadata}")


if __name__ == "__main__":
    main()
