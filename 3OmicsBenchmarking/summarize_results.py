import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmarking_common.results import save_comparison_tables


def main():
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(benchmark_dir, "results")
    for protocol in ["random", "unseen_cells", "unseen_drugs", "unseen_both"]:
        protocol_root = os.path.join(results_root, protocol)
        if not os.path.isdir(protocol_root):
            continue
        output_prefix = os.path.join(results_root, f"{protocol}_3omics")
        save_comparison_tables(protocol_root, output_prefix)


if __name__ == "__main__":
    main()
