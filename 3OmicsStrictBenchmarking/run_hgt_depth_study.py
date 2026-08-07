import argparse
import os
import sys

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from benchmark_wrappers.fusecdr_hgt_depth_study import (  # noqa: E402
    EXPERIMENT_FOLDS,
    EXPERIMENT_VARIANTS,
    analyze_variant_predictions,
    collect_variant_summary,
    run_variant_training,
    run_propagation_probe,
    save_experiment_manifest,
    summarize_distance_rows,
    summarize_probe_rows,
)


DEFAULT_DATASETS = ("dataset-1", "dataset-2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict FUSE-CDR branch and HGT-depth study."
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=EXPERIMENT_VARIANTS,
        default=list(EXPERIMENT_VARIANTS),
    )
    parser.add_argument("--fold-ids", nargs="+", type=int, choices=EXPERIMENT_FOLDS, default=list(EXPERIMENT_FOLDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--probe-dataset", choices=DEFAULT_DATASETS, default="dataset-2")
    parser.add_argument("--max-probe-pairs", type=int, default=100)
    parser.add_argument("--probe-tolerance", type=float, default=1e-8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-distance-analysis", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.max_probe_pairs < 1:
        raise ValueError("--max-probe-pairs must be positive")
    if args.probe_tolerance < 0:
        raise ValueError("--probe-tolerance cannot be negative")

    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.abspath(
        args.output_root
        or os.path.join(benchmark_dir, "results", "hgt_depth_study_historical_v1")
    )
    os.makedirs(output_root, exist_ok=True)
    datasets = list(dict.fromkeys(args.datasets))
    variants = list(dict.fromkeys(args.variants))
    fold_ids = sorted(set(args.fold_ids))
    if not args.skip_probe and args.probe_dataset not in datasets:
        raise ValueError("--probe-dataset must also be included in --datasets")

    save_experiment_manifest(
        output_root=output_root,
        benchmark_dir=benchmark_dir,
        datasets=datasets,
        variants=variants,
        fold_ids=fold_ids,
        epochs=args.epochs,
        probe_dataset=args.probe_dataset,
        max_probe_pairs=args.max_probe_pairs,
        tolerance=args.probe_tolerance,
    )

    if not args.skip_training:
        for dataset in datasets:
            for variant in variants:
                print(f"> Training strict study: dataset={dataset} variant={variant}", flush=True)
                run_variant_training(
                    root_dir=ROOT_DIR,
                    benchmark_dir=benchmark_dir,
                    output_root=output_root,
                    dataset=dataset,
                    variant=variant,
                    device=args.device,
                    epochs=args.epochs,
                    fold_ids=fold_ids,
                )

    variant_summary = collect_variant_summary(
        output_root=output_root,
        datasets=datasets,
        variants=variants,
    )
    variant_summary.to_csv(os.path.join(output_root, "variant_summary.csv"), index=False)

    if not args.skip_distance_analysis:
        distance_rows = []
        for dataset in datasets:
            for variant in variants:
                print(
                    f"> Analyzing directed distances: dataset={dataset} variant={variant}",
                    flush=True,
                )
                distance_rows.extend(
                    analyze_variant_predictions(
                        root_dir=ROOT_DIR,
                        benchmark_dir=benchmark_dir,
                        output_root=output_root,
                        dataset=dataset,
                        variant=variant,
                        fold_ids=fold_ids,
                    )
                )
        distance_frame = pd.DataFrame(distance_rows)
        distance_frame.to_csv(os.path.join(output_root, "distance_metrics.csv"), index=False)
        summarize_distance_rows(distance_frame).to_csv(
            os.path.join(output_root, "distance_summary.csv"),
            index=False,
        )

    if not args.skip_probe:
        print(f"> Running causal propagation probe on {args.probe_dataset}", flush=True)
        probe_frame = run_propagation_probe(
            root_dir=ROOT_DIR,
            benchmark_dir=benchmark_dir,
            output_root=output_root,
            dataset=args.probe_dataset,
            variants=variants,
            fold_ids=fold_ids,
            device_name=args.device,
            max_pairs_per_distance=args.max_probe_pairs,
            tolerance=args.probe_tolerance,
        )
        probe_frame.to_csv(os.path.join(output_root, "propagation_probe.csv"), index=False)
        summarize_probe_rows(probe_frame).to_csv(
            os.path.join(output_root, "propagation_summary.csv"),
            index=False,
        )

    print(f"> Saved HGT depth study outputs to: {output_root}", flush=True)


if __name__ == "__main__":
    main()
