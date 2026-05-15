import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from flexibility_utils import (
    FINAL_DATASET_DIR,
    FLEXIBILITY_OUTPUTS_DIR,
    FLEXIBILITY_VIEWS_DIR,
    build_dataset_view,
    build_noisy_pathway_tables,
    list_real_omics_stems,
    run_main_flexible,
    summarize_flexible_summary,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4: noisy modality injection study.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--noise-counts", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--noise-mode", type=str, default="permute", choices=["permute", "gaussian", "hybrid"])
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = FLEXIBILITY_OUTPUTS_DIR / "exp4_noisy_modality" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    base_stems = list_real_omics_stems(dataset_root)

    rows = []
    for noise_count in args.noise_counts:
        view_dir = FLEXIBILITY_VIEWS_DIR / "exp4_noisy_modality" / f"{tag}_noise_{noise_count}"
        extra_tables = (
            {}
            if noise_count == 0
            else build_noisy_pathway_tables(
                base_dataset_root=dataset_root,
                count=noise_count,
                seed=args.seed,
                mode=args.noise_mode,
            )
        )
        build_dataset_view(
            view_dir,
            base_dataset_root=dataset_root,
            include_stems=base_stems,
            extra_tables=extra_tables,
        )

        run_name = f"exp4_{tag}_noise_{noise_count}"
        summary = run_main_flexible(
            run_name=run_name,
            dataset_root=view_dir,
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
        )
        metrics = summarize_flexible_summary(summary)
        rows.append(
            {
                "noise_count": noise_count,
                "noise_mode": args.noise_mode,
                "run_name": run_name,
                "run_dir": summary["run_dir"],
                "view_dir": str(view_dir),
                **metrics,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(experiment_dir / "results.csv", index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "exp4_noisy_modality",
            "dataset_root": str(dataset_root),
            "base_stems": base_stems,
            "noise_counts": args.noise_counts,
            "noise_mode": args.noise_mode,
            "epochs": args.epochs,
            "k_fold": args.k_fold,
            "seed": args.seed,
            "device": args.device,
            "results_csv": str(experiment_dir / "results.csv"),
        },
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
