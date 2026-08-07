import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from data_flexible import list_available_omics
from flexibility_utils import (
    FINAL_DATASET_DIR,
    FLEXIBILITY_OUTPUTS_DIR,
    FLEXIBILITY_VIEWS_DIR,
    build_dataset_view,
    build_pathway_shards,
    run_main_flexible,
    summarize_flexible_summary,
    write_json,
)


EXPERIMENT_VERSION = "sharded_v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: pathway sharding flexibility study.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--shard-counts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = FLEXIBILITY_OUTPUTS_DIR / "exp2_pathway_sharding" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for shard_count in args.shard_counts:
        view_dir = FLEXIBILITY_VIEWS_DIR / "exp2_pathway_sharding" / f"{tag}_shards_{shard_count}"
        shard_tables = build_pathway_shards(
            base_dataset_root=dataset_root,
            shard_count=shard_count,
        )
        shard_stems = sorted(shard_tables)
        if len(shard_stems) != shard_count:
            raise RuntimeError(
                f"Expected {shard_count} pathway shards, generated {len(shard_stems)}"
            )

        build_dataset_view(
            view_dir,
            base_dataset_root=dataset_root,
            include_stems=[],
            extra_tables=shard_tables,
        )
        if (view_dir / "pathway.csv").exists():
            raise RuntimeError(
                f"Invalid pathway-sharding view: original pathway.csv is present in {view_dir}"
            )
        missing_shards = [
            stem for stem in shard_stems if not (view_dir / f"{stem}.csv").is_file()
        ]
        if missing_shards:
            raise RuntimeError(
                f"Invalid pathway-sharding view; missing shard files: {missing_shards}"
            )
        discovered_stems = sorted(
            entry["stem"] for entry in list_available_omics(str(view_dir))
        )
        if discovered_stems != shard_stems:
            raise RuntimeError(
                "Invalid pathway-sharding view; expected only "
                f"{shard_stems}, discovered {discovered_stems}"
            )
        print(
            f"Prepared {shard_count} pathway shard(s): "
            + ", ".join(
                f"{stem}[{shard_tables[stem].shape[1]}]" for stem in shard_stems
            )
        )

        # Exact stem selectors avoid the ambiguity between pathway.csv and the
        # broader pathway category.
        run_name = f"exp2_{EXPERIMENT_VERSION}_{tag}_shards_{shard_count}"
        summary = run_main_flexible(
            run_name=run_name,
            dataset_root=view_dir,
            omics=shard_stems,
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
        )
        metrics = summarize_flexible_summary(summary)
        rows.append(
            {
                "shard_count": shard_count,
                "run_name": run_name,
                "run_dir": summary["run_dir"],
                "view_dir": str(view_dir),
                "shard_stems": "|".join(shard_stems),
                "shard_widths": "|".join(
                    str(shard_tables[stem].shape[1]) for stem in shard_stems
                ),
                **metrics,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(experiment_dir / "results.csv", index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "exp2_pathway_sharding",
            "experiment_version": EXPERIMENT_VERSION,
            "dataset_root": str(dataset_root),
            "pathway_source": str(dataset_root / "pathway.csv"),
            "selector_mode": "explicit_shard_stems",
            "shard_counts": args.shard_counts,
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
