import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from dataset_claim_utils import (
    DATASET_CLAIM_OUTPUTS_DIR,
    ensure_dataset_claim_split_dir,
    get_add_one_in_configs,
    result_row_from_summary,
    run_or_reuse_flexible,
    save_incremental_experiment_artifacts,
    slugify,
)
from flexibility_utils import FINAL_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-claim experiment 3: add-one-in analysis from strong compact bases.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = DATASET_CLAIM_OUTPUTS_DIR / "dc3_add_one_in" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_dataset_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )

    progress_path = experiment_dir / "results.csv"
    existing = pd.read_csv(progress_path) if progress_path.is_file() else pd.DataFrame()
    completed = set(existing["config_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict(orient="records") if not existing.empty else []

    configs = get_add_one_in_configs(dataset_root)
    metadata = {
        "experiment": "dc3_add_one_in",
        "dataset_root": str(dataset_root),
        "split_dir": str(split_dir),
        "epochs": args.epochs,
        "k_fold": args.k_fold,
        "seed": args.seed,
        "device": args.device,
        "configs": configs,
    }
    save_incremental_experiment_artifacts(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    for config in configs:
        if config["name"] in completed:
            continue
        run_name = f"dc3_{tag}_{slugify(config['name'])}"
        summary = run_or_reuse_flexible(
            run_name=run_name,
            dataset_root=dataset_root,
            omics=config["omics"],
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
            split_dir=split_dir,
        )
        row = result_row_from_summary(
            experiment="dc3_add_one_in",
            config_name=config["name"],
            omics=config["omics"],
            run_name=run_name,
            summary=summary,
            extra_fields={
                "base_name": config.get("base_name", ""),
                "added_stem": config.get("added_stem", ""),
            },
        )
        rows.append(row)
        completed.add(config["name"])
        save_incremental_experiment_artifacts(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    results_df = pd.DataFrame(rows)
    base_rows = results_df[results_df["added_stem"] == ""].set_index("base_name")
    if not base_rows.empty:
        delta_rows = []
        for row in results_df.itertuples(index=False):
            base_name = getattr(row, "base_name", "")
            if not base_name or base_name not in base_rows.index:
                continue
            base = base_rows.loc[base_name]
            delta_rows.append(
                {
                    **row._asdict(),
                    "delta_auc_vs_base": float(row.auc - base["auc"]),
                    "delta_aupr_vs_base": float(row.aupr - base["aupr"]),
                    "delta_f1_vs_base": float(row.f1 - base["f1"]),
                    "delta_acc_vs_base": float(row.acc - base["acc"]),
                }
            )
        pd.DataFrame(delta_rows).to_csv(experiment_dir / "add_one_in_deltas.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
