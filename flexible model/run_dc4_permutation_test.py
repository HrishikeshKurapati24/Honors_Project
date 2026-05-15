import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from dataset_claim_utils import (
    DATASET_CLAIM_OUTPUTS_DIR,
    DATASET_CLAIM_VIEWS_DIR,
    build_permuted_modality_view,
    ensure_dataset_claim_split_dir,
    list_real_omics_stems,
    result_row_from_summary,
    run_or_reuse_flexible,
    save_incremental_experiment_artifacts,
    slugify,
)
from flexibility_utils import FINAL_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-claim experiment 4: modality permutation test.")
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
    experiment_dir = DATASET_CLAIM_OUTPUTS_DIR / "dc4_permutation_test" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_dir = ensure_dataset_claim_split_dir(
        dataset_root=dataset_root,
        split_dir=args.split_dir,
        seed=args.seed,
        k_fold=args.k_fold,
    )
    real_stems = list_real_omics_stems(dataset_root)

    progress_path = experiment_dir / "results.csv"
    existing = pd.read_csv(progress_path) if progress_path.is_file() else pd.DataFrame()
    completed = set(existing["config_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict(orient="records") if not existing.empty else []

    configs = [{"name": "full_7omics_real", "omics": real_stems, "permuted_stem": ""}] + [
        {"name": f"permute_{stem}", "omics": real_stems, "permuted_stem": stem}
        for stem in real_stems
    ]
    metadata = {
        "experiment": "dc4_permutation_test",
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
        run_name = f"dc4_{tag}_{slugify(config['name'])}"
        if config["permuted_stem"]:
            view_dir = DATASET_CLAIM_VIEWS_DIR / "dc4_permutation_test" / f"{tag}_{slugify(config['permuted_stem'])}"
            dataset_view_root = build_permuted_modality_view(
                output_dir=view_dir,
                dataset_root=dataset_root,
                permute_stem=config["permuted_stem"],
                seed=args.seed,
            )
        else:
            dataset_view_root = dataset_root
        summary = run_or_reuse_flexible(
            run_name=run_name,
            dataset_root=dataset_view_root,
            omics=config["omics"],
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
            split_dir=split_dir,
        )
        row = result_row_from_summary(
            experiment="dc4_permutation_test",
            config_name=config["name"],
            omics=config["omics"],
            run_name=run_name,
            summary=summary,
            extra_fields={
                "permuted_stem": config["permuted_stem"],
                "dataset_view_root": str(dataset_view_root),
            },
        )
        rows.append(row)
        completed.add(config["name"])
        save_incremental_experiment_artifacts(experiment_dir=experiment_dir, rows=rows, metadata=metadata)

    results_df = pd.DataFrame(rows)
    baseline = results_df[results_df["config_name"] == "full_7omics_real"]
    if not baseline.empty:
        baseline = baseline.iloc[0]
        results_df["delta_auc_vs_real"] = results_df["auc"] - float(baseline["auc"])
        results_df["delta_aupr_vs_real"] = results_df["aupr"] - float(baseline["aupr"])
        results_df["delta_f1_vs_real"] = results_df["f1"] - float(baseline["f1"])
        results_df["delta_acc_vs_real"] = results_df["acc"] - float(baseline["acc"])
    results_df.to_csv(experiment_dir / "permutation_deltas.csv", index=False)
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
