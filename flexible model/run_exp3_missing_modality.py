import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from data_flexible import dataload_flexible, list_available_omics, process_flexible
from main_flexible import (
    FUSECDR,
    build_encoder_configs,
    build_hetero_global_graph,
    evaluate_split,
    move_omics_to_device,
    resolve_device,
)
from flexibility_utils import (
    FINAL_DATASET_DIR,
    FLEXIBILITY_OUTPUTS_DIR,
    list_real_omics_stems,
    read_json,
    run_main_flexible,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 3: missing-modality robustness study.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--drop-fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--samples-per-fraction", type=int, default=5)
    parser.add_argument("--base-run-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def _build_entry_map(dataset_root: Path) -> Dict[str, Tuple[str, str]]:
    return {
        entry["stem"]: (entry["category"], entry["subtype"])
        for entry in list_available_omics(str(dataset_root))
    }


def _select_base_stems(dataset_root: Path, base_config: Dict) -> List[str]:
    base_selected = base_config.get("omics")
    real_stems = set(list_real_omics_stems(dataset_root))
    if not base_selected:
        return sorted(real_stems)
    return sorted(stem for stem in base_selected if stem in real_stems)


def _build_scenarios(
    stems: Sequence[str],
    entry_map: Dict[str, Tuple[str, str]],
    drop_fractions: Sequence[float],
    samples_per_fraction: int,
    seed: int,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    scenarios = [{"scenario_id": "full", "drop_type": "none", "drop_stems": [], "keep_stems": list(stems)}]
    seen = {tuple()}

    for fraction in drop_fractions:
        drop_count = max(1, min(len(stems) - 1, round(len(stems) * fraction)))
        unique_candidates = []
        for _ in range(samples_per_fraction * 10):
            candidate = tuple(sorted(rng.choice(stems, size=drop_count, replace=False).tolist()))
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
            if len(unique_candidates) >= samples_per_fraction:
                break
        for idx, drop_stems in enumerate(unique_candidates, start=1):
            drop_stems = list(drop_stems)
            keep_stems = [stem for stem in stems if stem not in drop_stems]
            scenarios.append(
                {
                    "scenario_id": f"random_drop_frac_{fraction:.2f}_sample_{idx}",
                    "drop_type": "random_fraction",
                    "drop_fraction": float(fraction),
                    "drop_stems": drop_stems,
                    "keep_stems": keep_stems,
                }
            )

    stems_by_category: Dict[str, List[str]] = defaultdict(list)
    for stem in stems:
        category, _ = entry_map[stem]
        stems_by_category[category].append(stem)
    for category, category_stems in sorted(stems_by_category.items()):
        if len(category_stems) >= len(stems):
            continue
        key = tuple(sorted(category_stems))
        if key in seen:
            continue
        scenarios.append(
            {
                "scenario_id": f"category_drop_{category}",
                "drop_type": "category",
                "drop_stems": sorted(category_stems),
                "keep_stems": [stem for stem in stems if stem not in category_stems],
            }
        )
    return scenarios


def _filter_omics_for_keep_stems(
    omics_tensors: Dict[str, Dict[str, torch.Tensor]],
    entry_map: Dict[str, Tuple[str, str]],
    keep_stems: Sequence[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    keep_pairs = {entry_map[stem] for stem in keep_stems}
    filtered: Dict[str, Dict[str, torch.Tensor]] = {}
    for category, subtype_map in omics_tensors.items():
        kept_subtypes = {
            subtype: tensor
            for subtype, tensor in subtype_map.items()
            if (category, subtype) in keep_pairs
        }
        if kept_subtypes:
            filtered[category] = kept_subtypes
    return filtered


def _load_split_tables(split_dir: str | None, fold_id: int):
    if not split_dir:
        return None
    fold_dir = Path(split_dir) / f"fold_{fold_id}"
    return {
        "train": pd.read_csv(fold_dir / "train.csv"),
        "val": pd.read_csv(fold_dir / "val.csv"),
        "test": pd.read_csv(fold_dir / "test.csv"),
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = FLEXIBILITY_OUTPUTS_DIR / "exp3_missing_modality" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.base_run_dir:
        base_run_dir = Path(args.base_run_dir).resolve()
    else:
        run_name = f"exp3_{tag}_base_full"
        summary = run_main_flexible(
            run_name=run_name,
            dataset_root=dataset_root,
            omics=list_real_omics_stems(dataset_root),
            epoch=args.epochs,
            k_fold=args.k_fold,
            seed=args.seed,
            device=args.device,
            save_checkpoints=True,
        )
        base_run_dir = Path(summary["run_dir"])

    base_config = read_json(base_run_dir / "config.json")
    base_summary = read_json(base_run_dir / "summary.json")
    dataset_root = Path(base_config.get("dataset_root") or dataset_root).resolve()
    if not base_config.get("save_checkpoints", False):
        raise ValueError(
            f"Base run at {base_run_dir} was created without --save_checkpoints; "
            "experiment 3 needs per-fold checkpoints for re-evaluation under dropped modalities."
        )
    runtime_device = resolve_device(args.device)
    loaded = dataload_flexible(str(dataset_root), selected_omics=base_config.get("omics"))
    entry_map = _build_entry_map(dataset_root)
    base_stems = _select_base_stems(dataset_root, base_config)
    scenarios = _build_scenarios(
        stems=base_stems,
        entry_map=entry_map,
        drop_fractions=args.drop_fractions,
        samples_per_fraction=args.samples_per_fraction,
        seed=args.seed,
    )

    rows = []
    for fold_metric in base_summary["folds"]:
        fold_id = int(fold_metric["fold"])
        split_tables = _load_split_tables(base_config.get("split_dir"), fold_id)
        processed = process_flexible(
            loaded=loaded,
            k_folds=int(base_config["k_fold"]),
            current_fold=max(fold_id - 1, 0),
            data_split_seed=int(base_config["seed"]),
            drug_batch_size=int(base_config["drug_batch_size"]),
            split_tables=split_tables,
        )
        encoder_configs = build_encoder_configs(
            omics_tensors=processed.omics_tensors,
            fusion_dim=int(base_config["fusion_dim"]),
        )
        model = FUSECDR(
            atom_shape=processed.atom_shape,
            encoder_configs=encoder_configs,
            metadata=(
                ["drug", "cell"],
                [
                    ("drug", "responds_to", "cell"),
                    ("cell", "similar_to", "cell"),
                    ("drug", "similar_to", "drug"),
                ],
            ),
            hidden_dim=int(base_config["hidden_channels"]),
            output_dim=int(base_config["output_channels"]),
            fusion_dim=int(base_config["fusion_dim"]),
            dropout=float(base_config["dropout"]),
            num_layers=int(base_config["num_layers"]),
            heads=int(base_config["heads"]),
            drug_num_gnn_layers=int(base_config["drug_num_gnn_layers"]),
        ).to(runtime_device)
        state_dict = torch.load(fold_metric["checkpoint"], map_location=runtime_device)
        model.load_state_dict(state_dict)

        label_pos = processed.label_pos.to(runtime_device)
        test_mask = processed.test_mask.to(runtime_device)
        train_edge_tensor = torch.tensor(processed.train_edge, dtype=torch.long, device=runtime_device)
        global_edge_index_dict = build_hetero_global_graph(
            cell_similarity_tensor=processed.similarity_tensor.to(runtime_device),
            drug_phys_tensor=processed.physicochemical_tensor.to(runtime_device),
            top_k=int(base_config["top_k"]),
            device=runtime_device,
        )
        omics_device = move_omics_to_device(processed.omics_tensors, runtime_device)

        for scenario in scenarios:
            filtered_omics = _filter_omics_for_keep_stems(
                omics_tensors=omics_device,
                entry_map=entry_map,
                keep_stems=scenario["keep_stems"],
            )
            auc, aupr, f1, acc, _, _ = evaluate_split(
                model=model,
                drug_loader=processed.drug_loader,
                omics_data_device=filtered_omics,
                train_edge_tensor=train_edge_tensor,
                eval_mask=test_mask,
                label_pos=label_pos,
                nb_celllines=processed.nb_celllines,
                nb_drugs=processed.nb_drugs,
                global_edge_index_dict=global_edge_index_dict,
                device=runtime_device,
            )
            rows.append(
                {
                    "fold": fold_id,
                    "scenario_id": scenario["scenario_id"],
                    "drop_type": scenario["drop_type"],
                    "drop_count": len(scenario["drop_stems"]),
                    "drop_stems": "|".join(scenario["drop_stems"]),
                    "keep_stems": "|".join(scenario["keep_stems"]),
                    "auc": auc,
                    "aupr": aupr,
                    "f1": f1,
                    "acc": acc,
                }
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(experiment_dir / "scenario_metrics.csv", index=False)
    aggregate_df = (
        results_df.groupby(["scenario_id", "drop_type", "drop_count", "drop_stems", "keep_stems"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_std=("aupr", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
        )
    )
    aggregate_df.to_csv(experiment_dir / "aggregate_metrics.csv", index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "exp3_missing_modality",
            "dataset_root": str(dataset_root),
            "base_run_dir": str(base_run_dir),
            "base_stems": base_stems,
            "drop_fractions": args.drop_fractions,
            "samples_per_fraction": args.samples_per_fraction,
            "scenario_metrics_csv": str(experiment_dir / "scenario_metrics.csv"),
            "aggregate_metrics_csv": str(experiment_dir / "aggregate_metrics.csv"),
        },
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
