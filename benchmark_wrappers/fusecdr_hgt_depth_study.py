import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from benchmarking_common import read_json, resolve_device, set_seed, write_json
from benchmarking_common.hgt_depth import (
    DRUG_RESPONSE,
    annotate_pair_distances,
    assert_response_edges_exclude_pairs,
    bounded_shortest_distances,
    build_typed_adjacency,
    distance_metric_rows,
)
from benchmarking_common.results import load_best_config
from benchmarking_common.splits import (
    PROTOCOL_RANDOM,
    canonicalize_response_pairs,
    create_historical_random_folds,
)
from benchmark_wrappers import fusecdr_strict_runner
from benchmark_wrappers.common import (
    build_shared_similarity_graphs,
    load_fold_bundle_tables,
)


EXPERIMENT_SEED = 0
EXPERIMENT_FOLDS = (1, 2, 3, 4, 5)
EXPERIMENT_VARIANTS = ("local_only", "hgt_2_only", "hgt_2", "hgt_3")
VARIANT_LOCAL_LAYERS = {
    "local_only": 2,
    "hgt_2_only": 0,
    "hgt_2": 2,
    "hgt_3": 2,
}
VARIANT_GLOBAL_LAYERS = {
    "local_only": 0,
    "hgt_2_only": 2,
    "hgt_2": 2,
    "hgt_3": 3,
}
HETERO_METADATA = (
    ["drug", "cell"],
    [
        ("drug", "responds_to", "cell"),
        ("cell", "similar_to", "cell"),
        ("drug", "similar_to", "drug"),
    ],
)
_VERIFIED_SPLIT_AUDITS: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}


@dataclass
class ProbeContext:
    model: torch.nn.Module
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    baseline_output: Dict[str, object]
    drug_ids: List[str]
    cell_ids: List[str]
    adjacency: List[Tuple[int, ...]]


def variant_results_dir(output_root: str, dataset: str, variant: str) -> str:
    if variant not in VARIANT_GLOBAL_LAYERS:
        raise ValueError(f"Unknown study variant: {variant}")
    return os.path.join(output_root, dataset, variant)


def validate_experiment_scope(
    split_dir: str,
    *,
    fold_ids: Sequence[int],
) -> Dict:
    manifest_path = os.path.join(split_dir, "split_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing strict split manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    expected = {"protocol": "random", "seed": EXPERIMENT_SEED, "n_splits": len(EXPERIMENT_FOLDS)}
    mismatches = {
        key: (manifest.get(key), expected_value)
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Depth study requires the existing strict random folds: {mismatches}")

    invalid_folds = sorted(set(fold_ids) - set(EXPERIMENT_FOLDS))
    if invalid_folds:
        raise ValueError(f"Unsupported folds for the fixed study: {invalid_folds}")
    return manifest


def _labeled_pair_set(frame: pd.DataFrame) -> set[Tuple[str, str, int]]:
    canonical = canonicalize_response_pairs(frame)
    return {
        (str(row.cell_id), str(row.drug_id), int(row.label))
        for row in canonical.itertuples(index=False)
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_study_split(
    *,
    benchmark_dir: str,
    output_root: str,
    dataset: str,
    fold_ids: Sequence[int],
) -> str:
    prepared_pairs_path = os.path.join(benchmark_dir, "prepared", dataset, "response_pairs.csv")
    split_dir = os.path.join(benchmark_dir, "splits", "random", dataset)
    validate_experiment_scope(split_dir, fold_ids=fold_ids)

    audit_key = (os.path.abspath(split_dir), tuple(sorted(fold_ids)))
    audit_payload = _VERIFIED_SPLIT_AUDITS.get(audit_key)
    if audit_payload is None:
        prepared_frame = pd.read_csv(prepared_pairs_path)
        prepared_pairs = _labeled_pair_set(prepared_frame)
        expected_folds = create_historical_random_folds(
            prepared_frame,
            seed=EXPERIMENT_SEED,
            n_splits=len(EXPERIMENT_FOLDS),
        )
        reference_root = os.path.join(benchmark_dir, "results", "random", dataset)
        reference_models = sorted(
            name
            for name in os.listdir(reference_root)
            if os.path.isdir(os.path.join(reference_root, name))
        )
        audit_rows = []
        for fold in fold_ids:
            bundle = load_fold_bundle_tables(split_dir, fold)
            observed = {
                name: _labeled_pair_set(bundle[name])
                for name in ("train", "val", "test")
            }
            expected = {
                name: _labeled_pair_set(expected_folds[fold - 1][name])
                for name in ("train", "val", "test")
            }
            if observed != expected:
                raise ValueError(
                    f"Canonical {dataset} fold {fold} does not match historical_random_v1"
                )
            if (
                observed["train"] & observed["val"]
                or observed["train"] & observed["test"]
                or observed["val"] & observed["test"]
            ):
                raise ValueError(f"Canonical {dataset} fold {fold} contains overlapping splits")
            if observed["train"] | observed["val"] | observed["test"] != prepared_pairs:
                raise ValueError(
                    f"Canonical {dataset} fold {fold} does not partition the prepared pairs"
                )

            prediction_matches = {}
            for model_name in reference_models:
                prediction_path = os.path.join(
                    reference_root,
                    model_name,
                    f"fold_{fold}_predictions.csv",
                )
                if not os.path.isfile(prediction_path):
                    continue
                prediction_matches[model_name] = (
                    _labeled_pair_set(pd.read_csv(prediction_path)) == observed["test"]
                )
            if prediction_matches and not all(prediction_matches.values()):
                failed = sorted(name for name, matched in prediction_matches.items() if not matched)
                raise ValueError(
                    f"Canonical {dataset} fold {fold} disagrees with saved predictions: {failed}"
                )

            fold_dir = os.path.join(split_dir, f"fold_{fold}")
            audit_rows.append(
                {
                    "fold": fold,
                    "train_pairs": len(observed["train"]),
                    "val_pairs": len(observed["val"]),
                    "test_pairs": len(observed["test"]),
                    "saved_prediction_matches": prediction_matches,
                    "sha256": {
                        name: _sha256(os.path.join(fold_dir, f"{name}.csv"))
                        for name in ("train", "val", "test")
                    },
                }
            )
        audit_payload = {
            "dataset": dataset,
            "protocol": PROTOCOL_RANDOM,
            "seed": EXPERIMENT_SEED,
            "n_splits": len(EXPERIMENT_FOLDS),
            "split_generator": "historical_random_v1",
            "validation_seed_policy": "fixed_base_seed_per_outer_fold",
            "prepared_response_pairs": len(prepared_pairs),
            "canonical_split_dir": split_dir,
            "folds": audit_rows,
        }
        _VERIFIED_SPLIT_AUDITS[audit_key] = audit_payload

    write_json(
        os.path.join(output_root, "split_audits", f"{dataset}.json"),
        audit_payload,
    )
    return split_dir


def load_depth_config(benchmark_dir: str, dataset: str) -> Dict:
    best_config_dir = os.path.join(benchmark_dir, "results", "random", dataset, "FUSECDR")
    payload = load_best_config(best_config_dir)
    config = dict(payload.get("config", {}))
    for key in ("num_layers", "num_local_layers", "num_global_layers", "save_checkpoints", "fold_ids"):
        config.pop(key, None)
    return config


def _selected_omics(prepared_dir: str) -> List[str]:
    metadata_path = os.path.join(prepared_dir, "metadata.json")
    metadata = read_json(metadata_path) if os.path.isfile(metadata_path) else {}
    return list(
        metadata.get("omics_for_fusecdr")
        or [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ]
    )


def run_variant_training(
    *,
    root_dir: str,
    benchmark_dir: str,
    output_root: str,
    dataset: str,
    variant: str,
    device: str,
    epochs: int,
    fold_ids: Sequence[int],
) -> Dict:
    if variant not in VARIANT_GLOBAL_LAYERS:
        raise ValueError(f"Variant must be one of {EXPERIMENT_VARIANTS}, found {variant}")
    local_layers = VARIANT_LOCAL_LAYERS[variant]
    global_layers = VARIANT_GLOBAL_LAYERS[variant]
    prepared_dir = os.path.join(benchmark_dir, "prepared", dataset)
    split_dir = ensure_study_split(
        benchmark_dir=benchmark_dir,
        output_root=output_root,
        dataset=dataset,
        fold_ids=fold_ids,
    )
    config = load_depth_config(benchmark_dir, dataset)
    results_dir = variant_results_dir(output_root, dataset, variant)
    return fusecdr_strict_runner.run(
        root_dir=root_dir,
        prepared_dir=prepared_dir,
        split_dir=split_dir,
        results_dir=results_dir,
        device=device,
        seed=EXPERIMENT_SEED,
        epochs=epochs,
        num_layers=2,
        num_local_layers=local_layers,
        num_global_layers=global_layers,
        save_checkpoints=True,
        fold_ids=list(fold_ids),
        **config,
    )


def _aligned_graph_inputs(loaded) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
    cell_ids = sorted({str(item[0]) for item in loaded.data_new})
    drug_ids = sorted({str(item[1]) for item in loaded.data_new})
    cell_features = torch.from_numpy(
        loaded.similarity_feature.loc[cell_ids].to_numpy(dtype=np.float32)
    )
    sample_phys = next(iter(loaded.physicochemical_feature.values()))
    drug_features = torch.from_numpy(
        np.stack(
            [
                loaded.physicochemical_feature.get(
                    drug_id,
                    np.zeros_like(sample_phys, dtype=np.float32),
                )
                for drug_id in drug_ids
            ]
        ).astype(np.float32)
    )
    return cell_ids, drug_ids, cell_features, drug_features


def build_fold_graph(
    *,
    loaded,
    train_pairs: pd.DataFrame,
    top_k: int,
    device: torch.device,
) -> Tuple[List[str], List[str], Dict[Tuple[str, str, str], torch.Tensor]]:
    cell_ids, drug_ids, cell_features, drug_features = _aligned_graph_inputs(loaded)
    edge_index_dict = build_shared_similarity_graphs(
        cell_similarity_features=cell_features,
        drug_similarity_features=drug_features,
        top_k=top_k,
        device=device,
    )
    drug_to_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    cell_to_index = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    positive_pairs = train_pairs[train_pairs["label"] == 1]
    response_edges = [
        (drug_to_index[str(row.drug_id)], cell_to_index[str(row.cell_id)])
        for row in positive_pairs.itertuples(index=False)
        if str(row.drug_id) in drug_to_index and str(row.cell_id) in cell_to_index
    ]
    if response_edges:
        edge_index_dict[DRUG_RESPONSE] = torch.tensor(
            response_edges,
            dtype=torch.long,
            device=device,
        ).T.contiguous()
    else:
        edge_index_dict[DRUG_RESPONSE] = torch.empty((2, 0), dtype=torch.long, device=device)
    return cell_ids, drug_ids, edge_index_dict


def analyze_variant_predictions(
    *,
    root_dir: str,
    benchmark_dir: str,
    output_root: str,
    dataset: str,
    variant: str,
    fold_ids: Sequence[int],
) -> List[Dict[str, float | int | str]]:
    if variant not in VARIANT_GLOBAL_LAYERS:
        raise ValueError(f"Unknown study variant: {variant}")
    local_layers = VARIANT_LOCAL_LAYERS[variant]
    global_layers = VARIANT_GLOBAL_LAYERS[variant]
    prepared_dir = os.path.join(benchmark_dir, "prepared", dataset)
    split_dir = ensure_study_split(
        benchmark_dir=benchmark_dir,
        output_root=output_root,
        dataset=dataset,
        fold_ids=fold_ids,
    )
    config = load_depth_config(benchmark_dir, dataset)
    top_k = int(config.get("top_k", 10))
    module = fusecdr_strict_runner._load_fusecdr_module(root_dir)
    loaded = module.dataload_flexible(prepared_dir, selected_omics=_selected_omics(prepared_dir))
    rows: List[Dict[str, float | int | str]] = []
    results_dir = variant_results_dir(output_root, dataset, variant)

    for fold in fold_ids:
        bundle = load_fold_bundle_tables(split_dir, fold)
        cell_ids, drug_ids, edge_index_dict = build_fold_graph(
            loaded=loaded,
            train_pairs=bundle["train"],
            top_k=top_k,
            device=torch.device("cpu"),
        )
        predictions_path = os.path.join(results_dir, f"fold_{fold}_predictions.csv")
        if not os.path.isfile(predictions_path):
            raise FileNotFoundError(f"Missing variant-study predictions: {predictions_path}")
        predictions = pd.read_csv(predictions_path)
        predictions["drug_id"] = predictions["drug_id"].astype(str)
        predictions["cell_id"] = predictions["cell_id"].astype(str)
        assert_response_edges_exclude_pairs(
            predictions,
            drug_ids=drug_ids,
            cell_ids=cell_ids,
            response_edge_index=edge_index_dict[DRUG_RESPONSE],
        )
        annotated = annotate_pair_distances(
            predictions,
            drug_ids=drug_ids,
            cell_ids=cell_ids,
            edge_index_dict=edge_index_dict,
            max_distance=3,
        )
        annotated.to_csv(
            os.path.join(results_dir, f"fold_{fold}_distance_predictions.csv"),
            index=False,
        )
        rows.extend(
            distance_metric_rows(
                annotated,
                dataset=dataset,
                variant=variant,
                local_layers=local_layers,
                global_layers=global_layers,
                fold=fold,
            )
        )
    return rows


def summarize_distance_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(
            [
                "dataset",
                "variant",
                "local_layers",
                "global_layers",
                "distance_bucket",
            ],
            as_index=False,
        )
        .agg(
            folds=("fold", "nunique"),
            pair_count=("pair_count", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_std=("aupr", "std"),
        )
        .sort_values(["dataset", "variant", "distance_bucket"])
    )


def collect_variant_summary(
    *,
    output_root: str,
    datasets: Sequence[str],
    variants: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for variant in variants:
            results_dir = variant_results_dir(output_root, dataset, variant)
            summary_path = os.path.join(results_dir, "summary.json")
            metrics_path = os.path.join(results_dir, "fold_metrics.csv")
            if not os.path.isfile(summary_path) or not os.path.isfile(metrics_path):
                continue
            summary = read_json(summary_path)
            fold_metrics = pd.read_csv(metrics_path)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "local_layers": VARIANT_LOCAL_LAYERS[variant],
                    "global_layers": VARIANT_GLOBAL_LAYERS[variant],
                    "folds": int(fold_metrics["fold"].nunique()),
                    "auc_mean": float(summary["mean"].get("auc", np.nan)),
                    "auc_std": float(summary["std"].get("auc", np.nan)),
                    "aupr_mean": float(summary["mean"].get("aupr", np.nan)),
                    "aupr_std": float(summary["std"].get("aupr", np.nan)),
                    "parameter_count": int(fold_metrics["parameter_count"].iloc[0]),
                    "best_epoch_mean": float(fold_metrics["best_epoch"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_probe_context(
    *,
    root_dir: str,
    benchmark_dir: str,
    output_root: str,
    dataset: str,
    variant: str,
    fold: int,
    device: torch.device,
) -> ProbeContext:
    prepared_dir = os.path.join(benchmark_dir, "prepared", dataset)
    split_dir = ensure_study_split(
        benchmark_dir=benchmark_dir,
        output_root=output_root,
        dataset=dataset,
        fold_ids=[fold],
    )
    checkpoint_file = fusecdr_strict_runner.checkpoint_path(
        variant_results_dir(output_root, dataset, variant),
        fold,
    )
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Missing variant-study checkpoint: {checkpoint_file}")
    checkpoint = _load_checkpoint(checkpoint_file, device)
    module = fusecdr_strict_runner._load_fusecdr_module(root_dir)
    selected_omics = list(checkpoint["selected_omics"])
    loaded = module.dataload_flexible(prepared_dir, selected_omics=selected_omics)
    bundle = load_fold_bundle_tables(split_dir, fold)
    processed = module.process_flexible(
        loaded=loaded,
        k_folds=len(EXPERIMENT_FOLDS),
        current_fold=fold - 1,
        data_split_seed=EXPERIMENT_SEED + fold * 1000,
        drug_batch_size=0,
        split_tables={
            "train": bundle["train"],
            "val": bundle["val"],
            "test": bundle["test"],
        },
    )
    config = load_depth_config(benchmark_dir, dataset)
    top_k = int(config.get("top_k", 10))
    runtime = fusecdr_strict_runner._prepare_runtime(processed, device, top_k)
    encoder_configs = module.build_encoder_configs(
        omics_tensors=processed.omics_tensors,
        fusion_dim=int(checkpoint["model_config"]["fusion_dim"]),
    )
    model = module.FUSECDR(
        atom_shape=processed.atom_shape,
        encoder_configs=encoder_configs,
        metadata=HETERO_METADATA,
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    drug_batch = next(iter(processed.drug_loader)).to(device)
    batch_drug_indices = drug_batch.drug_idx.long().to(device)
    edge_index_dict = module.get_batch_hetero_graph(
        global_edge_index_dict=runtime["global_edge_index_dict"],
        batch_drug_indices=batch_drug_indices,
        train_edge_subset=runtime["train_edge_tensor"],
        device=device,
    )
    with torch.no_grad():
        baseline_output = model(
            drug_feature=drug_batch.x,
            drug_adj=drug_batch.edge_index,
            ibatch=drug_batch.batch,
            omics_data=runtime["omics_data_device"],
            hetero_graph_edge_index_dict=edge_index_dict,
        )

    cell_ids = sorted({str(item[0]) for item in loaded.data_new})
    drug_ids = sorted({str(item[1]) for item in loaded.data_new})
    adjacency = build_typed_adjacency(
        edge_index_dict,
        num_drugs=len(drug_ids),
        num_cells=len(cell_ids),
    )
    return ProbeContext(
        model=model,
        edge_index_dict=edge_index_dict,
        baseline_output=baseline_output,
        drug_ids=drug_ids,
        cell_ids=cell_ids,
        adjacency=adjacency,
    )


def _cell_probe_pairs(
    context: ProbeContext,
    *,
    test_predictions: pd.DataFrame,
    max_pairs_per_distance: int,
    fold: int,
) -> pd.DataFrame:
    pair_choices = test_predictions.copy()
    pair_choices["drug_id"] = pair_choices["drug_id"].astype(str)
    pair_choices["cell_id"] = pair_choices["cell_id"].astype(str)
    pair_choices = pair_choices.sort_values(
        ["cell_id", "label", "drug_id"],
        ascending=[True, False, True],
    ).drop_duplicates("cell_id", keep="first")
    pair_by_cell = {
        str(row.cell_id): row
        for row in pair_choices.itertuples(index=False)
    }
    drug_to_index = {value: index for index, value in enumerate(context.drug_ids)}
    num_drugs = len(context.drug_ids)
    candidates: Dict[int, List[Dict[str, int | str]]] = {2: [], 3: []}

    for source_cell_index, source_cell_id in enumerate(context.cell_ids):
        distances = bounded_shortest_distances(
            context.adjacency,
            num_drugs + source_cell_index,
            max_distance=3,
        )
        for target_node, distance in distances.items():
            if distance not in candidates or target_node < num_drugs:
                continue
            target_cell_index = target_node - num_drugs
            target_cell_id = context.cell_ids[target_cell_index]
            pair = pair_by_cell.get(target_cell_id)
            if pair is None or str(pair.drug_id) not in drug_to_index:
                continue
            candidates[distance].append(
                {
                    "source_cell_index": source_cell_index,
                    "source_cell_id": source_cell_id,
                    "target_cell_index": target_cell_index,
                    "target_cell_id": target_cell_id,
                    "pair_drug_index": drug_to_index[str(pair.drug_id)],
                    "pair_drug_id": str(pair.drug_id),
                    "label": int(pair.label),
                    "directed_distance": distance,
                }
            )

    sampled = []
    for distance, rows in candidates.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        if len(frame) > max_pairs_per_distance:
            frame = frame.sample(
                n=max_pairs_per_distance,
                random_state=EXPERIMENT_SEED + fold * 100 + distance,
            )
        sampled.append(frame.sort_values(["source_cell_id", "target_cell_id"]))
    if not sampled:
        return pd.DataFrame()
    return pd.concat(sampled, ignore_index=True)


def _control_cell_index(
    context: ProbeContext,
    *,
    target_cell_index: int,
    excluded_cell_index: int,
    depth: int,
    offset: int,
) -> int | None:
    num_drugs = len(context.drug_ids)
    target_node = num_drugs + target_cell_index
    candidates = []
    for cell_index in range(len(context.cell_ids)):
        if cell_index in {excluded_cell_index, target_cell_index}:
            continue
        distances = bounded_shortest_distances(
            context.adjacency,
            num_drugs + cell_index,
            max_distance=depth,
        )
        if target_node not in distances:
            candidates.append(cell_index)
    if not candidates:
        return None
    return candidates[offset % len(candidates)]


def _measure_cell_source_intervention(
    context: ProbeContext,
    *,
    source_cell_index: int,
    pair_drug_index: int,
    target_cell_index: int,
) -> Dict[str, float]:
    model = context.model
    output = context.baseline_output
    input_embeddings = output["input_embeddings"]
    local_embeddings = output["local_embeddings"]
    fused_embeddings = output["node_embeddings"]
    replacement = input_embeddings["cell"].mean(dim=0)
    perturbed_cells = input_embeddings["cell"].clone()
    source_change = torch.linalg.vector_norm(
        perturbed_cells[source_cell_index] - replacement
    )
    perturbed_cells[source_cell_index] = replacement
    perturbed_inputs = {
        "drug": input_embeddings["drug"],
        "cell": perturbed_cells,
    }

    with torch.no_grad():
        if model.use_global_branch:
            global_embeddings = output["global_embeddings"]
            if global_embeddings is None:
                raise RuntimeError("Missing baseline HGT embeddings")
            perturbed_global = model.encode_global(perturbed_inputs, context.edge_index_dict)
            baseline_branch_cell = global_embeddings["cell"][target_cell_index]
            perturbed_branch_cell = perturbed_global["cell"][target_cell_index]
            if model.use_local_branch:
                if local_embeddings is None or model.fusion is None:
                    raise RuntimeError("Missing local embeddings or branch fusion")
                local_cell = local_embeddings["cell"][target_cell_index].unsqueeze(0)
                perturbed_fused_cell, _ = model.fusion(
                    local_cell,
                    perturbed_branch_cell.unsqueeze(0),
                )
            else:
                perturbed_fused_cell = perturbed_branch_cell.unsqueeze(0)
        else:
            if local_embeddings is None:
                raise RuntimeError("Missing baseline GraphSAGE embeddings")
            perturbed_local = model.encode_local(perturbed_inputs, context.edge_index_dict)
            baseline_branch_cell = local_embeddings["cell"][target_cell_index]
            perturbed_branch_cell = perturbed_local["cell"][target_cell_index]
            perturbed_fused_cell = perturbed_branch_cell.unsqueeze(0)
        branch_change = torch.linalg.vector_norm(
            perturbed_branch_cell - baseline_branch_cell
        )

        baseline_fused_cell = fused_embeddings["cell"][target_cell_index].unsqueeze(0)
        baseline_fused_drug = fused_embeddings["drug"][pair_drug_index].unsqueeze(0)
        baseline_logit = model.predictor(
            torch.cat([baseline_fused_drug, baseline_fused_cell], dim=1)
        ).view(())
        perturbed_logit = model.predictor(
            torch.cat([baseline_fused_drug, perturbed_fused_cell], dim=1)
        ).view(())

    return {
        "source_input_change": float(source_change.detach().cpu()),
        "target_branch_change": float(branch_change.detach().cpu()),
        "baseline_logit": float(baseline_logit.detach().cpu()),
        "perturbed_logit": float(perturbed_logit.detach().cpu()),
        "absolute_logit_change": float(torch.abs(baseline_logit - perturbed_logit).detach().cpu()),
        "signed_logit_change": float((baseline_logit - perturbed_logit).detach().cpu()),
    }


def run_propagation_probe(
    *,
    root_dir: str,
    benchmark_dir: str,
    output_root: str,
    dataset: str,
    variants: Sequence[str],
    fold_ids: Sequence[int],
    device_name: str,
    max_pairs_per_distance: int,
    tolerance: float,
) -> pd.DataFrame:
    device = resolve_device(device_name)
    rows: List[Dict[str, float | int | str]] = []
    for variant in variants:
        local_layers = VARIANT_LOCAL_LAYERS[variant]
        global_layers = VARIANT_GLOBAL_LAYERS[variant]
        if global_layers > 0:
            message_depth = global_layers
            probe_branch = "hgt_only" if local_layers == 0 else "hgt_with_local_fixed"
        else:
            message_depth = local_layers
            probe_branch = "graphsage_only"
        for fold in fold_ids:
            set_seed(EXPERIMENT_SEED + fold * 1000)
            context = load_probe_context(
                root_dir=root_dir,
                benchmark_dir=benchmark_dir,
                output_root=output_root,
                dataset=dataset,
                variant=variant,
                fold=fold,
                device=device,
            )
            distance_path = os.path.join(
                output_root,
                dataset,
                variant,
                f"fold_{fold}_distance_predictions.csv",
            )
            annotated = pd.read_csv(distance_path)
            annotated["drug_id"] = annotated["drug_id"].astype(str)
            annotated["cell_id"] = annotated["cell_id"].astype(str)
            sampled = _cell_probe_pairs(
                context,
                test_predictions=annotated,
                max_pairs_per_distance=max_pairs_per_distance,
                fold=fold,
            )
            for offset, pair in enumerate(sampled.itertuples(index=False)):
                source_effect = _measure_cell_source_intervention(
                    context,
                    source_cell_index=int(pair.source_cell_index),
                    pair_drug_index=int(pair.pair_drug_index),
                    target_cell_index=int(pair.target_cell_index),
                )
                control_index = _control_cell_index(
                    context,
                    target_cell_index=int(pair.target_cell_index),
                    excluded_cell_index=int(pair.source_cell_index),
                    depth=message_depth,
                    offset=offset,
                )
                control_effect = (
                    _measure_cell_source_intervention(
                        context,
                        source_cell_index=control_index,
                        pair_drug_index=int(pair.pair_drug_index),
                        target_cell_index=int(pair.target_cell_index),
                    )
                    if control_index is not None
                    else None
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "local_layers": local_layers,
                        "global_layers": global_layers,
                        "probe_branch": probe_branch,
                        "message_depth": message_depth,
                        "fold": fold,
                        "directed_distance": int(pair.directed_distance),
                        "expected_reachable": int(
                            message_depth >= int(pair.directed_distance)
                        ),
                        "source_cell_id": str(pair.source_cell_id),
                        "target_cell_id": str(pair.target_cell_id),
                        "pair_drug_id": str(pair.pair_drug_id),
                        "label": int(pair.label),
                        **source_effect,
                        "source_nonzero": int(source_effect["target_branch_change"] > tolerance),
                        "control_cell_id": (
                            context.cell_ids[control_index] if control_index is not None else ""
                        ),
                        "control_target_branch_change": (
                            control_effect["target_branch_change"] if control_effect else np.nan
                        ),
                        "control_absolute_logit_change": (
                            control_effect["absolute_logit_change"] if control_effect else np.nan
                        ),
                        "control_nonzero": (
                            int(control_effect["target_branch_change"] > tolerance)
                            if control_effect
                            else 0
                        ),
                    }
                )
            del context
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_probe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(
            [
                "dataset",
                "variant",
                "local_layers",
                "global_layers",
                "probe_branch",
                "message_depth",
                "directed_distance",
                "expected_reachable",
            ],
            as_index=False,
        )
        .agg(
            samples=("source_cell_id", "count"),
            source_nonzero_rate=("source_nonzero", "mean"),
            source_branch_change_mean=("target_branch_change", "mean"),
            source_branch_change_median=("target_branch_change", "median"),
            source_logit_change_mean=("absolute_logit_change", "mean"),
            control_nonzero_rate=("control_nonzero", "mean"),
            control_branch_change_mean=("control_target_branch_change", "mean"),
            control_logit_change_mean=("control_absolute_logit_change", "mean"),
        )
        .sort_values(["dataset", "variant", "directed_distance"])
    )


def save_experiment_manifest(
    *,
    output_root: str,
    benchmark_dir: str,
    datasets: Sequence[str],
    variants: Sequence[str],
    fold_ids: Sequence[int],
    epochs: int,
    probe_dataset: str,
    max_probe_pairs: int,
    tolerance: float,
) -> None:
    write_json(
        os.path.join(output_root, "experiment_manifest.json"),
        {
            "experiment": "strict_branch_isolation_hgt_depth_and_propagation",
            "benchmark_dir": os.path.abspath(benchmark_dir),
            "protocol": "random",
            "split_source": "canonical_3omics_strict_random",
            "split_generator": "historical_random_v1",
            "validation_seed_policy": "fixed_base_seed_per_outer_fold",
            "seed": EXPERIMENT_SEED,
            "folds": list(fold_ids),
            "datasets": list(datasets),
            "variants": {
                variant: {
                    "local_layers": VARIANT_LOCAL_LAYERS[variant],
                    "global_layers": VARIANT_GLOBAL_LAYERS[variant],
                }
                for variant in variants
            },
            "epochs": epochs,
            "reuse_dataset_best_configs": True,
            "test_pair_distances": [2, 3],
            "path_definition": "shortest directed path over train-positive response and top-k similarity edges",
            "probe_path_type": "directed cell-similarity paths of exact length 2 or 3",
            "probe_dataset": probe_dataset,
            "max_probe_pairs_per_distance_per_fold": max_probe_pairs,
            "probe_tolerance": tolerance,
        },
    )
