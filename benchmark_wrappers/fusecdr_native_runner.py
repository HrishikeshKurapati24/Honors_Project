import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn

from benchmarking_common import read_json, resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import save_model_outputs
from benchmark_wrappers.common import (
    all_scope_entities,
    build_prediction_rows,
    filter_pairs_by_scope,
    load_external_module,
    load_fold_bundle_tables,
    load_prepared_dataset,
    protocol_from_bundle,
    resolve_fold_ids,
    scope_entities_for_split,
    subset_frame,
)


def _load_fusecdr_module(root_dir: str):
    flexible_dir = os.path.join(root_dir, "flexible model")
    return load_external_module(
        module_name="benchmark_fusecdr_native_main",
        file_path=os.path.join(flexible_dir, "main_flexible.py"),
        extra_paths=[flexible_dir],
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=["cell_id", "drug_id", "label"])


def _concat_pairs(*tables: pd.DataFrame) -> pd.DataFrame:
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return _empty_pairs()
    return pd.concat(non_empty, ignore_index=True).drop_duplicates(["cell_id", "drug_id"], keep="first")


@dataclass
class NativeScope:
    cell_ids: List[str]
    drug_ids: List[str]
    omics_tensors: Dict[str, Dict[str, torch.Tensor]]
    similarity_tensor: torch.Tensor
    drug_input_tensor: torch.Tensor
    drug_similarity_tensor: torch.Tensor
    drug_input_dim: int


def _split_omics_selector(selector: str) -> Tuple[str, str]:
    if "_" not in selector:
        return selector, selector
    category, subtype = selector.split("_", 1)
    return category, subtype


def _build_scope(
    dataset: Dict,
    cell_ids: List[str],
    drug_ids: List[str],
    selected_omics: List[str],
    drug_input_table: str,
) -> NativeScope:
    tables = dataset["tables"]
    similarity = subset_frame(tables["similarity"], cell_ids)
    drug_input = subset_frame(tables[drug_input_table], drug_ids)
    if "physicochemical" in tables:
        drug_similarity = subset_frame(tables["physicochemical"], drug_ids)
    else:
        drug_similarity = drug_input

    omics_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for selector in selected_omics:
        category, subtype = _split_omics_selector(selector)
        frame = subset_frame(tables[selector], cell_ids)
        omics_tensors.setdefault(category, {})[subtype] = torch.from_numpy(
            frame.to_numpy(dtype="float32")
        )

    return NativeScope(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        omics_tensors=omics_tensors,
        similarity_tensor=torch.from_numpy(similarity.to_numpy(dtype="float32")),
        drug_input_tensor=torch.from_numpy(drug_input.to_numpy(dtype="float32")),
        drug_similarity_tensor=torch.from_numpy(drug_similarity.to_numpy(dtype="float32")),
        drug_input_dim=int(drug_input.shape[1]),
    )


def _pairs_to_local_tensors(
    pairs: pd.DataFrame,
    cell_ids: List[str],
    drug_ids: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pairs.empty:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_map = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    rows = []
    labels = []
    for row in pairs.itertuples(index=False):
        if row.cell_id not in cell_map or row.drug_id not in drug_map:
            continue
        rows.append([cell_map[row.cell_id], drug_map[row.drug_id]])
        labels.append(float(row.label))
    if not rows:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def _build_edge_dict(module, scope: NativeScope, train_pairs: pd.DataFrame, device: torch.device, top_k: int):
    edge_dict = module.build_hetero_global_graph(
        cell_similarity_tensor=scope.similarity_tensor.to(device),
        drug_phys_tensor=scope.drug_similarity_tensor.to(device),
        top_k=top_k,
        device=device,
    )
    pair_idx, labels = _pairs_to_local_tensors(train_pairs, scope.cell_ids, scope.drug_ids, device)
    positive_idx = pair_idx[labels == 1]
    if positive_idx.numel() == 0:
        responds = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        responds = torch.stack([positive_idx[:, 1], positive_idx[:, 0]], dim=0)
    edge_dict[("drug", "responds_to", "cell")] = responds
    return edge_dict


def _predict_pairs(
    model,
    scope: NativeScope,
    pair_indices: torch.Tensor,
    edge_dict,
    omics_data_device,
    device: torch.device,
) -> torch.Tensor:
    out = model(
        drug_feature=scope.drug_input_tensor.to(device),
        drug_adj=None,
        ibatch=None,
        omics_data=omics_data_device,
        hetero_graph_edge_index_dict=edge_dict,
        drug_indices=pair_indices[:, 1],
        cell_indices=pair_indices[:, 0],
    )
    return torch.sigmoid(out["logits"].view(-1))


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 400,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    fusion_channels: int | None = None,
    hidden_channels: int = 256,
    output_channels: int = 64,
    dropout: float = 0.2,
    num_layers: int = 2,
    heads: int = 4,
    top_k: int = 10,
    contrastive_weight: float = 0.005,
    temperature: float = 0.05,
    warmup_epochs: int = 10,
    max_contrastive_pairs: int = 2048,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    module = _load_fusecdr_module(root_dir)
    runtime_device = resolve_device(device)
    dataset = load_prepared_dataset(prepared_dir)
    metadata_path = os.path.join(prepared_dir, "metadata.json")
    prepared_metadata = read_json(metadata_path) if os.path.isfile(metadata_path) else {}
    selected_omics = prepared_metadata.get("omics_for_fusecdr", ["transcriptomics_expression", "genomics_cnv"])
    drug_input_table = prepared_metadata.get("drug_input_table", "drug_fingerprint")
    fusion_dim = fusion_channels if fusion_channels is not None else hidden_channels

    metadata = (
        ["drug", "cell"],
        [
            ("drug", "responds_to", "cell"),
            ("cell", "similar_to", "cell"),
            ("drug", "similar_to", "drug"),
        ],
    )

    fold_metrics = []
    prediction_rows_by_fold = {}

    for fold in resolve_fold_ids(split_dir, fold_ids):
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = protocol_from_bundle(bundle)
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == "random":
            all_cells, all_drugs = all_scope_entities(bundle)
            train_scope = _build_scope(dataset, all_cells, all_drugs, selected_omics, drug_input_table)
            val_scope = train_scope
            test_scope = train_scope
            train_pairs_scope = train_pairs
            val_pairs_scope = val_pairs
            test_pairs_scope = test_pairs
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            train_scope = _build_scope(dataset, train_cells, train_drugs, selected_omics, drug_input_table)
            val_scope = _build_scope(dataset, val_cells, val_drugs, selected_omics, drug_input_table)
            test_scope = _build_scope(dataset, test_cells, test_drugs, selected_omics, drug_input_table)

            train_pairs_scope = filter_pairs_by_scope(train_pairs, train_cells, train_drugs)
            val_pairs_scope = filter_pairs_by_scope(val_pairs, val_cells, val_drugs)
            test_pairs_scope = filter_pairs_by_scope(test_pairs, test_cells, test_drugs)

        encoder_configs = module.build_encoder_configs(
            omics_tensors=train_scope.omics_tensors,
            fusion_dim=fusion_dim,
        )
        model = module.FUSECDR(
            atom_shape=1,
            encoder_configs=encoder_configs,
            metadata=metadata,
            hidden_dim=hidden_channels,
            output_dim=output_channels,
            fusion_dim=fusion_dim,
            dropout=dropout,
            num_layers=num_layers,
            heads=heads,
            drug_encoder_type="fingerprint",
            drug_input_dim=train_scope.drug_input_dim,
        ).to(runtime_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        contrastive_criterion = module.SupConLoss(temperature=temperature)

        train_pairs_idx, train_labels = _pairs_to_local_tensors(
            train_pairs_scope, train_scope.cell_ids, train_scope.drug_ids, runtime_device
        )
        val_pairs_idx, val_labels = _pairs_to_local_tensors(
            val_pairs_scope, val_scope.cell_ids, val_scope.drug_ids, runtime_device
        )
        test_pairs_idx, test_labels = _pairs_to_local_tensors(
            test_pairs_scope, test_scope.cell_ids, test_scope.drug_ids, runtime_device
        )

        train_omics = module.move_omics_to_device(train_scope.omics_tensors, runtime_device)
        val_omics = module.move_omics_to_device(val_scope.omics_tensors, runtime_device)
        test_omics = module.move_omics_to_device(test_scope.omics_tensors, runtime_device)

        train_edge_dict = _build_edge_dict(module, train_scope, train_pairs_scope, runtime_device, top_k)
        val_edge_dict = _build_edge_dict(module, val_scope, train_pairs_scope, runtime_device, top_k)
        test_edge_dict = _build_edge_dict(module, test_scope, train_pairs_scope, runtime_device, top_k)

        best_val_auc = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            train_out = model(
                drug_feature=train_scope.drug_input_tensor.to(runtime_device),
                drug_adj=None,
                ibatch=None,
                omics_data=train_omics,
                hetero_graph_edge_index_dict=train_edge_dict,
                drug_indices=train_pairs_idx[:, 1],
                cell_indices=train_pairs_idx[:, 0],
            )
            logits = train_out["logits"].view(-1)
            cls_loss = criterion(logits, train_labels)

            cont_loss = torch.tensor(0.0, device=runtime_device)
            if epoch >= warmup_epochs and train_pairs_idx.numel() > 0:
                pair_repr = torch.cat(
                    [
                        train_out["drug_embeddings"][train_pairs_idx[:, 1]],
                        train_out["cell_embeddings"][train_pairs_idx[:, 0]],
                    ],
                    dim=1,
                )
                labels_cl = train_labels.long()
                unique_labels = torch.unique(labels_cl)
                if len(unique_labels) >= 2:
                    counts = torch.stack([(labels_cl == c).sum() for c in unique_labels])
                    if torch.all(counts >= 2):
                        if pair_repr.shape[0] > max_contrastive_pairs:
                            perm = torch.randperm(pair_repr.shape[0], device=runtime_device)[:max_contrastive_pairs]
                            pair_repr = pair_repr[perm]
                            labels_cl = labels_cl[perm]
                        cont_loss = contrastive_criterion(model.projection_head(pair_repr), labels_cl)

            loss = cls_loss + contrastive_weight * cont_loss
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_scores = _predict_pairs(
                    model=model,
                    scope=val_scope,
                    pair_indices=val_pairs_idx,
                    edge_dict=val_edge_dict,
                    omics_data_device=val_omics,
                    device=runtime_device,
                )
            val_metrics = compute_binary_metrics(val_labels, val_scores)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("FUSECDR native runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = _predict_pairs(
                model=model,
                scope=test_scope,
                pair_indices=test_pairs_idx,
                edge_dict=test_edge_dict,
                omics_data_device=test_omics,
                device=runtime_device,
            ).cpu()

        metrics = compute_binary_metrics(test_labels.cpu(), test_scores)
        fold_metrics.append(
            {
                "fold": fold,
                "best_val_auc": float(best_val_auc),
                "auc": metrics["auc"],
                "aupr": metrics["aupr"],
                "f1": metrics["f1"],
                "acc": metrics["acc"],
            }
        )
        prediction_rows_by_fold[fold] = build_prediction_rows(
            test_pairs_scope.reset_index(drop=True),
            test_scores.numpy(),
        )

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "FUSECDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "selected_omics": selected_omics,
            "drug_input": prepared_metadata.get("drug_input", "fingerprint"),
            "drug_input_table": drug_input_table,
            "cell_similarity_source": prepared_metadata.get("cell_similarity_source", "similarity"),
            "drug_similarity_source": prepared_metadata.get("drug_similarity_source", drug_input_table),
            "config": {
                "lr": lr,
                "fusion_channels": fusion_dim,
                "hidden_channels": hidden_channels,
                "output_channels": output_channels,
            },
        },
    )
