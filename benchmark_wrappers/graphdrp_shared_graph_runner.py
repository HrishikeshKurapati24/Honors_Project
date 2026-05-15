import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from benchmarking_common import resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import (
    load_completed_folds,
    load_saved_predictions,
    save_fold_result,
    save_model_outputs,
)
from benchmarking_common.strict_contract import (
    validate_strict_model_contract,
    validate_strict_prepared_metadata,
    validate_strict_response_edge_index,
)
from benchmark_wrappers.common import (
    build_prediction_rows,
    build_pyg_graphs,
    build_strict_response_edge_index,
    build_shared_similarity_graphs,
    load_external_module,
    load_fold_bundle_tables,
    load_hkl_features,
    load_prepared_dataset,
    protocol_from_bundle,
    resolve_fold_ids,
    scope_entities_for_split,
)

_SCOPE_CACHE: Dict[Tuple[str, int, Tuple[str, ...], Tuple[str, ...]], Dict] = {}


def _load_graphdrp_module(root_dir: str):
    model_dir = os.path.join(root_dir, "benchmark models", "GraphDRP-master", "benchmark implementation")
    return load_external_module(
        module_name="benchmark_graphdrp_shared_graph_model",
        file_path=os.path.join(model_dir, "model_shared_graph.py"),
        extra_paths=[model_dir],
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=["cell_id", "drug_id", "label"])


def _concat_pairs(*tables: pd.DataFrame) -> pd.DataFrame:
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return _empty_pairs()
    return pd.concat(non_empty, ignore_index=True).drop_duplicates(["cell_id", "drug_id"], keep="first")


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
    indices = []
    labels = []
    for row in pairs.itertuples(index=False):
        cell_idx = cell_map.get(row.cell_id)
        drug_idx = drug_map.get(row.drug_id)
        if cell_idx is None or drug_idx is None:
            continue
        indices.append([cell_idx, drug_idx])
        labels.append(float(row.label))
    if not indices:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    return (
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
    )


def _build_scope(dataset: Dict, scope_pairs: pd.DataFrame, train_pairs: pd.DataFrame, device: torch.device, top_k: int) -> Dict:
    cell_ids = sorted(scope_pairs["cell_id"].astype(str).unique().tolist())
    drug_ids = sorted(scope_pairs["drug_id"].astype(str).unique().tolist())
    train_pair_key = tuple(
        (str(row.cell_id), str(row.drug_id))
        for row in train_pairs[["cell_id", "drug_id"]].itertuples(index=False)
    )
    cache_key = (dataset["prepared_dir"], int(top_k), tuple(cell_ids), tuple(drug_ids), train_pair_key)
    cached = _SCOPE_CACHE.get(cache_key)
    if cached is None:
        mutation_df = dataset["tables"]["genomics_mutation"].loc[cell_ids]
        expression_df = dataset["tables"]["transcriptomics_expression"].loc[cell_ids]
        methylation_df = dataset["tables"]["epigenomics_methylation"].loc[cell_ids]
        similarity_df = dataset["tables"]["similarity"].loc[cell_ids]
        physicochemical_df = dataset["tables"]["physicochemical"].loc[drug_ids]

        graph_dir = os.path.join(dataset["prepared_dir"], "drug_graph_feat")
        graph_features = load_hkl_features(graph_dir, drug_ids)
        graphs = build_pyg_graphs(drug_ids, graph_features)
        graph_batch = Batch.from_data_list(graphs)

        cell_features = torch.from_numpy(
            np.concatenate(
                [
                    mutation_df.to_numpy(dtype=np.float32),
                    expression_df.to_numpy(dtype=np.float32),
                    methylation_df.to_numpy(dtype=np.float32),
                ],
                axis=1,
            )
        )
        edge_dict = build_shared_similarity_graphs(
            cell_similarity_features=similarity_df.to_numpy(dtype=np.float32),
            drug_similarity_features=physicochemical_df.to_numpy(dtype=np.float32),
            top_k=top_k,
            device=torch.device("cpu"),
        )
        response_edge_index = build_strict_response_edge_index(
            cell_ids=cell_ids,
            drug_ids=drug_ids,
            train_pairs=_concat_pairs(train_pairs),
            device=torch.device("cpu"),
        )
        validate_strict_response_edge_index(
            cell_ids=cell_ids,
            drug_ids=drug_ids,
            train_pairs=_concat_pairs(train_pairs),
            response_edge_index=response_edge_index,
        )
        cached = {
            "cell_ids": cell_ids,
            "drug_ids": drug_ids,
            "cell_features": cell_features,
            "graph_batch": graph_batch,
            "cell_edge_index": edge_dict[("cell", "similar_to", "cell")],
            "drug_edge_index": edge_dict[("drug", "similar_to", "drug")],
            "response_edge_index": response_edge_index,
        }
        _SCOPE_CACHE[cache_key] = cached

    return {
        "cell_ids": cached["cell_ids"],
        "drug_ids": cached["drug_ids"],
        "cell_features": cached["cell_features"].to(device),
        "graph_batch": cached["graph_batch"].to(device),
        "cell_edge_index": cached["cell_edge_index"].to(device),
        "drug_edge_index": cached["drug_edge_index"].to(device),
        "response_edge_index": cached["response_edge_index"].to(device),
    }


def _predict(model, scope: Dict, pair_indices: torch.Tensor) -> torch.Tensor:
    logits = _predict_logits(model, scope, pair_indices)
    return torch.sigmoid(logits)


def _predict_logits(model, scope: Dict, pair_indices: torch.Tensor) -> torch.Tensor:
    drug_embeddings = model.encode_drugs(scope["graph_batch"], scope["drug_edge_index"])
    cell_embeddings = model.encode_cells(scope["cell_features"], scope["cell_edge_index"])
    cell_embeddings, drug_embeddings = model.refine_with_response_edges(
        cell_embeddings,
        drug_embeddings,
        scope["response_edge_index"],
    )
    return model.predict_pair_logits(drug_embeddings, cell_embeddings, pair_indices)


def _predict_scores_chunked(
    model,
    scope: Dict,
    pair_indices: torch.Tensor,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    if pair_indices.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=pair_indices.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        drug_embeddings = model.encode_drugs(scope["graph_batch"], scope["drug_edge_index"])
        cell_embeddings = model.encode_cells(scope["cell_features"], scope["cell_edge_index"])
        cell_embeddings, drug_embeddings = model.refine_with_response_edges(
            cell_embeddings,
            drug_embeddings,
            scope["response_edge_index"],
        )
        scores = []
        for start in range(0, pair_indices.shape[0], batch_size):
            logits = model.predict_pair_logits(drug_embeddings, cell_embeddings, pair_indices[start : start + batch_size])
            scores.append(torch.sigmoid(logits.float()))
    return torch.cat(scores, dim=0)


def _train_epoch(
    model,
    scope: Dict,
    pair_indices: torch.Tensor,
    labels: torch.Tensor,
    optimizer,
    scaler,
    criterion,
    batch_size: int,
    use_amp: bool,
) -> None:
    if pair_indices.numel() == 0:
        return
    order = torch.randperm(pair_indices.shape[0], device=pair_indices.device)
    for start in range(0, pair_indices.shape[0], batch_size):
        subset = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = _predict_logits(model, scope, pair_indices[subset])
            loss = criterion(logits, labels[subset])
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 120,
    lr: float = 5e-4,
    dropout: float = 0.2,
    model_type: str = "GAT_GCN",
    batch_size: int = 64,
    top_k: int = 10,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)
    use_amp = runtime_device.type == "cuda"
    module = _load_graphdrp_module(root_dir)
    dataset = load_prepared_dataset(prepared_dir)
    validate_strict_prepared_metadata(prepared_dir, dataset["metadata"])
    validate_strict_model_contract(
        "GraphDRP",
        predictive_inputs=(
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ),
        graph_inputs=("similarity.csv", "physicochemical.csv", "train_pairs"),
    )

    fold_metrics = []
    prediction_rows_by_fold = {}
    completed_folds = {int(row["fold"]): row for row in load_completed_folds(results_dir)}

    for fold in resolve_fold_ids(split_dir, fold_ids):
        if fold in completed_folds:
            print(f"> Fold {fold} already complete. Reusing saved outputs.", flush=True)
            fold_metrics.append(completed_folds[fold])
            prediction_rows_by_fold[fold] = load_saved_predictions(results_dir, fold)
            continue
        set_seed(seed + fold * 1000)
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = protocol_from_bundle(bundle)
        train_pairs = bundle["train"].reset_index(drop=True)
        val_pairs = bundle["val"].reset_index(drop=True)
        test_pairs = bundle["test"].reset_index(drop=True)

        if protocol == "random":
            scope_pairs = dataset["response_pairs"]
            train_scope = _build_scope(dataset, scope_pairs, train_pairs, runtime_device, top_k)
            val_scope = train_scope
            test_scope = train_scope
        else:
            train_scope = _build_scope(dataset, _concat_pairs(train_pairs), train_pairs, runtime_device, top_k)
            val_scope = _build_scope(dataset, _concat_pairs(train_pairs, val_pairs), train_pairs, runtime_device, top_k)
            test_scope = _build_scope(dataset, _concat_pairs(train_pairs, test_pairs), train_pairs, runtime_device, top_k)

        train_indices, train_labels = _pairs_to_local_tensors(
            train_pairs,
            train_scope["cell_ids"],
            train_scope["drug_ids"],
            runtime_device,
        )
        val_indices, val_labels = _pairs_to_local_tensors(
            val_pairs,
            val_scope["cell_ids"],
            val_scope["drug_ids"],
            runtime_device,
        )
        test_indices, test_labels = _pairs_to_local_tensors(
            test_pairs,
            test_scope["cell_ids"],
            test_scope["drug_ids"],
            runtime_device,
        )

        model = module.get_model(
            model_type=model_type,
            atom_dim=int(train_scope["graph_batch"].x.shape[1]),
            num_features_xt=train_scope["cell_features"].shape[1],
            dropout=dropout,
        ).to(runtime_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_val_auc = -1.0
        best_state = None
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            model.train()
            _train_epoch(
                model=model,
                scope=train_scope,
                pair_indices=train_indices,
                labels=train_labels,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                batch_size=batch_size,
                use_amp=use_amp,
            )

            model.eval()
            with torch.no_grad():
                val_scores = _predict_scores_chunked(model, val_scope, val_indices, batch_size=batch_size, use_amp=use_amp)
                val_metrics = compute_binary_metrics(val_labels, val_scores)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("GraphDRP strict runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = _predict_scores_chunked(model, test_scope, test_indices, batch_size=batch_size, use_amp=use_amp)
        test_metrics = compute_binary_metrics(test_labels, test_scores)

        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "auc": test_metrics["auc"],
            "aupr": test_metrics["aupr"],
            "f1": test_metrics["f1"],
            "acc": test_metrics["acc"],
        }
        prediction_rows = build_prediction_rows(test_pairs, test_scores.detach().cpu().numpy())
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "GraphDRP",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "graph_builder": "topk_directed_cosine",
            "graph_inputs": {
                "cell_graph": "similarity.csv",
                "drug_similarity_graph": "physicochemical.csv",
                "response_graph": "train_pairs",
                "drug_structure": "drug_graph_feat",
            },
            "config": {
                "epochs": epochs,
                "lr": lr,
                "dropout": dropout,
                "model_type": model_type,
                "batch_size": batch_size,
                "top_k": top_k,
            },
        },
    )
