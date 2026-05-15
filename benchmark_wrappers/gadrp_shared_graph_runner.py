import copy
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PyGDataLoader

from benchmarking_common import read_json, resolve_device, set_seed
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
    all_scope_entities,
    build_cosine_similarity_matrix,
    build_prediction_rows,
    build_pyg_graphs,
    build_strict_response_edge_index,
    build_shared_similarity_graphs,
    edge_index_to_neighbor_lists,
    filter_pairs_by_scope,
    load_fold_bundle_tables,
    load_hkl_features,
    load_prepared_dataset,
    protocol_from_bundle,
    resolve_fold_ids,
    scope_entities_for_split,
    subset_frame,
    tensor_to_numpy,
)
from benchmark_wrappers.gadrp_runner import GADRPClassifier


def _sym_adj(adj: coo_matrix) -> coo_matrix:
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()


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


def _build_pair_graph_from_shared_edges(
    pair_indices: torch.Tensor,
    cell_similarity: np.ndarray,
    drug_similarity: np.ndarray,
    cell_edge_index: torch.Tensor,
    drug_edge_index: torch.Tensor,
    support_positions: np.ndarray,
    num_cells: int,
    num_drugs: int,
    pair_top_k: int,
    device: torch.device,
) -> torch.Tensor:
    pair_index_np = tensor_to_numpy(pair_indices).astype(np.int64)
    pair_num = int(pair_index_np.shape[0])
    if pair_num == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            size=(0, 0),
            device=device,
        )

    cell_neighbors = edge_index_to_neighbor_lists(cell_edge_index, num_cells)
    drug_neighbors = edge_index_to_neighbor_lists(drug_edge_index, num_drugs)
    pair_lookup = np.full((num_cells, num_drugs), -1, dtype=np.int64)
    pair_lookup[pair_index_np[:, 0], pair_index_np[:, 1]] = np.arange(pair_num, dtype=np.int64)

    support_mask = np.zeros(pair_num, dtype=bool)
    support_mask[support_positions] = True

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []
    topn = max(1, int(pair_top_k))

    for node_idx, (cell_idx, drug_idx) in enumerate(pair_index_np):
        candidates = {node_idx: 1.0}
        for cell_neighbor in cell_neighbors[cell_idx]:
            cell_score = float(cell_similarity[cell_idx, cell_neighbor])
            for drug_neighbor in drug_neighbors[drug_idx]:
                pair_neighbor = int(pair_lookup[cell_neighbor, drug_neighbor])
                if pair_neighbor < 0:
                    continue
                if pair_neighbor != node_idx and not support_mask[pair_neighbor]:
                    continue
                score = (cell_score + float(drug_similarity[drug_idx, drug_neighbor])) / 2.0
                current = candidates.get(pair_neighbor)
                if current is None or score > current:
                    candidates[pair_neighbor] = score

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:topn]
        for pair_neighbor, score in ranked:
            rows.append(node_idx)
            cols.append(pair_neighbor)
            values.append(score)

    edge_idx = coo_matrix((values, (rows, cols)), shape=(pair_num, pair_num))
    edge_idx = _sym_adj(edge_idx)
    indices = np.vstack((edge_idx.row, edge_idx.col))
    return torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(edge_idx.data, dtype=torch.float32, device=device),
        size=(pair_num, pair_num),
        device=device,
    ).coalesce()


@dataclass
class StrictGadrpScope:
    cell_ids: List[str]
    drug_ids: List[str]
    cell_tensors: List[torch.Tensor]
    graph_pair_indices: torch.Tensor
    query_pair_positions: torch.Tensor
    labels: torch.Tensor
    pair_graph: torch.Tensor
    graph_batch: object


def _build_scope(
    dataset: Dict,
    train_pairs: pd.DataFrame,
    query_pairs: pd.DataFrame,
    cell_ids: List[str],
    drug_ids: List[str],
    device: torch.device,
    top_k: int,
    pair_top_k: int,
) -> StrictGadrpScope:
    tables = dataset["tables"]
    mutation = subset_frame(tables["genomics_mutation"], cell_ids)
    expression = subset_frame(tables["transcriptomics_expression"], cell_ids)
    methylation = subset_frame(tables["epigenomics_methylation"], cell_ids)
    similarity = subset_frame(tables["similarity"], cell_ids)
    physicochemical = subset_frame(tables["physicochemical"], drug_ids)

    graph_dir = os.path.join(dataset["prepared_dir"], "drug_graph_feat")
    graph_features = load_hkl_features(graph_dir, drug_ids)
    graphs = build_pyg_graphs(drug_ids, graph_features)
    graph_loader = PyGDataLoader(graphs, batch_size=len(graphs), shuffle=False)
    graph_batch = next(iter(graph_loader)).to(device)

    support_pairs = filter_pairs_by_scope(train_pairs, cell_ids, drug_ids)
    query_pairs = filter_pairs_by_scope(query_pairs, cell_ids, drug_ids)
    scope_pairs = pd.concat([support_pairs, query_pairs], ignore_index=True).drop_duplicates(
        ["cell_id", "drug_id"],
        keep="first",
    )
    scope_pairs = scope_pairs.reset_index(drop=True)
    graph_pair_indices, _ = _pairs_to_local_tensors(scope_pairs, cell_ids, drug_ids, device)
    query_lookup = {
        (row.cell_id, row.drug_id): idx
        for idx, row in enumerate(scope_pairs.itertuples(index=False))
    }
    query_pair_positions = torch.tensor(
        [query_lookup[(row.cell_id, row.drug_id)] for row in query_pairs.itertuples(index=False)],
        dtype=torch.long,
        device=device,
    )
    labels = torch.tensor(query_pairs["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    cell_tensors = [
        torch.from_numpy(mutation.to_numpy(dtype="float32")).to(device),
        torch.from_numpy(expression.to_numpy(dtype="float32")).to(device),
        torch.from_numpy(methylation.to_numpy(dtype="float32")).to(device),
    ]

    shared_graphs = build_shared_similarity_graphs(
        cell_similarity_features=similarity.to_numpy(dtype=np.float32),
        drug_similarity_features=physicochemical.to_numpy(dtype=np.float32),
        top_k=top_k,
        device=device,
    )
    response_edge_index = build_strict_response_edge_index(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        train_pairs=support_pairs,
        device=device,
    )
    validate_strict_response_edge_index(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        train_pairs=support_pairs,
        response_edge_index=response_edge_index,
    )
    cell_similarity = build_cosine_similarity_matrix(similarity.to_numpy(dtype=np.float32))
    drug_similarity = build_cosine_similarity_matrix(physicochemical.to_numpy(dtype=np.float32))
    pair_graph = _build_pair_graph_from_shared_edges(
        pair_indices=graph_pair_indices,
        cell_similarity=cell_similarity,
        drug_similarity=drug_similarity,
        cell_edge_index=shared_graphs[("cell", "similar_to", "cell")],
        drug_edge_index=shared_graphs[("drug", "similar_to", "drug")],
        support_positions=query_pair_positions.new_tensor(
            [query_lookup[(row.cell_id, row.drug_id)] for row in support_pairs.itertuples(index=False)],
            dtype=torch.long,
        ).cpu().numpy(),
        num_cells=len(cell_ids),
        num_drugs=len(drug_ids),
        pair_top_k=pair_top_k,
        device=device,
    )
    return StrictGadrpScope(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        cell_tensors=cell_tensors,
        graph_pair_indices=graph_pair_indices,
        query_pair_positions=query_pair_positions,
        labels=labels,
        pair_graph=pair_graph,
        graph_batch=graph_batch,
    )


def _predict_pairs(model: GADRPClassifier, scope: StrictGadrpScope) -> torch.Tensor:
    all_scores = model(
        cell_tensors=scope.cell_tensors,
        pair_graph=scope.pair_graph,
        pair_indices=scope.graph_pair_indices,
        graph_batch=scope.graph_batch,
    )
    return all_scores[scope.query_pair_positions]


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 150,
    lr: float = 1e-4,
    dropout: float = 0.2,
    weight_decay: float = 0.0,
    top_k: int = 10,
    pair_top_k: int = 10,
    irgcn_layers: int = 5,
    alpha: float = 0.1,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)

    model_root = os.path.join(root_dir, "benchmark models", "GADRP-main")
    if model_root not in sys.path:
        sys.path.insert(0, model_root)

    dataset = load_prepared_dataset(prepared_dir)
    validate_strict_prepared_metadata(prepared_dir, dataset["metadata"])
    validate_strict_model_contract(
        "GADRP",
        predictive_inputs=(
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ),
        graph_inputs=("similarity.csv", "physicochemical.csv", "train_pairs"),
    )
    prepared_metadata = read_json(os.path.join(prepared_dir, "metadata.json")) if os.path.isfile(os.path.join(prepared_dir, "metadata.json")) else {}
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
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == "random":
            all_cells, all_drugs = all_scope_entities(bundle)
            train_scope = _build_scope(dataset, train_pairs, train_pairs, all_cells, all_drugs, runtime_device, top_k, pair_top_k)
            val_scope = _build_scope(dataset, train_pairs, val_pairs, all_cells, all_drugs, runtime_device, top_k, pair_top_k)
            test_scope = _build_scope(dataset, train_pairs, test_pairs, all_cells, all_drugs, runtime_device, top_k, pair_top_k)
            test_pairs_scope = test_pairs
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            train_pairs_scope = filter_pairs_by_scope(train_pairs, train_cells, train_drugs)
            val_pairs_scope = filter_pairs_by_scope(val_pairs, val_cells, val_drugs)
            test_pairs_scope = filter_pairs_by_scope(test_pairs, test_cells, test_drugs)

            train_scope = _build_scope(dataset, train_pairs_scope, train_pairs_scope, train_cells, train_drugs, runtime_device, top_k, pair_top_k)
            val_scope = _build_scope(dataset, train_pairs_scope, val_pairs_scope, val_cells, val_drugs, runtime_device, top_k, pair_top_k)
            test_scope = _build_scope(dataset, train_pairs_scope, test_pairs_scope, test_cells, test_drugs, runtime_device, top_k, pair_top_k)

        atom_dim = int(train_scope.graph_batch.x.shape[1])
        model = GADRPClassifier(
            cell_input_dims=[tensor.shape[1] for tensor in train_scope.cell_tensors],
            mode="graph",
            atom_dim=atom_dim,
            dropout=dropout,
            irgcn_layers=irgcn_layers,
            alpha=alpha,
        ).to(runtime_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        best_val_auc = -1.0
        best_state = None
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            model.train()
            optimizer.zero_grad()
            train_scores = _predict_pairs(model, train_scope)
            loss = criterion(train_scores, train_scope.labels)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_scores = _predict_pairs(model, val_scope)
            val_metrics = compute_binary_metrics(val_scope.labels, val_scores)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Strict GADRP runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = _predict_pairs(model, test_scope).cpu()

        metrics = compute_binary_metrics(test_scope.labels.cpu(), test_scores)
        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "auc": metrics["auc"],
            "aupr": metrics["aupr"],
            "f1": metrics["f1"],
            "acc": metrics["acc"],
        }
        prediction_rows = build_prediction_rows(
            test_pairs_scope.reset_index(drop=True),
            test_scores.numpy(),
        )
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "GADRP",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "cell_similarity_source": prepared_metadata.get("cell_graph_source", "similarity.csv"),
            "drug_similarity_source": prepared_metadata.get("drug_similarity_graph_source", "physicochemical.csv"),
            "config": {
                "lr": lr,
                "dropout": dropout,
                "top_k": top_k,
                "pair_top_k": pair_top_k,
                "irgcn_layers": irgcn_layers,
                "alpha": alpha,
            },
        },
    )
