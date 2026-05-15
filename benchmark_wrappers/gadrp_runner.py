import copy
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.nn import GINConv, global_max_pool

from benchmarking_common import read_json, resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import save_model_outputs
from benchmark_wrappers.common import (
    all_scope_entities,
    build_prediction_rows,
    build_pyg_graphs,
    filter_pairs_by_scope,
    load_fold_bundle_tables,
    load_hkl_features,
    load_prepared_dataset,
    protocol_from_bundle,
    resolve_fold_ids,
    scope_entities_for_split,
    subset_frame,
)


def _sym_adj(adj: coo_matrix) -> coo_matrix:
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()


def _abs_pearson_similarity(df: pd.DataFrame) -> np.ndarray:
    values = df.to_numpy(dtype=np.float32)
    scaled = MinMaxScaler().fit_transform(values)
    sim = np.corrcoef(scaled)
    sim = np.nan_to_num(np.abs(sim), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return sim


def _mean_abs_pearson_similarity(frames: Sequence[pd.DataFrame]) -> np.ndarray:
    matrices = [_abs_pearson_similarity(frame) for frame in frames]
    if not matrices:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(matrices, axis=0)
    return stacked.mean(axis=0).astype(np.float32)


def _topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    n = sim.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.int64)
    k = max(1, min(k, n))
    return np.argsort(sim, axis=1)[:, -k:].astype(np.int64)


def _build_pair_graph_from_similarity(
    pair_indices: np.ndarray,
    num_cells: int,
    num_drugs: int,
    cell_sim: np.ndarray,
    drug_sim: np.ndarray,
    top_k: int,
    pair_top_k: int,
    device: torch.device,
) -> torch.Tensor:
    pair_num = int(pair_indices.shape[0])
    if pair_num == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            size=(0, 0),
            device=device,
        )

    drug_topk = _topk_indices(drug_sim, top_k)
    cell_topk = _topk_indices(cell_sim, top_k)
    pair_lookup = np.full((num_cells, num_drugs), -1, dtype=np.int64)
    pair_lookup[pair_indices[:, 0], pair_indices[:, 1]] = np.arange(pair_num, dtype=np.int64)

    rows: List[int] = []
    cols: List[int] = []
    values: List[float] = []

    for node_idx, (cell_idx, drug_idx) in enumerate(pair_indices):
        candidates = {node_idx: 1.0}
        for drug_neighbor in drug_topk[drug_idx]:
            drug_score = float(drug_sim[drug_idx, drug_neighbor])
            for cell_neighbor in cell_topk[cell_idx]:
                pair_neighbor = int(pair_lookup[cell_neighbor, drug_neighbor])
                if pair_neighbor < 0:
                    continue
                score = (drug_score + float(cell_sim[cell_idx, cell_neighbor])) / 2.0
                existing = candidates.get(pair_neighbor)
                if existing is None or score > existing:
                    candidates[pair_neighbor] = score

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:pair_top_k]
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


class GadrpModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.SELU(),
            nn.Linear(2048, 1024),
            nn.SELU(),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GadrpGraphEncoder(nn.Module):
    def __init__(self, atom_dim: int, hidden_dim: int = 200, num_layers: int = 3):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.layers.append(GINConv(mlp, train_eps=True))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.atom_embed(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        h = global_max_pool(h, batch)
        return self.proj(h)


@dataclass
class GadrpScope:
    mode: str
    cell_ids: List[str]
    drug_ids: List[str]
    cell_tensors: List[torch.Tensor]
    pair_indices: torch.Tensor
    labels: torch.Tensor
    pair_graph: torch.Tensor
    fingerprint_tensor: torch.Tensor | None
    graph_batch: object | None


class GADRPClassifier(nn.Module):
    def __init__(
        self,
        cell_input_dims: Sequence[int],
        mode: str,
        fingerprint_dim: int | None = None,
        atom_dim: int | None = None,
        dropout: float = 0.2,
        irgcn_layers: int = 5,
        alpha: float = 0.1,
    ):
        super().__init__()
        from model.drug_cell_encoder import Drug_cell_encoder

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        if mode == "native":
            if fingerprint_dim is None:
                raise ValueError("fingerprint_dim is required for native GADRP mode")
            self.drug_encoder = nn.Linear(fingerprint_dim, 200)
        else:
            if atom_dim is None:
                raise ValueError("atom_dim is required for graph GADRP mode")
            self.drug_encoder = GadrpGraphEncoder(atom_dim=atom_dim, hidden_dim=200)

        self.cell_encoders = nn.ModuleList([GadrpModalityEncoder(dim, 400) for dim in cell_input_dims])
        self.cellfc1 = nn.Linear(400 * len(cell_input_dims), 200)
        self.embedding = Drug_cell_encoder(400, device=None, num_layers=irgcn_layers, alpha=alpha, dropout=dropout)
        self.att = nn.Parameter(torch.full((irgcn_layers,), 1.0 / max(1, irgcn_layers)))

        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(
        self,
        cell_tensors: List[torch.Tensor],
        pair_graph: torch.Tensor,
        pair_indices: torch.Tensor,
        fingerprint_tensor: torch.Tensor | None = None,
        graph_batch=None,
    ) -> torch.Tensor:
        if pair_indices.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=pair_graph.device)
        if self.mode == "native":
            if fingerprint_tensor is None:
                raise ValueError("fingerprint_tensor is required in native mode")
            drug_feature = self.dropout(self.relu(self.drug_encoder(fingerprint_tensor)))
        else:
            if graph_batch is None:
                raise ValueError("graph_batch is required in graph mode")
            drug_feature = self.drug_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

        cell_latents = [encoder(tensor) for encoder, tensor in zip(self.cell_encoders, cell_tensors)]
        cell_feature = self.dropout(self.relu(self.cellfc1(torch.cat(cell_latents, dim=1))))

        pair_feature = torch.cat((drug_feature[pair_indices[:, 1]], cell_feature[pair_indices[:, 0]]), dim=1)

        pair_features = self.embedding(pair_feature, pair_graph)
        feature = torch.zeros_like(pair_features[0])
        for idx, pair_hidden in enumerate(pair_features):
            feature = feature + self.att[idx] * pair_hidden
        feature = self.dropout(feature)
        feature = self.dropout(self.relu(self.fc1(feature)))
        feature = self.dropout(self.relu(self.fc2(feature)))
        return self.sigmoid(self.fc3(feature)).view(-1)


def _resolve_mode(dataset: Dict) -> str:
    tables = dataset["tables"]
    if "drug_fingerprint" in tables and "genomics_cnv" in tables:
        return "native"
    return "graph"


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
        if row.cell_id not in cell_map or row.drug_id not in drug_map:
            continue
        indices.append([cell_map[row.cell_id], drug_map[row.drug_id]])
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


def _build_scope(
    dataset: Dict,
    scope_pairs: pd.DataFrame,
    cell_ids: List[str],
    drug_ids: List[str],
    cell_sim_global: pd.DataFrame,
    drug_sim_global: pd.DataFrame,
    mode: str,
    device: torch.device,
    top_k: int,
    pair_top_k: int,
) -> GadrpScope:
    tables = dataset["tables"]

    if mode == "native":
        cell_frames = [
            subset_frame(tables["transcriptomics_expression"], cell_ids),
            subset_frame(tables["genomics_cnv"], cell_ids),
        ]
        fingerprint = subset_frame(tables["drug_fingerprint"], drug_ids)
        graph_batch = None
    else:
        cell_frames = [
            subset_frame(tables["genomics_mutation"], cell_ids),
            subset_frame(tables["transcriptomics_expression"], cell_ids),
            subset_frame(tables["epigenomics_methylation"], cell_ids),
        ]
        fingerprint = None
        graph_dir = os.path.join(dataset["prepared_dir"], "drug_graph_feat")
        graph_features = load_hkl_features(graph_dir, drug_ids)
        graphs = build_pyg_graphs(drug_ids, graph_features)
        graph_loader = PyGDataLoader(graphs, batch_size=len(graphs), shuffle=False)
        graph_batch = next(iter(graph_loader)).to(device)

    pair_indices, labels = _pairs_to_local_tensors(scope_pairs, cell_ids, drug_ids, device)
    cell_tensors = [torch.from_numpy(frame.to_numpy(dtype="float32")).to(device) for frame in cell_frames]
    cell_sim = cell_sim_global.loc[cell_ids, cell_ids].to_numpy(dtype=np.float32)
    drug_sim = drug_sim_global.loc[drug_ids, drug_ids].to_numpy(dtype=np.float32)
    pair_graph = _build_pair_graph_from_similarity(
        pair_indices=pair_indices.detach().cpu().numpy(),
        num_cells=len(cell_ids),
        num_drugs=len(drug_ids),
        cell_sim=cell_sim,
        drug_sim=drug_sim,
        top_k=top_k,
        pair_top_k=pair_top_k,
        device=device,
    )

    fingerprint_tensor = None
    if fingerprint is not None:
        fingerprint_tensor = torch.from_numpy(fingerprint.to_numpy(dtype="float32")).to(device)

    return GadrpScope(
        mode=mode,
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        cell_tensors=cell_tensors,
        pair_indices=pair_indices,
        labels=labels,
        pair_graph=pair_graph,
        fingerprint_tensor=fingerprint_tensor,
        graph_batch=graph_batch,
    )


def _predict_pairs(model: GADRPClassifier, scope: GadrpScope) -> torch.Tensor:
    return model(
        cell_tensors=scope.cell_tensors,
        pair_graph=scope.pair_graph,
        pair_indices=scope.pair_indices,
        fingerprint_tensor=scope.fingerprint_tensor,
        graph_batch=scope.graph_batch,
    )


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
    prepared_metadata = read_json(os.path.join(prepared_dir, "metadata.json")) if os.path.isfile(os.path.join(prepared_dir, "metadata.json")) else {}
    mode = _resolve_mode(dataset)
    tables = dataset["tables"]

    if mode == "native":
        if "transcriptomics_miRNA" in tables and "epigenomics_methylation" in tables:
            cell_sim_global = pd.DataFrame(
                _mean_abs_pearson_similarity(
                    [
                        tables["transcriptomics_miRNA"],
                        tables["epigenomics_methylation"],
                    ]
                ),
                index=tables["transcriptomics_miRNA"].index,
                columns=tables["transcriptomics_miRNA"].index,
            )
            cell_similarity_source = prepared_metadata.get(
                "gadrp_cell_similarity_source",
                ["transcriptomics_miRNA", "epigenomics_methylation"],
            )
        else:
            cell_sim_features = tables["similarity"]
            cell_sim_global = pd.DataFrame(
                _abs_pearson_similarity(cell_sim_features),
                index=cell_sim_features.index,
                columns=cell_sim_features.index,
            )
            cell_similarity_source = prepared_metadata.get(
                "native_similarity_modalities",
                prepared_metadata.get("cell_similarity_source", ["similarity"]),
            )

        if "physicochemical" in tables:
            drug_sim_features = tables["physicochemical"]
            drug_similarity_source = prepared_metadata.get(
                "gadrp_drug_similarity_source",
                prepared_metadata.get("drug_similarity_source", "physicochemical"),
            )
        else:
            drug_sim_features = tables["drug_fingerprint"]
            drug_similarity_source = "drug_fingerprint"
    else:
        cell_sim_features = tables["similarity"]
        cell_sim_global = pd.DataFrame(
            _abs_pearson_similarity(cell_sim_features),
            index=cell_sim_features.index,
            columns=cell_sim_features.index,
        )
        drug_sim_features = tables["physicochemical"]
        cell_similarity_source = ["similarity"]
        drug_similarity_source = "physicochemical"

    drug_sim_global = pd.DataFrame(
        _abs_pearson_similarity(drug_sim_features),
        index=drug_sim_features.index,
        columns=drug_sim_features.index,
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
            train_pairs_scope = train_pairs
            val_pairs_scope = val_pairs
            test_pairs_scope = test_pairs
            train_scope = _build_scope(
                dataset,
                train_pairs_scope,
                all_cells,
                all_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )
            val_scope = _build_scope(
                dataset,
                val_pairs_scope,
                all_cells,
                all_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )
            test_scope = _build_scope(
                dataset,
                test_pairs_scope,
                all_cells,
                all_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            train_pairs_scope = filter_pairs_by_scope(train_pairs, train_cells, train_drugs)
            val_pairs_scope = filter_pairs_by_scope(val_pairs, val_cells, val_drugs)
            test_pairs_scope = filter_pairs_by_scope(test_pairs, test_cells, test_drugs)
            train_scope = _build_scope(
                dataset,
                train_pairs_scope,
                train_cells,
                train_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )
            val_scope = _build_scope(
                dataset,
                val_pairs_scope,
                val_cells,
                val_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )
            test_scope = _build_scope(
                dataset,
                test_pairs_scope,
                test_cells,
                test_drugs,
                cell_sim_global,
                drug_sim_global,
                mode,
                runtime_device,
                top_k,
                pair_top_k,
            )

        if mode == "native":
            model = GADRPClassifier(
                cell_input_dims=[tensor.shape[1] for tensor in train_scope.cell_tensors],
                mode=mode,
                fingerprint_dim=int(train_scope.fingerprint_tensor.shape[1]),
                dropout=dropout,
                irgcn_layers=irgcn_layers,
                alpha=alpha,
            ).to(runtime_device)
        else:
            atom_dim = int(test_scope.graph_batch.x.shape[1])
            model = GADRPClassifier(
                cell_input_dims=[tensor.shape[1] for tensor in train_scope.cell_tensors],
                mode=mode,
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
            raise RuntimeError("GADRP runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = _predict_pairs(model, test_scope).cpu()

        metrics = compute_binary_metrics(test_scope.labels.cpu(), test_scores)
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
            "model": "GADRP",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "mode": mode,
            "cell_similarity_source": cell_similarity_source,
            "drug_similarity_source": drug_similarity_source,
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
