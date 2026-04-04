import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data

from benchmarking_common import load_module_from_path, read_json
from benchmarking_common.drug_features import load_graph_feature
from benchmarking_common.splits import (
    PROTOCOL_RANDOM,
    canonicalize_response_pairs,
    list_fold_ids,
    load_fold,
    load_fold_bundle,
    normalize_identifier,
)


def add_sys_paths(paths: Iterable[str]) -> None:
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def load_external_module(module_name: str, file_path: str, extra_paths: Iterable[str] | None = None):
    if extra_paths:
        add_sys_paths(extra_paths)
    return load_module_from_path(module_name, file_path)


def load_prepared_dataset(prepared_dir: str) -> Dict:
    metadata = {}
    metadata_path = os.path.join(prepared_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        metadata = read_json(metadata_path)

    response_pairs = canonicalize_response_pairs(pd.read_csv(os.path.join(prepared_dir, "response_pairs.csv")))
    tables = {}
    for name in [
        "genomics_mutation",
        "transcriptomics_expression",
        "epigenomics_methylation",
        "similarity",
        "physicochemical",
        "pathway",
    ]:
        path = os.path.join(prepared_dir, f"{name}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.map(normalize_identifier)
            tables[name] = df

    aux = {}
    aux_dir = os.path.join(prepared_dir, "aux")
    if os.path.isdir(aux_dir):
        for name in sorted(os.listdir(aux_dir)):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(aux_dir, name)
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.map(normalize_identifier)
            aux[os.path.splitext(name)[0]] = df

    return {
        "prepared_dir": prepared_dir,
        "response_pairs": response_pairs,
        "tables": tables,
        "aux": aux,
        "metadata": metadata,
    }


def load_fold_tables(split_dir: str, fold: int) -> Dict[str, pd.DataFrame]:
    fold_tables = load_fold(split_dir, fold)
    return {name: canonicalize_response_pairs(df) for name, df in fold_tables.items()}


def load_fold_bundle_tables(split_dir: str, fold: int) -> Dict:
    bundle = load_fold_bundle(split_dir, fold)
    return {
        "train": canonicalize_response_pairs(bundle["train"]),
        "val": canonicalize_response_pairs(bundle["val"]),
        "test": canonicalize_response_pairs(bundle["test"]),
        "entities": bundle.get("entities", {}),
        "manifest": bundle.get("manifest", {"protocol": PROTOCOL_RANDOM}),
    }


def resolve_fold_ids(split_dir: str, fold_ids: List[int] | None = None) -> List[int]:
    if fold_ids:
        return sorted(fold_ids)
    return list_fold_ids(split_dir)


def calculate_graph_feat(feat_mat: np.ndarray, adj_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype="float32")
    for i, nodes in enumerate(adj_list):
        for each in nodes:
            adj_mat[i, int(each)] = 1
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return feat_mat, adj_index


def load_hkl_features(graph_dir: str, drug_ids: Iterable[str]) -> Dict[str, Tuple[np.ndarray, List[List[int]], List[int]]]:
    features = {}
    for drug_id in sorted(set(map(str, drug_ids))):
        path = os.path.join(graph_dir, f"{drug_id}.hkl")
        if not os.path.isfile(path):
            continue
        features[drug_id] = load_graph_feature(path)
    return features


def sorted_cell_drug_ids(response_pairs: pd.DataFrame) -> Tuple[List[str], List[str]]:
    return (
        sorted(response_pairs["cell_id"].astype(str).unique().tolist()),
        sorted(response_pairs["drug_id"].astype(str).unique().tolist()),
    )


def combine_sorted_ids(*groups: Iterable[str]) -> List[str]:
    combined = set()
    for group in groups:
        combined.update(map(str, group))
    return sorted(combined)


def filter_pairs_by_scope(
    response_pairs: pd.DataFrame,
    cell_ids: Iterable[str] | None = None,
    drug_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = response_pairs
    if cell_ids is not None:
        out = out[out["cell_id"].isin(list(cell_ids))]
    if drug_ids is not None:
        out = out[out["drug_id"].isin(list(drug_ids))]
    return canonicalize_response_pairs(out)


def subset_frame(df: pd.DataFrame, ids: List[str]) -> pd.DataFrame:
    if not ids:
        return df.iloc[0:0].copy()
    return df.loc[ids]


def protocol_from_bundle(bundle: Dict) -> str:
    return bundle.get("manifest", {}).get("protocol", PROTOCOL_RANDOM)


def all_scope_entities(bundle: Dict) -> Tuple[List[str], List[str]]:
    return (
        combine_sorted_ids(bundle["train"]["cell_id"], bundle["val"]["cell_id"], bundle["test"]["cell_id"]),
        combine_sorted_ids(bundle["train"]["drug_id"], bundle["val"]["drug_id"], bundle["test"]["drug_id"]),
    )


def scope_entities_for_split(bundle: Dict, split_name: str) -> Tuple[List[str], List[str]]:
    protocol = protocol_from_bundle(bundle)
    entities = bundle.get("entities", {})
    if protocol == PROTOCOL_RANDOM or not entities:
        return all_scope_entities(bundle)

    train_cells = entities.get("train_cells", [])
    val_cells = entities.get("val_cells", [])
    test_cells = entities.get("test_cells", [])
    train_drugs = entities.get("train_drugs", [])
    val_drugs = entities.get("val_drugs", [])
    test_drugs = entities.get("test_drugs", [])

    if split_name == "train":
        return sorted(train_cells), sorted(train_drugs)
    if split_name == "val":
        return combine_sorted_ids(train_cells, val_cells), combine_sorted_ids(train_drugs, val_drugs)
    if split_name == "test":
        return combine_sorted_ids(train_cells, test_cells), combine_sorted_ids(train_drugs, test_drugs)
    raise KeyError(f"Unknown split_name '{split_name}'")


def build_label_vector(cell_ids: List[str], drug_ids: List[str], response_pairs: pd.DataFrame) -> torch.Tensor:
    cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_map = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    pos_pairs = response_pairs[response_pairs["label"] == 1]
    if pos_pairs.empty:
        return torch.zeros(len(cell_ids) * len(drug_ids), dtype=torch.float32)
    row = pos_pairs["cell_id"].map(cell_map).to_numpy()
    col = pos_pairs["drug_id"].map(drug_map).to_numpy()
    label = coo_matrix(
        (np.ones(len(pos_pairs), dtype=float), (row, col)),
        shape=(len(cell_ids), len(drug_ids)),
    ).toarray()
    return torch.from_numpy(label.astype(np.float32)).view(-1)


def build_flat_mask(cell_ids: List[str], drug_ids: List[str], pairs: pd.DataFrame) -> torch.Tensor:
    cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_map = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    if pairs.empty:
        return torch.zeros(len(cell_ids) * len(drug_ids), dtype=torch.bool)
    row = pairs["cell_id"].map(cell_map).to_numpy()
    col = pairs["drug_id"].map(drug_map).to_numpy()
    mask = coo_matrix(
        (np.ones(len(pairs), dtype=bool), (row, col)),
        shape=(len(cell_ids), len(drug_ids)),
    ).toarray()
    return torch.from_numpy(mask).view(-1)


def build_graphcdr_train_edge(cell_ids: List[str], drug_ids: List[str], train_pairs: pd.DataFrame) -> np.ndarray:
    cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_map = {drug_id: idx + len(cell_ids) for idx, drug_id in enumerate(drug_ids)}
    if train_pairs.empty:
        return np.empty((0, 3), dtype=np.int64)
    rows = []
    for row in train_pairs.itertuples(index=False):
        label = 1 if int(row.label) == 1 else -1
        rows.append([cell_map[row.cell_id], drug_map[row.drug_id], label])
    return np.asarray(rows, dtype=np.int64)


def build_redcdr_allpairs(cell_ids: List[str], drug_ids: List[str], response_pairs: pd.DataFrame) -> np.ndarray:
    cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_map = {drug_id: idx + len(cell_ids) for idx, drug_id in enumerate(drug_ids)}
    rows = []
    for row in response_pairs.itertuples(index=False):
        label = 1 if int(row.label) == 1 else -1
        rows.append([cell_map[row.cell_id], drug_map[row.drug_id], label])
    allpairs = np.asarray(rows, dtype=np.int64)
    return allpairs[allpairs[:, 2].argsort()]


def build_redcdr_split_objects(
    allpairs: np.ndarray,
    cell_ids: List[str],
    drug_ids: List[str],
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    nb_celllines = len(cell_ids)
    nb_drugs = len(drug_ids)

    def _pairs_to_mask(pairs: pd.DataFrame) -> np.ndarray:
        if pairs.empty:
            return np.zeros((nb_celllines, nb_drugs), dtype=bool)
        cell_map = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        drug_map = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
        row = pairs["cell_id"].map(cell_map).to_numpy()
        col = pairs["drug_id"].map(drug_map).to_numpy()
        return coo_matrix((np.ones(len(pairs), dtype=bool), (row, col)), shape=(nb_celllines, nb_drugs)).toarray()

    train_mask = torch.from_numpy(_pairs_to_mask(train_pairs)).view(-1)
    val_mask = torch.from_numpy(_pairs_to_mask(val_pairs)).view(-1)
    test_mask = torch.from_numpy(_pairs_to_mask(test_pairs)).view(-1)

    label_pos = build_label_vector(cell_ids, drug_ids, pd.DataFrame(
        {
            "cell_id": [cell_ids[row[0]] for row in allpairs if row[2] == 1],
            "drug_id": [drug_ids[row[1] - nb_celllines] for row in allpairs if row[2] == 1],
            "label": [1 for row in allpairs if row[2] == 1],
        }
    ))

    train_edge = build_redcdr_allpairs(cell_ids, drug_ids, train_pairs)
    mirrored = train_edge[:, [1, 0, 2]] if train_edge.size else train_edge
    train_edge = np.vstack((train_edge, mirrored)) if train_edge.size else train_edge
    return train_mask, val_mask, test_mask, train_edge, label_pos


def build_pyg_graphs(drug_ids: List[str], graph_features: Dict[str, Tuple[np.ndarray, List[List[int]], List[int]]]) -> List[Data]:
    graphs: List[Data] = []
    for drug_idx, drug_id in enumerate(drug_ids):
        feat_mat, adj_list, _ = graph_features[drug_id]
        feat, edge_index = calculate_graph_feat(feat_mat, adj_list)
        data = Data(
            x=torch.tensor(feat, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        data.drug_idx = torch.tensor([drug_idx], dtype=torch.long)
        graphs.append(data)
    return graphs


def build_prediction_rows(pairs: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
    rows = []
    for pair, prediction in zip(pairs.itertuples(index=False), predictions):
        rows.append(
            {
                "cell_id": pair.cell_id,
                "drug_id": pair.drug_id,
                "label": int(pair.label),
                "prediction": float(prediction),
            }
        )
    return rows


def build_prediction_rows_from_mask(
    cell_ids: List[str],
    drug_ids: List[str],
    flat_mask,
    flat_labels,
    flat_predictions,
) -> List[Dict]:
    mask_np = tensor_to_numpy(flat_mask).astype(bool).reshape(len(cell_ids), len(drug_ids))
    labels_np = tensor_to_numpy(flat_labels).reshape(len(cell_ids), len(drug_ids))
    preds_np = tensor_to_numpy(flat_predictions).reshape(len(cell_ids), len(drug_ids))
    rows = []
    for cell_idx, cell_id in enumerate(cell_ids):
        for drug_idx, drug_id in enumerate(drug_ids):
            if not mask_np[cell_idx, drug_idx]:
                continue
            rows.append(
                {
                    "cell_id": cell_id,
                    "drug_id": drug_id,
                    "label": int(labels_np[cell_idx, drug_idx]),
                    "prediction": float(preds_np[cell_idx, drug_idx]),
                }
            )
    return rows


def tensor_to_numpy(tensor) -> np.ndarray:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)
