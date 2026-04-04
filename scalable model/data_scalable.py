import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from torch_geometric.data import Data
try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    from torch_geometric.data import DataLoader as PyGDataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from benchmarking_common.drug_features import load_graph_feature


def normalize_identifier(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def normalize_pubchem_id(value) -> str:
    return normalize_identifier(value)


def normalize_label(value) -> int:
    v = int(value)
    if v == -1:
        return 0
    if v == 1:
        return 1
    return 1 if v > 0 else 0


def _resolve_response_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cell_candidates = ["cell_line_id", "cell_id", "DepMapID"]
    drug_candidates = ["pubchem_id", "drug_id", "drug_name", "PUBCHEM_CID"]
    cell_col = next((column for column in cell_candidates if column in df.columns), None)
    drug_col = next((column for column in drug_candidates if column in df.columns), None)
    if cell_col is None or drug_col is None:
        raise KeyError(
            "response_pairs.csv must contain a supported cell and drug identifier column. "
            f"Columns found: {list(df.columns)}"
        )
    return cell_col, drug_col


def calculate_graph_feat(feat_mat: np.ndarray, adj_list: List[List[int]]):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype="float32")
    for i, nodes in enumerate(adj_list):
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def cmask(num: int, ratio: float, seed: int) -> np.ndarray:
    mask = np.ones(num, dtype=bool)
    mask[0 : int(ratio * num)] = False
    rng = np.random.default_rng(seed)
    rng.shuffle(mask)
    return mask


ALLOWED_OMICS_TYPES = {
    "genomics",
    "epigenomics",
    "transcriptomics",
    "proteomics",
    "metabolomics",
    "pathway",
}
RESERVED_NON_OMICS_STEMS = {"response_pairs", "similarity", "physicochemical"}
SUPPORTED_OMICS_CATEGORY_SUBTYPE = {
    ("genomics", "mutation"),
    ("epigenomics", "chromatin"),
    ("epigenomics", "methylation"),
    ("transcriptomics", "expression"),
    ("proteomics", "reverse_phase"),
    ("metabolomics", "profile"),
    ("pathway", "pathway"),
}


def _split_omics_stem(stem: str) -> Tuple[str, str]:
    # Preferred naming convention: {omics_type}_{subtype}
    # Legacy compatibility: pathway.csv -> (pathway, pathway)
    for category in sorted(ALLOWED_OMICS_TYPES, key=len, reverse=True):
        prefix = f"{category}_"
        if stem.startswith(prefix):
            subtype = stem[len(prefix) :]
            if subtype:
                return category, subtype
            return "", ""

    if stem in ALLOWED_OMICS_TYPES:
        return stem, stem
    return "", ""


def _discover_omics_entries(dataset_root: str) -> List[Dict[str, str]]:
    csv_paths = [
        os.path.join(dataset_root, fname)
        for fname in sorted(os.listdir(dataset_root))
        if fname.endswith(".csv")
    ]

    entries: List[Dict[str, str]] = []
    for path in csv_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in RESERVED_NON_OMICS_STEMS:
            continue

        category, subtype = _split_omics_stem(stem)
        if category not in ALLOWED_OMICS_TYPES:
            continue

        if (category, subtype) not in SUPPORTED_OMICS_CATEGORY_SUBTYPE:
            continue

        entries.append(
            {
                "stem": stem,
                "category": category,
                "subtype": subtype,
                "path": path,
                "file": os.path.basename(path),
            }
        )
    return entries


def list_available_omics(dataset_root: str) -> List[Dict[str, str]]:
    """
    Return discovered omics selectors from a prepared SOULCDR dataset directory.
    Each selector can be used via `--omics <stem>` or by category.
    """
    entries = _discover_omics_entries(dataset_root)
    if not entries:
        raise ValueError(
            f"No omics CSVs discovered in {dataset_root}. "
            "Expected stems like {omics_type}_{subtype}.csv."
        )
    return entries


def _select_omics_entries(
    entries: List[Dict[str, str]], selected_omics: Optional[List[str]]
) -> List[Dict[str, str]]:
    if not entries:
        return []

    if not selected_omics:
        return entries

    by_stem = {entry["stem"]: entry for entry in entries}
    by_category: Dict[str, List[Dict[str, str]]] = {}
    for entry in entries:
        by_category.setdefault(entry["category"], []).append(entry)

    selected: Dict[Tuple[str, str], Dict[str, str]] = {}
    unknown_tokens: List[str] = []
    for token in selected_omics:
        if token in by_stem:
            entry = by_stem[token]
            selected[(entry["category"], entry["subtype"])] = entry
            continue
        if token in by_category:
            for entry in by_category[token]:
                selected[(entry["category"], entry["subtype"])] = entry
            continue
        unknown_tokens.append(token)

    if unknown_tokens:
        valid_stems = sorted(by_stem.keys())
        valid_categories = sorted(by_category.keys())
        raise ValueError(
            "Unknown --omics token(s): "
            + ", ".join(unknown_tokens)
            + f". Valid stems: {valid_stems}. Valid categories: {valid_categories}"
        )

    selected_entries = list(selected.values())
    selected_entries.sort(key=lambda x: (x["category"], x["subtype"], x["stem"]))
    return selected_entries


@dataclass
class LoadedScalableData:
    drug_feature: Dict[str, Tuple[np.ndarray, List[List[int]], List[int]]]
    omics_features: Dict[str, Dict[str, pd.DataFrame]]
    similarity_feature: pd.DataFrame
    data_new: List[Tuple[str, str, int]]
    nb_celllines: int
    nb_drugs: int
    physicochemical_feature: Dict[str, np.ndarray]
    selected_omics_stems: List[str]


@dataclass
class ProcessedFoldData:
    drug_loader: PyGDataLoader
    omics_tensors: Dict[str, Dict[str, torch.Tensor]]
    train_edge: np.ndarray
    label_pos: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    atom_shape: int
    physicochemical_tensor: torch.Tensor
    similarity_tensor: torch.Tensor
    nb_celllines: int
    nb_drugs: int


def dataload_scalable(
    dataset_root: str, selected_omics: Optional[List[str]] = None
) -> LoadedScalableData:
    """
    Load and align data from a SOULCDR-compatible dataset directory.

    Expected files under dataset_root:
      - drug_graph_feat/*.hkl
      - response_pairs.csv
      - similarity.csv
      - physicochemical.csv
      - omics CSV files with stems compatible with:
          {omics_type}_{subtype}.csv
        or legacy single-token stems (e.g., transcriptomics.csv)
      - supported omics pairs are restricted to:
          genomics_mutation
          epigenomics_chromatin
          epigenomics_methylation
          transcriptomics_expression
          proteomics_reverse_phase
          metabolomics_profile
          pathway (legacy single token)

    selected_omics:
      - None/empty: load all discovered omics files
      - list of stem tokens and/or categories to include
    """
    drug_feature_dir = os.path.join(dataset_root, "drug_graph_feat")
    response_file = os.path.join(dataset_root, "response_pairs.csv")
    similarity_csv = os.path.join(dataset_root, "similarity.csv")
    physicochemical_csv = os.path.join(dataset_root, "physicochemical.csv")

    for required_path in [drug_feature_dir, response_file, similarity_csv, physicochemical_csv]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required path not found: {required_path}")

    # 1) Drug HKL features
    drug_feature: Dict[str, Tuple[np.ndarray, List[List[int]], List[int]]] = {}
    for filename in sorted(os.listdir(drug_feature_dir)):
        if not filename.endswith(".hkl"):
            continue
        pubchem_id = normalize_identifier(filename.split(".")[0])
        feat_mat, adj_list, degree_list = load_graph_feature(os.path.join(drug_feature_dir, filename))
        drug_feature[pubchem_id] = (feat_mat, adj_list, degree_list)

    if not drug_feature:
        raise ValueError(f"No .hkl files found under {drug_feature_dir}")

    # 2) Omics + similarity
    discovered_entries = _discover_omics_entries(dataset_root)
    selected_entries = _select_omics_entries(discovered_entries, selected_omics)
    if not selected_entries:
        raise ValueError(
            "No omics files selected. "
            "Provide at least one valid --omics selector or omit --omics to use all."
        )

    omics_file_map: Dict[str, Dict[str, str]] = {}
    selected_stems: List[str] = []
    for entry in selected_entries:
        category = entry["category"]
        subtype = entry["subtype"]
        path = entry["path"]
        omics_file_map.setdefault(category, {})[subtype] = path
        selected_stems.append(entry["stem"])

    omics_features: Dict[str, Dict[str, pd.DataFrame]] = {}
    index_sets = []

    for category, subtype_map in omics_file_map.items():
        omics_features[category] = {}
        for subtype, path in subtype_map.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing omics file: {path}")
            df = pd.read_csv(path, header=0, index_col=[0])
            df.index = df.index.astype(str)
            omics_features[category][subtype] = df
            index_sets.append(set(df.index))

    similarity_feature = pd.read_csv(similarity_csv, header=0, index_col=[0])
    similarity_feature.index = similarity_feature.index.astype(str)
    index_sets.append(set(similarity_feature.index))

    common_cells = sorted(set.intersection(*index_sets))
    if not common_cells:
        raise ValueError("No common cell_line_id values across omics + similarity files.")

    for category, subtype_map in omics_features.items():
        for subtype, df in subtype_map.items():
            omics_features[category][subtype] = df.loc[common_cells]
    similarity_feature = similarity_feature.loc[common_cells]

    # 3) Physicochemical (aligned by normalized PUBCHEM ID)
    phys_df = pd.read_csv(physicochemical_csv, header=0, index_col=[0])
    phys_df.index = [normalize_identifier(idx) for idx in phys_df.index]
    physicochemical_feature: Dict[str, np.ndarray] = {}
    for idx, row in phys_df.iterrows():
        physicochemical_feature[idx] = row.values.astype("float32")

    # 4) Response pairs: filter, relabel (-1 -> 0), deduplicate
    resp = pd.read_csv(response_file, header=0)
    cell_col, drug_col = _resolve_response_columns(resp)
    resp = resp.rename(columns={cell_col: "cell_line_id", drug_col: "pubchem_id"})
    resp["cell_line_id"] = resp["cell_line_id"].apply(normalize_identifier)
    resp["pubchem_id"] = resp["pubchem_id"].apply(normalize_identifier)
    resp["label"] = resp["label"].apply(normalize_label)

    resp = resp[resp["pubchem_id"].isin(drug_feature.keys())]
    resp = resp[resp["cell_line_id"].isin(common_cells)]

    data_idx = [
        (str(r.cell_line_id), str(r.pubchem_id), int(r.label))
        for _, r in resp.iterrows()
    ]

    # Prefer positive label if duplicate (cell, drug) appears multiple times.
    data_sort = sorted(data_idx, key=lambda x: (x[0], x[1], x[2]), reverse=True)
    data_seen = set()
    data_new: List[Tuple[str, str, int]] = []
    for pair in data_sort:
        key = (pair[0], pair[1])
        if key not in data_seen:
            data_seen.add(key)
            data_new.append(pair)

    data_new = sorted(data_new, key=lambda x: (x[0], x[1]))

    nb_celllines = len({x[0] for x in data_new})
    nb_drugs = len({x[1] for x in data_new})

    if nb_celllines == 0 or nb_drugs == 0:
        raise ValueError("No valid response pairs after filtering/alignment.")

    return LoadedScalableData(
        drug_feature=drug_feature,
        omics_features=omics_features,
        similarity_feature=similarity_feature,
        data_new=data_new,
        nb_celllines=nb_celllines,
        nb_drugs=nb_drugs,
        physicochemical_feature=physicochemical_feature,
        selected_omics_stems=sorted(selected_stems),
    )


def _build_mask(pairs: np.ndarray, nb_celllines: int, nb_drugs: int) -> torch.Tensor:
    if pairs.shape[0] == 0:
        return torch.zeros(nb_celllines * nb_drugs, dtype=torch.bool)
    mat = coo_matrix(
        (np.ones(pairs.shape[0], dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(nb_celllines, nb_drugs),
    ).toarray()
    return torch.from_numpy(mat).view(-1)


def process_scalable(
    loaded: LoadedScalableData,
    k_folds: int,
    current_fold: int,
    data_split_seed: int,
    drug_batch_size: int,
    split_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> ProcessedFoldData:
    """
    Convert loaded aligned frames/features into tensors/loaders and split masks.

    Splitting protocol matches reference main.py process() behavior:
    - If k_folds > 1: outer test fold + inner validation split (10% of full dataset)
    - Else: 80/10/10 split via deterministic masks
    """
    data_new = loaded.data_new
    drug_feature = loaded.drug_feature
    omics_features = loaded.omics_features
    similarity_feature = loaded.similarity_feature
    physicochemical_feature = loaded.physicochemical_feature

    cell_ids = sorted({item[0] for item in data_new})
    drug_ids = sorted({item[1] for item in data_new})

    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    drug_id_to_idx = {did: i for i, did in enumerate(drug_ids)}

    # All pairs: (cell_idx, drug_idx, label)
    allpairs = np.array(
        [
            [cell_id_to_idx[cid], drug_id_to_idx[did], lbl]
            for cid, did, lbl in data_new
        ],
        dtype=np.int64,
    )
    allpairs = allpairs[allpairs[:, 2].argsort()]

    nb_celllines = len(cell_ids)
    nb_drugs = len(drug_ids)

    # 1) Drug graph loader
    atom_shape = next(iter(drug_feature.values()))[0].shape[-1]
    graphs = []
    for global_drug_idx, drug_id in enumerate(drug_ids):
        if drug_id not in drug_feature:
            raise ValueError(f"Drug ID {drug_id} missing in drug_feature dictionary")
        feat_mat, adj_list, _ = drug_feature[drug_id]
        feat, edge_index = calculate_graph_feat(feat_mat, adj_list)
        graph_data = Data(
            x=torch.tensor(feat, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        graph_data.drug_idx = torch.tensor([global_drug_idx], dtype=torch.long)
        graphs.append(graph_data)

    if drug_batch_size <= 0:
        drug_batch_size = nb_drugs

    drug_loader = PyGDataLoader(
        graphs,
        batch_size=min(drug_batch_size, nb_drugs),
        shuffle=False,
    )

    # 2) Omics tensors in nested dict format expected by scalable cell module
    omics_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    for category, subtype_map in omics_features.items():
        omics_tensors[category] = {}
        for subtype, df in subtype_map.items():
            aligned = df.copy()
            aligned.index = aligned.index.astype(str)
            aligned = aligned.loc[cell_ids]
            omics_tensors[category][subtype] = torch.from_numpy(
                aligned.values.astype(np.float32)
            )

    # 3) Similarity tensor (cell-cell graph features)
    sim_df = similarity_feature.copy()
    sim_df.index = sim_df.index.astype(str)
    sim_df = sim_df.loc[cell_ids]
    similarity_tensor = torch.from_numpy(sim_df.values.astype(np.float32))

    # 4) Physicochemical tensor (drug-drug graph features)
    sample_phys = next(iter(physicochemical_feature.values()))
    phys_dim = len(sample_phys)
    phys_rows = []
    for drug_id in drug_ids:
        if drug_id in physicochemical_feature:
            phys_rows.append(physicochemical_feature[drug_id])
        else:
            phys_rows.append(np.zeros(phys_dim, dtype=np.float32))
    physicochemical_tensor = torch.from_numpy(np.stack(phys_rows).astype(np.float32))

    def _pairs_df_to_array(pairs_df: pd.DataFrame) -> np.ndarray:
        if pairs_df is None or pairs_df.empty:
            return np.empty((0, 3), dtype=np.int64)
        rows = []
        for row in pairs_df.itertuples(index=False):
            cell_id = normalize_identifier(row.cell_id)
            drug_id = normalize_identifier(row.drug_id)
            if cell_id not in cell_id_to_idx or drug_id not in drug_id_to_idx:
                continue
            rows.append([cell_id_to_idx[cell_id], drug_id_to_idx[drug_id], int(row.label)])
        if not rows:
            return np.empty((0, 3), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)

    # 5) Split protocol
    if split_tables is not None:
        train = _pairs_df_to_array(split_tables["train"])
        val = _pairs_df_to_array(split_tables["val"])
        test = _pairs_df_to_array(split_tables["test"])
    elif k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=data_split_seed)
        splits = list(kf.split(allpairs))
        if current_fold < 0 or current_fold >= len(splits):
            raise ValueError(
                f"current_fold={current_fold} out of range for k_folds={k_folds}"
            )

        train_val_idx, test_idx = splits[current_fold]
        test = allpairs[test_idx][:, 0:3]

        val_size = int(len(allpairs) * 0.1)
        val_size = max(1, min(val_size, len(train_val_idx) - 1))
        if val_size <= 0:
            raise ValueError("Unable to create validation split from K-fold train partition")

        rng = np.random.RandomState(data_split_seed)
        perm = rng.permutation(len(train_val_idx))
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        train = allpairs[train_idx][:, 0:3]
        val = allpairs[val_idx][:, 0:3]
    else:
        test_mask_idx = cmask(len(allpairs), 0.1, data_split_seed)
        test = allpairs[~test_mask_idx][:, 0:3]
        train_val = allpairs[test_mask_idx]

        val_mask_idx = cmask(len(train_val), 0.1, data_split_seed + 1)
        train = train_val[val_mask_idx][:, 0:3]
        val = train_val[~val_mask_idx][:, 0:3]

    train_mask = _build_mask(train, nb_celllines, nb_drugs)
    val_mask = _build_mask(val, nb_celllines, nb_drugs)
    test_mask = _build_mask(test, nb_celllines, nb_drugs)

    pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]
    if pos_edge.shape[0] == 0:
        label_pos = torch.zeros(nb_celllines * nb_drugs, dtype=torch.float32)
    else:
        label_pos_arr = coo_matrix(
            (np.ones(pos_edge.shape[0]), (pos_edge[:, 0], pos_edge[:, 1])),
            shape=(nb_celllines, nb_drugs),
        ).toarray()
        label_pos = torch.from_numpy(label_pos_arr).type(torch.FloatTensor).view(-1)

    # Train edges are used for batch graph construction (responds_to)
    train_edge = train.astype(np.int64)

    return ProcessedFoldData(
        drug_loader=drug_loader,
        omics_tensors=omics_tensors,
        train_edge=train_edge,
        label_pos=label_pos,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        atom_shape=atom_shape,
        physicochemical_tensor=physicochemical_tensor,
        similarity_tensor=similarity_tensor,
        nb_celllines=nb_celllines,
        nb_drugs=nb_drugs,
    )
