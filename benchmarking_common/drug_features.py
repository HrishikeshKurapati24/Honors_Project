import os
import shutil
from typing import Dict, Iterable, List, Tuple

import h5py
import hickle as hkl
import numpy as np
import pandas as pd

from benchmarking_common import ensure_dir


def load_graph_keys(graph_dir: str) -> List[str]:
    if not os.path.isdir(graph_dir):
        return []
    return sorted([os.path.splitext(name)[0] for name in os.listdir(graph_dir) if name.endswith(".hkl")])


def copy_graph_subset(source_dir: str, output_dir: str, drug_ids: Iterable[str]) -> None:
    ensure_dir(output_dir)
    source_keys = {key: f"{key}.hkl" for key in load_graph_keys(source_dir)}
    for drug_id in sorted(set(map(str, drug_ids))):
        if drug_id not in source_keys:
            continue
        src = os.path.join(source_dir, source_keys[drug_id])
        dst = os.path.join(output_dir, f"{drug_id}.hkl")
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        shutil.copy2(src, dst)


def _sorted_hickle_keys(group: h5py.Group) -> List[str]:
    def sort_key(name: str) -> Tuple[int, str]:
        if name.startswith("data_"):
            try:
                return int(name.split("_", 1)[1]), name
            except ValueError:
                pass
        return 10**9, name

    return sorted(group.keys(), key=sort_key)


def _encode_hickle_type_attr(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        flat_values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        flat_values = list(value)
    else:
        flat_values = [value]

    encoded = []
    for item in flat_values:
        if isinstance(item, bytes):
            encoded.append(item)
        else:
            encoded.append(str(item).encode("utf-8"))

    max_len = max((len(item) for item in encoded), default=1)
    return np.asarray(encoded, dtype=f"|S{max_len}")


def _repair_hickle_type_attrs(path: str) -> bool:
    changed = False

    def maybe_repair(node) -> None:
        nonlocal changed
        if "type" not in node.attrs:
            return
        current = node.attrs["type"]
        if isinstance(current, np.ndarray):
            needs_repair = current.dtype.kind in {"U", "O"}
        else:
            needs_repair = isinstance(current, str)
        if not needs_repair:
            return
        encoded = _encode_hickle_type_attr(current)
        del node.attrs["type"]
        node.attrs.create("type", encoded, dtype=encoded.dtype)
        changed = True

    with h5py.File(path, "r+") as handle:
        maybe_repair(handle)
        handle.visititems(lambda _name, obj: maybe_repair(obj))
    return changed


def repair_graph_hkl_directory(graph_dir: str) -> None:
    if not os.path.isdir(graph_dir):
        return
    for key in load_graph_keys(graph_dir):
        _repair_hickle_type_attrs(os.path.join(graph_dir, f"{key}.hkl"))


def _load_graph_feature_fallback(path: str) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    def load_node(node):
        if isinstance(node, h5py.Dataset):
            value = np.asarray(node)
            if value.shape == ():
                return value.item()
            return value.tolist()
        return [load_node(node[key]) for key in _sorted_hickle_keys(node)]

    with h5py.File(path, "r") as handle:
        payload = handle["data_0"]
        feat_mat = np.asarray(payload["data_0"])
        adj_list = load_node(payload["data_1"])
        degree_list = load_node(payload["data_2"])

    return feat_mat, adj_list, degree_list


def load_graph_feature(path: str) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    try:
        loaded = hkl.load(path)
    except Exception as original_error:
        loaded = None
        try:
            repaired = _repair_hickle_type_attrs(path)
            if repaired:
                loaded = hkl.load(path)
        except Exception:
            loaded = None

        if loaded is None:
            try:
                loaded = _load_graph_feature_fallback(path)
            except Exception:
                raise original_error

    feat_mat, adj_list, degree_list = loaded
    feat_mat = np.asarray(feat_mat)

    if isinstance(adj_list, np.ndarray):
        if adj_list.ndim == 1:
            adj_list = [adj_list.tolist()]
        else:
            adj_list = adj_list.tolist()
    elif adj_list and not isinstance(adj_list[0], (list, tuple, np.ndarray)):
        adj_list = [list(adj_list)]
    adj_list = [list(map(int, nodes)) for nodes in adj_list]

    if isinstance(degree_list, np.ndarray):
        degree_list = degree_list.tolist()
    elif not isinstance(degree_list, (list, tuple)):
        degree_list = [degree_list]
    degree_list = [int(value) for value in degree_list]
    return feat_mat, adj_list, degree_list


def load_smiles_txt(path: str) -> Dict[str, str]:
    smiles_map: Dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            smiles_map[str(parts[0]).strip()] = parts[1].strip()
    return smiles_map


def build_graph_from_smiles(smiles: str) -> Tuple[np.ndarray, List[List[int]], List[int]]:
    try:
        import deepchem as dc
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "Generating new HKL graph features requires deepchem and rdkit."
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    featurizer = dc.feat.ConvMolFeaturizer()
    mol_object = featurizer.featurize([mol])[0]
    features = mol_object.get_atom_features()
    degree_list = mol_object.deg_list
    adj_list = mol_object.get_adjacency_list()
    return features, adj_list, degree_list


def build_hkl_graphs_from_smiles(smiles_map: Dict[str, str], output_dir: str) -> None:
    ensure_dir(output_dir)
    for drug_id, smiles in smiles_map.items():
        output_path = os.path.join(output_dir, f"{drug_id}.hkl")
        if os.path.exists(output_path):
            continue
        hkl.dump(build_graph_from_smiles(smiles), output_path)
        _repair_hickle_type_attrs(output_path)


def build_pubchem_fingerprint(cid: str, smiles: str | None = None) -> np.ndarray:
    try:
        import pubchempy as pcp

        compound = pcp.Compound.from_cid(int(cid))
        fingerprint = "".join("{:04b}".format(int(token, 16)) for token in compound.fingerprint)
        return np.asarray([int(char) for char in fingerprint], dtype=np.float32)
    except Exception:
        if smiles is None:
            raise
        from rdkit import Chem
        from rdkit.Chem import RDKFingerprint

        mol = Chem.MolFromSmiles(smiles)
        fp = RDKFingerprint(mol, fpSize=920)
        return np.asarray(list(fp), dtype=np.float32)


def build_fingerprint_table(drug_ids: Iterable[str], smiles_map: Dict[str, str], output_csv: str) -> pd.DataFrame:
    ensure_dir(os.path.dirname(output_csv))
    rows = []
    for drug_id in sorted(set(map(str, drug_ids))):
        smiles = smiles_map.get(drug_id)
        vector = build_pubchem_fingerprint(drug_id, smiles=smiles)
        rows.append([drug_id] + vector.astype(int).tolist())
    columns = ["drug_id"] + [str(index) for index in range(len(rows[0]) - 1)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    return df
