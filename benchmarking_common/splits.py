import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from benchmarking_common import ensure_dir

CELL_ALIASES = ("cell_id", "cell_line_id", "cell_line", "DepMapID", "depMapID")
DRUG_ALIASES = ("drug_id", "pubchem_id", "drug_name", "PUBCHEM_CID")

PROTOCOL_RANDOM = "random"
PROTOCOL_UNSEEN_CELLS = "unseen_cells"
PROTOCOL_UNSEEN_DRUGS = "unseen_drugs"
PROTOCOL_UNSEEN_BOTH = "unseen_both"
SUPPORTED_PROTOCOLS = (
    PROTOCOL_RANDOM,
    PROTOCOL_UNSEEN_CELLS,
    PROTOCOL_UNSEEN_DRUGS,
    PROTOCOL_UNSEEN_BOTH,
)


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


def normalize_label(value) -> int:
    if pd.isna(value):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"true", "sensitive", "yes"}:
            return 1
        return 0
    if numeric == -1:
        return 0
    return 1 if numeric > 0 else 0


def _resolve_column(columns: List[str], candidates: List[str], default: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if default in columns:
        return default
    raise KeyError(f"Unable to resolve a column from {candidates} in {columns}")


def canonicalize_response_pairs(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    cell_col = _resolve_column(columns, list(CELL_ALIASES), "cell_id")
    drug_col = _resolve_column(columns, list(DRUG_ALIASES), "drug_id")
    if "label" not in df.columns:
        raise KeyError("Response pairs table must contain a label column")

    out = pd.DataFrame(
        {
            "cell_id": df[cell_col].map(normalize_identifier),
            "drug_id": df[drug_col].map(normalize_identifier),
            "label": df["label"].map(normalize_label),
        }
    )
    out = out[(out["cell_id"] != "") & (out["drug_id"] != "")]
    out = out.sort_values(["cell_id", "drug_id", "label"], ascending=[True, True, False])
    out = out.drop_duplicates(["cell_id", "drug_id"], keep="first")
    out = out.sort_values(["cell_id", "drug_id"]).reset_index(drop=True)
    return out


def _entity_split(values: Sequence[str], seed: int, n_splits: int, val_ratio_of_full: float) -> List[Dict[str, List[str]]]:
    values = np.asarray(sorted(map(str, values)), dtype=object)
    if values.size < n_splits:
        raise ValueError(f"Cannot create {n_splits} folds from only {values.size} entities")

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Dict[str, List[str]]] = []
    for fold_id, (train_val_idx, test_idx) in enumerate(splitter.split(values), start=1):
        train_val_values = values[train_val_idx]
        test_values = values[test_idx]

        val_size = int(len(values) * val_ratio_of_full)
        val_size = max(1, min(val_size, len(train_val_values) - 1))
        rng = np.random.RandomState(seed + fold_id * 1000)  # fold-dependent seed for independent val draws
        perm = rng.permutation(len(train_val_values))
        val_values = train_val_values[perm[:val_size]]
        train_values = train_val_values[perm[val_size:]]

        folds.append(
            {
                "fold": fold_id,
                "train": sorted(train_values.tolist()),
                "val": sorted(val_values.tolist()),
                "test": sorted(test_values.tolist()),
            }
        )
    return folds


def _pairs_for_entities(
    canonical: pd.DataFrame,
    allowed_cells: Sequence[str] | None = None,
    allowed_drugs: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = canonical
    if allowed_cells is not None:
        out = out[out["cell_id"].isin(allowed_cells)]
    if allowed_drugs is not None:
        out = out[out["drug_id"].isin(allowed_drugs)]
    return out.reset_index(drop=True)


def _build_fold_record(
    fold_id: int,
    protocol: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_cells: Sequence[str],
    val_cells: Sequence[str],
    test_cells: Sequence[str],
    train_drugs: Sequence[str],
    val_drugs: Sequence[str],
    test_drugs: Sequence[str],
) -> Dict:
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"{protocol} fold {fold_id} produced an empty split "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
        )

    entities = {
        "train_cells": sorted(map(str, train_cells)),
        "val_cells": sorted(map(str, val_cells)),
        "test_cells": sorted(map(str, test_cells)),
        "train_drugs": sorted(map(str, train_drugs)),
        "val_drugs": sorted(map(str, val_drugs)),
        "test_drugs": sorted(map(str, test_drugs)),
    }
    _validate_entities(protocol, entities)
    return {
        "fold": fold_id,
        "protocol": protocol,
        "train": canonicalize_response_pairs(train_df),
        "val": canonicalize_response_pairs(val_df),
        "test": canonicalize_response_pairs(test_df),
        "entities": entities,
    }


def _validate_entities(protocol: str, entities: Dict[str, List[str]]) -> None:
    def _assert_disjoint(first: str, second: str) -> None:
        overlap = set(entities[first]) & set(entities[second])
        if overlap:
            raise ValueError(f"{protocol} leakage between {first} and {second}: {sorted(overlap)[:5]}")

    if protocol in {PROTOCOL_UNSEEN_CELLS, PROTOCOL_UNSEEN_BOTH}:
        _assert_disjoint("train_cells", "val_cells")
        _assert_disjoint("train_cells", "test_cells")
        _assert_disjoint("val_cells", "test_cells")
    if protocol in {PROTOCOL_UNSEEN_DRUGS, PROTOCOL_UNSEEN_BOTH}:
        _assert_disjoint("train_drugs", "val_drugs")
        _assert_disjoint("train_drugs", "test_drugs")
        _assert_disjoint("val_drugs", "test_drugs")


def create_fusecdr_folds(
    response_pairs: pd.DataFrame,
    seed: int = 0,
    n_splits: int = 5,
    val_ratio_of_full: float = 0.1,
) -> List[Dict[str, pd.DataFrame]]:
    canonical = canonicalize_response_pairs(response_pairs)
    if canonical.empty:
        raise ValueError("No response pairs available for fold generation")

    allpairs = canonical[["cell_id", "drug_id", "label"]].to_numpy(dtype=object)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[Dict[str, pd.DataFrame]] = []

    for fold_id, (train_val_idx, test_idx) in enumerate(splitter.split(allpairs), start=1):
        test_df = canonical.iloc[test_idx].reset_index(drop=True)

        val_size = int(len(allpairs) * val_ratio_of_full)
        val_size = max(1, min(val_size, len(train_val_idx) - 1))
        rng = np.random.RandomState(seed + fold_id * 1000)  # fold-dependent seed for independent val draws
        perm = rng.permutation(len(train_val_idx))
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        train_df = canonical.iloc[train_idx].reset_index(drop=True)
        val_df = canonical.iloc[val_idx].reset_index(drop=True)
        folds.append(
            _build_fold_record(
                fold_id=fold_id,
                protocol=PROTOCOL_RANDOM,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_cells=sorted(train_df["cell_id"].unique()),
                val_cells=sorted(val_df["cell_id"].unique()),
                test_cells=sorted(test_df["cell_id"].unique()),
                train_drugs=sorted(train_df["drug_id"].unique()),
                val_drugs=sorted(val_df["drug_id"].unique()),
                test_drugs=sorted(test_df["drug_id"].unique()),
            )
        )
    return folds


def _create_unseen_cell_folds(
    canonical: pd.DataFrame,
    seed: int,
    n_splits: int,
    val_ratio_of_full: float,
) -> List[Dict]:
    cell_folds = _entity_split(canonical["cell_id"].unique(), seed, n_splits, val_ratio_of_full)
    all_drugs = sorted(canonical["drug_id"].unique().tolist())
    folds: List[Dict] = []
    for split in cell_folds:
        train_df = _pairs_for_entities(canonical, allowed_cells=split["train"], allowed_drugs=all_drugs)
        val_df = _pairs_for_entities(canonical, allowed_cells=split["val"], allowed_drugs=all_drugs)
        test_df = _pairs_for_entities(canonical, allowed_cells=split["test"], allowed_drugs=all_drugs)
        folds.append(
            _build_fold_record(
                fold_id=split["fold"],
                protocol=PROTOCOL_UNSEEN_CELLS,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_cells=split["train"],
                val_cells=split["val"],
                test_cells=split["test"],
                train_drugs=all_drugs,
                val_drugs=all_drugs,
                test_drugs=all_drugs,
            )
        )
    return folds


def _create_unseen_drug_folds(
    canonical: pd.DataFrame,
    seed: int,
    n_splits: int,
    val_ratio_of_full: float,
) -> List[Dict]:
    drug_folds = _entity_split(canonical["drug_id"].unique(), seed, n_splits, val_ratio_of_full)
    all_cells = sorted(canonical["cell_id"].unique().tolist())
    folds: List[Dict] = []
    for split in drug_folds:
        train_df = _pairs_for_entities(canonical, allowed_cells=all_cells, allowed_drugs=split["train"])
        val_df = _pairs_for_entities(canonical, allowed_cells=all_cells, allowed_drugs=split["val"])
        test_df = _pairs_for_entities(canonical, allowed_cells=all_cells, allowed_drugs=split["test"])
        folds.append(
            _build_fold_record(
                fold_id=split["fold"],
                protocol=PROTOCOL_UNSEEN_DRUGS,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_cells=all_cells,
                val_cells=all_cells,
                test_cells=all_cells,
                train_drugs=split["train"],
                val_drugs=split["val"],
                test_drugs=split["test"],
            )
        )
    return folds


def _create_unseen_both_folds(
    canonical: pd.DataFrame,
    seed: int,
    n_splits: int,
    val_ratio_of_full: float,
) -> List[Dict]:
    cell_folds = _entity_split(canonical["cell_id"].unique(), seed, n_splits, val_ratio_of_full)
    drug_folds = _entity_split(canonical["drug_id"].unique(), seed, n_splits, val_ratio_of_full)
    folds: List[Dict] = []
    for cell_split, drug_split in zip(cell_folds, drug_folds):
        train_df = _pairs_for_entities(canonical, allowed_cells=cell_split["train"], allowed_drugs=drug_split["train"])
        val_df = _pairs_for_entities(canonical, allowed_cells=cell_split["val"], allowed_drugs=drug_split["val"])
        test_df = _pairs_for_entities(canonical, allowed_cells=cell_split["test"], allowed_drugs=drug_split["test"])
        folds.append(
            _build_fold_record(
                fold_id=cell_split["fold"],
                protocol=PROTOCOL_UNSEEN_BOTH,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_cells=cell_split["train"],
                val_cells=cell_split["val"],
                test_cells=cell_split["test"],
                train_drugs=drug_split["train"],
                val_drugs=drug_split["val"],
                test_drugs=drug_split["test"],
            )
        )
    return folds


def create_protocol_folds(
    response_pairs: pd.DataFrame,
    protocol: str = PROTOCOL_RANDOM,
    seed: int = 0,
    n_splits: int = 5,
    val_ratio_of_full: float = 0.1,
) -> List[Dict]:
    canonical = canonicalize_response_pairs(response_pairs)
    if canonical.empty:
        raise ValueError("No response pairs available for fold generation")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol '{protocol}'. Expected one of {SUPPORTED_PROTOCOLS}")

    if protocol == PROTOCOL_RANDOM:
        return create_fusecdr_folds(canonical, seed=seed, n_splits=n_splits, val_ratio_of_full=val_ratio_of_full)
    if protocol == PROTOCOL_UNSEEN_CELLS:
        return _create_unseen_cell_folds(canonical, seed, n_splits, val_ratio_of_full)
    if protocol == PROTOCOL_UNSEEN_DRUGS:
        return _create_unseen_drug_folds(canonical, seed, n_splits, val_ratio_of_full)
    return _create_unseen_both_folds(canonical, seed, n_splits, val_ratio_of_full)


def save_protocol_folds(
    response_pairs: pd.DataFrame,
    output_dir: str,
    protocol: str = PROTOCOL_RANDOM,
    seed: int = 0,
    n_splits: int = 5,
    val_ratio_of_full: float = 0.1,
) -> str:
    ensure_dir(output_dir)
    folds = create_protocol_folds(
        response_pairs=response_pairs,
        protocol=protocol,
        seed=seed,
        n_splits=n_splits,
        val_ratio_of_full=val_ratio_of_full,
    )
    manifest = {
        "protocol": protocol,
        "seed": seed,
        "n_splits": n_splits,
        "val_ratio_of_full": val_ratio_of_full,
    }
    with open(os.path.join(output_dir, "split_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    for fold in folds:
        fold_dir = ensure_dir(os.path.join(output_dir, f"fold_{fold['fold']}"))
        fold["train"].to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        fold["val"].to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        fold["test"].to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        with open(os.path.join(fold_dir, "entities.json"), "w") as handle:
            json.dump(fold["entities"], handle, indent=2)
    return output_dir


def save_folds(response_pairs: pd.DataFrame, output_dir: str, seed: int = 0, n_splits: int = 5) -> str:
    return save_protocol_folds(
        response_pairs=response_pairs,
        output_dir=output_dir,
        protocol=PROTOCOL_RANDOM,
        seed=seed,
        n_splits=n_splits,
    )


def ensure_protocol_folds(
    response_pairs_path: str,
    output_dir: str,
    protocol: str = PROTOCOL_RANDOM,
    seed: int = 0,
    n_splits: int = 5,
    val_ratio_of_full: float = 0.1,
) -> str:
    expected = os.path.join(output_dir, "fold_1", "train.csv")
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    if os.path.exists(expected) and os.path.isfile(manifest_path):
        return output_dir
    response_pairs = pd.read_csv(response_pairs_path)
    return save_protocol_folds(
        response_pairs=response_pairs,
        output_dir=output_dir,
        protocol=protocol,
        seed=seed,
        n_splits=n_splits,
        val_ratio_of_full=val_ratio_of_full,
    )


def ensure_folds(response_pairs_path: str, output_dir: str, seed: int = 0, n_splits: int = 5) -> str:
    return ensure_protocol_folds(
        response_pairs_path=response_pairs_path,
        output_dir=output_dir,
        protocol=PROTOCOL_RANDOM,
        seed=seed,
        n_splits=n_splits,
    )


def list_fold_ids(output_dir: str) -> List[int]:
    if not os.path.isdir(output_dir):
        return []
    fold_ids = []
    for name in os.listdir(output_dir):
        if not name.startswith("fold_"):
            continue
        try:
            fold_ids.append(int(name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(fold_ids)


def load_split_manifest(output_dir: str) -> Dict:
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    if not os.path.isfile(manifest_path):
        return {"protocol": PROTOCOL_RANDOM, "seed": 0, "n_splits": len(list_fold_ids(output_dir))}
    with open(manifest_path) as handle:
        return json.load(handle)


def load_fold(output_dir: str, fold: int) -> Dict[str, pd.DataFrame]:
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    return {
        "train": pd.read_csv(os.path.join(fold_dir, "train.csv")),
        "val": pd.read_csv(os.path.join(fold_dir, "val.csv")),
        "test": pd.read_csv(os.path.join(fold_dir, "test.csv")),
    }


def load_fold_bundle(output_dir: str, fold: int) -> Dict:
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    entities_path = os.path.join(fold_dir, "entities.json")
    entities = {}
    if os.path.isfile(entities_path):
        with open(entities_path) as handle:
            entities = json.load(handle)
    bundle = load_fold(output_dir, fold)
    bundle["entities"] = entities
    bundle["manifest"] = load_split_manifest(output_dir)
    return bundle
