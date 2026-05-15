import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dataset_claim_utils import ensure_dir
from flexibility_utils import (
    FINAL_DATASET_DIR,
    FLEXIBLE_DIR,
    canonicalize_response_pairs,
    read_json,
    save_flexible_random_folds,
    write_json,
)


HGT_CLAIM_OUTPUTS_DIR = FLEXIBLE_DIR / "hgt_claim_outputs"
HGT_CLAIM_VIEWS_DIR = FLEXIBLE_DIR / "hgt_claim_views"
HGT_CLAIM_SPLITS_DIR = HGT_CLAIM_OUTPUTS_DIR / "shared_splits"


def default_split_dir(dataset_root: Path | str, seed: int, k_fold: int) -> Path:
    dataset_name = Path(dataset_root).resolve().name
    return HGT_CLAIM_SPLITS_DIR / f"{dataset_name}_seed{seed}_k{k_fold}"


def ensure_hgt_claim_split_dir(
    *,
    dataset_root: Path | str = FINAL_DATASET_DIR,
    split_dir: Path | str | None = None,
    seed: int = 0,
    k_fold: int = 5,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    resolved_split_dir = Path(split_dir).resolve() if split_dir else default_split_dir(dataset_root, seed, k_fold)
    manifest_path = resolved_split_dir / "split_manifest.json"
    expected_fold = resolved_split_dir / "fold_1" / "train.csv"
    if manifest_path.is_file() and expected_fold.is_file():
        return resolved_split_dir
    ensure_dir(resolved_split_dir)
    save_flexible_random_folds(
        response_pairs_path=dataset_root / "response_pairs.csv",
        output_dir=resolved_split_dir,
        seed=seed,
        n_splits=k_fold,
    )
    return resolved_split_dir


def load_fold_tables(split_dir: Path | str, fold_id: int) -> Dict[str, pd.DataFrame]:
    fold_dir = Path(split_dir) / f"fold_{fold_id}"
    return {
        "train": canonicalize_response_pairs(pd.read_csv(fold_dir / "train.csv")),
        "val": canonicalize_response_pairs(pd.read_csv(fold_dir / "val.csv")),
        "test": canonicalize_response_pairs(pd.read_csv(fold_dir / "test.csv")),
    }


def save_prediction_rows(path: Path | str, rows: List[Dict]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_prediction_rows(path: Path | str) -> List[Dict]:
    return pd.read_csv(path).to_dict(orient="records")


def save_fold_result(
    *,
    results_dir: Path | str,
    fold_id: int,
    metrics: Dict,
    prediction_rows: List[Dict],
    extra_payloads: Dict[str, pd.DataFrame] | None = None,
) -> None:
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    save_prediction_rows(results_dir / f"fold_{fold_id}_predictions.csv", prediction_rows)
    write_json(results_dir / f"fold_{fold_id}_metrics.json", metrics)
    if extra_payloads:
        for name, frame in extra_payloads.items():
            frame.to_csv(results_dir / f"fold_{fold_id}_{name}.csv", index=False)


def load_completed_fold_metrics(results_dir: Path | str) -> List[Dict]:
    results_dir = Path(results_dir)
    completed = []
    if not results_dir.is_dir():
        return completed
    for metrics_path in sorted(results_dir.glob("fold_*_metrics.json")):
        completed.append(read_json(metrics_path))
    completed.sort(key=lambda row: int(row["fold"]))
    return completed


def fold_is_complete(results_dir: Path | str, fold_id: int, extra_required_suffixes: Sequence[str] | None = None) -> bool:
    results_dir = Path(results_dir)
    required = [
        results_dir / f"fold_{fold_id}_metrics.json",
        results_dir / f"fold_{fold_id}_predictions.csv",
    ]
    for suffix in extra_required_suffixes or []:
        required.append(results_dir / f"fold_{fold_id}_{suffix}.csv")
    return all(path.is_file() for path in required)


def save_fold_metrics_table(results_dir: Path | str, fold_metrics: List[Dict]) -> Path:
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    frame = pd.DataFrame(sorted(fold_metrics, key=lambda row: int(row["fold"])))
    frame.to_csv(results_dir / "fold_metrics.csv", index=False)
    return results_dir / "fold_metrics.csv"


def save_summary(results_dir: Path | str, fold_metrics: List[Dict], metadata: Dict) -> Dict:
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    frame = pd.DataFrame(sorted(fold_metrics, key=lambda row: int(row["fold"])))
    numeric_cols = [col for col in ["test_auc", "test_aupr", "test_f1", "test_acc"] if col in frame.columns]
    mean = {col: float(frame[col].mean()) for col in numeric_cols} if not frame.empty else {}
    std = {col: float(frame[col].std(ddof=0)) for col in numeric_cols} if not frame.empty else {}
    summary = {
        "mean": mean,
        "std": std,
        "fold_count": int(len(frame)),
        "folds": frame.to_dict(orient="records"),
        "metadata": metadata,
    }
    write_json(results_dir / "summary.json", summary)
    return summary


def save_experiment_progress(
    *,
    experiment_dir: Path | str,
    rows: List[Dict],
    metadata: Dict,
    results_name: str = "results.csv",
    sort_metric: str = "auc",
) -> None:
    experiment_dir = Path(experiment_dir)
    ensure_dir(experiment_dir)
    frame = pd.DataFrame(rows)
    if not frame.empty and sort_metric in frame.columns:
        frame = frame.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    frame.to_csv(experiment_dir / results_name, index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            **metadata,
            "completed_runs": int(len(frame)),
            "results_csv": str(experiment_dir / results_name),
        },
    )


def load_experiment_rows(path: Path | str) -> List[Dict]:
    path = Path(path)
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def sorted_entity_ids(data_new: Sequence[Tuple[str, str, int]]) -> Tuple[List[str], List[str]]:
    return (
        sorted({item[0] for item in data_new}),
        sorted({item[1] for item in data_new}),
    )


def build_prediction_rows_from_mask(
    *,
    cell_ids: Sequence[str],
    drug_ids: Sequence[str],
    flat_mask: np.ndarray,
    flat_labels: np.ndarray,
    flat_predictions: np.ndarray,
) -> List[Dict]:
    rows: List[Dict] = []
    nb_drugs = len(drug_ids)
    for flat_idx, keep in enumerate(flat_mask.tolist()):
        if not keep:
            continue
        cell_idx = flat_idx // nb_drugs
        drug_idx = flat_idx % nb_drugs
        rows.append(
            {
                "cell_id": str(cell_ids[cell_idx]),
                "drug_id": str(drug_ids[drug_idx]),
                "label": int(flat_labels[flat_idx]),
                "prediction": float(flat_predictions[flat_idx]),
            }
        )
    return rows


def build_prediction_rows_from_pairs(
    *,
    pairs_df: pd.DataFrame,
    predictions: Sequence[float],
) -> List[Dict]:
    rows = []
    for row, prediction in zip(pairs_df.itertuples(index=False), predictions):
        rows.append(
            {
                "cell_id": str(row.cell_id),
                "drug_id": str(row.drug_id),
                "label": int(row.label),
                "prediction": float(prediction),
            }
        )
    return rows


def sparsify_positive_train_pairs(
    train_pairs: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if fraction <= 0:
        return canonicalize_response_pairs(train_pairs), train_pairs.iloc[0:0].copy()
    train_pairs = canonicalize_response_pairs(train_pairs)
    pos = train_pairs[train_pairs["label"] == 1].copy()
    if pos.empty:
        return train_pairs, pos
    withhold_count = int(round(len(pos) * fraction))
    withhold_count = max(1, min(withhold_count, len(pos) - 1)) if len(pos) > 1 else 0
    if withhold_count <= 0:
        return train_pairs, pos.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    withheld_indices = rng.choice(pos.index.to_numpy(), size=withhold_count, replace=False)
    withheld = train_pairs.loc[withheld_indices].copy().reset_index(drop=True)
    observed = train_pairs.drop(withheld_indices).reset_index(drop=True)
    return canonicalize_response_pairs(observed), canonicalize_response_pairs(withheld)


def sample_negative_pairs(
    train_pairs: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> pd.DataFrame:
    train_pairs = canonicalize_response_pairs(train_pairs)
    negatives = train_pairs[train_pairs["label"] == 0].copy()
    if negatives.empty or count <= 0:
        return negatives.iloc[0:0].copy()
    count = min(count, len(negatives))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(negatives.index.to_numpy(), size=count, replace=False)
    return canonicalize_response_pairs(negatives.loc[sampled_idx].copy().reset_index(drop=True))


def empty_edge_index(device) -> "torch.Tensor":
    import torch

    return torch.empty((2, 0), dtype=torch.long, device=device)


def filter_similarity_edges(
    edge_index_dict: Dict[Tuple[str, str, str], "torch.Tensor"],
    *,
    mode: str,
    device,
) -> Dict[Tuple[str, str, str], "torch.Tensor"]:
    filtered = {
        edge_type: edge_index.clone()
        for edge_type, edge_index in edge_index_dict.items()
    }
    if mode == "full_graph":
        return filtered
    if mode == "no_drug_similarity":
        filtered[("drug", "similar_to", "drug")] = empty_edge_index(device)
        return filtered
    if mode == "no_cell_similarity":
        filtered[("cell", "similar_to", "cell")] = empty_edge_index(device)
        return filtered
    if mode == "response_only":
        filtered[("drug", "similar_to", "drug")] = empty_edge_index(device)
        filtered[("cell", "similar_to", "cell")] = empty_edge_index(device)
        return filtered
    raise ValueError(f"Unsupported edge-ablation mode: {mode}")


def _neighbor_sets_from_similarity(frame: pd.DataFrame, top_k: int) -> Dict[str, set]:
    values = frame.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = values / norms
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    if n <= 1:
        return {str(frame.index[i]): set() for i in range(n)}
    k = max(1, min(top_k, n - 1))
    topk_indices = np.argsort(sim, axis=1)[:, -k:]
    return {
        str(frame.index[i]): {str(frame.index[j]) for j in topk_indices[i].tolist()}
        for i in range(n)
    }


def compute_path_support_rows(
    *,
    train_pairs: pd.DataFrame,
    eval_pairs: pd.DataFrame,
    cell_similarity_df: pd.DataFrame,
    drug_similarity_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    train_pairs = canonicalize_response_pairs(train_pairs)
    eval_pairs = canonicalize_response_pairs(eval_pairs)
    pos_train = train_pairs[train_pairs["label"] == 1]

    cell_to_drugs: Dict[str, set] = {}
    drug_to_cells: Dict[str, set] = {}
    for row in pos_train.itertuples(index=False):
        cell_to_drugs.setdefault(str(row.cell_id), set()).add(str(row.drug_id))
        drug_to_cells.setdefault(str(row.drug_id), set()).add(str(row.cell_id))

    cell_neighbors = _neighbor_sets_from_similarity(cell_similarity_df, top_k)
    drug_neighbors = _neighbor_sets_from_similarity(drug_similarity_df, top_k)

    rows = []
    for row in eval_pairs.itertuples(index=False):
        cell_id = str(row.cell_id)
        drug_id = str(row.drug_id)
        sim_drugs = drug_neighbors.get(drug_id, set())
        sim_cells = cell_neighbors.get(cell_id, set())
        drug_2hop = any(sim_drug in cell_to_drugs.get(cell_id, set()) for sim_drug in sim_drugs)
        cell_2hop = any(sim_cell in drug_to_cells.get(drug_id, set()) for sim_cell in sim_cells)

        three_hop = False
        if not drug_2hop and not cell_2hop:
            for sim_drug in sim_drugs:
                if three_hop:
                    break
                candidate_cells = drug_to_cells.get(sim_drug, set())
                if any(candidate_cell in sim_cells for candidate_cell in candidate_cells):
                    three_hop = True
                    break

        if drug_2hop and cell_2hop:
            bucket = "both_2hop"
        elif drug_2hop:
            bucket = "drug_2hop"
        elif cell_2hop:
            bucket = "cell_2hop"
        elif three_hop:
            bucket = "three_hop_only"
        else:
            bucket = "no_support"

        rows.append(
            {
                "cell_id": cell_id,
                "drug_id": drug_id,
                "label": int(row.label),
                "bucket": bucket,
                "drug_2hop": bool(drug_2hop),
                "cell_2hop": bool(cell_2hop),
                "three_hop_only": bool(three_hop),
            }
        )
    return pd.DataFrame(rows)
