from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from benchmarking_common.metrics import compute_binary_metrics


EdgeType = Tuple[str, str, str]
DRUG_SIMILARITY: EdgeType = ("drug", "similar_to", "drug")
CELL_SIMILARITY: EdgeType = ("cell", "similar_to", "cell")
DRUG_RESPONSE: EdgeType = ("drug", "responds_to", "cell")
DISTANCE_BUCKETS = ("exact_2_hop", "exact_3_hop")


def _edge_rows(edge_index: torch.Tensor | np.ndarray | None) -> Iterable[Tuple[int, int]]:
    if edge_index is None:
        return []
    if torch.is_tensor(edge_index):
        values = edge_index.detach().cpu().numpy()
    else:
        values = np.asarray(edge_index)
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], found {values.shape}")
    return ((int(source), int(target)) for source, target in values.T)


def build_typed_adjacency(
    edge_index_dict: Mapping[EdgeType, torch.Tensor | np.ndarray],
    *,
    num_drugs: int,
    num_cells: int,
) -> List[Tuple[int, ...]]:
    """Build one directed adjacency list with drugs followed by cells."""
    if num_drugs < 1 or num_cells < 1:
        raise ValueError("num_drugs and num_cells must both be positive")

    adjacency = [set() for _ in range(num_drugs + num_cells)]

    for source, target in _edge_rows(edge_index_dict.get(DRUG_SIMILARITY)):
        if not (0 <= source < num_drugs and 0 <= target < num_drugs):
            raise IndexError("Drug-similarity edge contains an out-of-range node index")
        adjacency[source].add(target)

    for source, target in _edge_rows(edge_index_dict.get(DRUG_RESPONSE)):
        if not (0 <= source < num_drugs and 0 <= target < num_cells):
            raise IndexError("Drug-response edge contains an out-of-range node index")
        adjacency[source].add(num_drugs + target)

    for source, target in _edge_rows(edge_index_dict.get(CELL_SIMILARITY)):
        if not (0 <= source < num_cells and 0 <= target < num_cells):
            raise IndexError("Cell-similarity edge contains an out-of-range node index")
        adjacency[num_drugs + source].add(num_drugs + target)

    return [tuple(sorted(neighbors)) for neighbors in adjacency]


def bounded_shortest_distances(
    adjacency: Sequence[Sequence[int]],
    source: int,
    *,
    max_distance: int,
) -> Dict[int, int]:
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    if source < 0 or source >= len(adjacency):
        raise IndexError(f"Source node {source} is outside the graph")

    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        distance = distances[node]
        if distance >= max_distance:
            continue
        for neighbor in adjacency[node]:
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append(neighbor)
    return distances


def distance_bucket(distance: int | None) -> str:
    if distance in (2, 3):
        return f"exact_{distance}_hop"
    if distance == 1:
        return "direct_1_hop"
    return "other"


def annotate_pair_distances(
    pairs: pd.DataFrame,
    *,
    drug_ids: Sequence[str],
    cell_ids: Sequence[str],
    edge_index_dict: Mapping[EdgeType, torch.Tensor | np.ndarray],
    max_distance: int = 3,
) -> pd.DataFrame:
    required = {"drug_id", "cell_id"}
    missing = required - set(pairs.columns)
    if missing:
        raise KeyError(f"Pair table is missing columns: {sorted(missing)}")

    normalized = pairs.copy()
    normalized["drug_id"] = normalized["drug_id"].astype(str)
    normalized["cell_id"] = normalized["cell_id"].astype(str)
    normalized_drugs = [str(value) for value in drug_ids]
    normalized_cells = [str(value) for value in cell_ids]
    drug_to_index = {value: index for index, value in enumerate(normalized_drugs)}
    cell_to_index = {value: index for index, value in enumerate(normalized_cells)}

    unknown_drugs = sorted(set(normalized["drug_id"]) - set(drug_to_index))
    unknown_cells = sorted(set(normalized["cell_id"]) - set(cell_to_index))
    if unknown_drugs or unknown_cells:
        raise KeyError(
            "Pair table contains graph-external identifiers: "
            f"drugs={unknown_drugs[:5]}, cells={unknown_cells[:5]}"
        )

    adjacency = build_typed_adjacency(
        edge_index_dict,
        num_drugs=len(normalized_drugs),
        num_cells=len(normalized_cells),
    )
    distance_cache: Dict[str, Dict[int, int]] = {}
    for drug_id in sorted(normalized["drug_id"].unique()):
        distance_cache[drug_id] = bounded_shortest_distances(
            adjacency,
            drug_to_index[drug_id],
            max_distance=max_distance,
        )

    distances: List[int | None] = []
    for row in normalized.itertuples(index=False):
        target = len(normalized_drugs) + cell_to_index[str(row.cell_id)]
        distances.append(distance_cache[str(row.drug_id)].get(target))

    normalized["directed_distance"] = pd.array(distances, dtype="Int64")
    normalized["distance_bucket"] = [distance_bucket(value) for value in distances]
    return normalized


def assert_response_edges_exclude_pairs(
    pairs: pd.DataFrame,
    *,
    drug_ids: Sequence[str],
    cell_ids: Sequence[str],
    response_edge_index: torch.Tensor | np.ndarray,
) -> None:
    drug_to_index = {str(value): index for index, value in enumerate(drug_ids)}
    cell_to_index = {str(value): index for index, value in enumerate(cell_ids)}
    response_edges = set(_edge_rows(response_edge_index))
    overlaps = []
    for row in pairs.itertuples(index=False):
        edge = (drug_to_index[str(row.drug_id)], cell_to_index[str(row.cell_id)])
        if edge in response_edges:
            overlaps.append((str(row.drug_id), str(row.cell_id)))
    if overlaps:
        raise ValueError(
            "Evaluation response edges leaked into the propagation graph: "
            f"{overlaps[:5]}"
        )


def distance_metric_rows(
    annotated_pairs: pd.DataFrame,
    *,
    dataset: str,
    variant: str,
    local_layers: int,
    global_layers: int,
    fold: int,
) -> List[Dict[str, float | int | str]]:
    required = {"label", "prediction", "distance_bucket"}
    missing = required - set(annotated_pairs.columns)
    if missing:
        raise KeyError(f"Annotated pair table is missing columns: {sorted(missing)}")

    rows: List[Dict[str, float | int | str]] = []
    for bucket in DISTANCE_BUCKETS:
        subset = annotated_pairs[annotated_pairs["distance_bucket"] == bucket]
        labels = subset["label"].to_numpy(dtype=np.int64)
        positive_count = int(labels.sum())
        negative_count = int(len(labels) - positive_count)
        eligible = positive_count > 0 and negative_count > 0
        metrics = (
            compute_binary_metrics(labels, subset["prediction"].to_numpy(dtype=np.float32))
            if eligible
            else {"auc": np.nan, "aupr": np.nan, "f1": np.nan, "acc": np.nan}
        )
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "local_layers": int(local_layers),
                "global_layers": int(global_layers),
                "fold": int(fold),
                "distance_bucket": bucket,
                "pair_count": int(len(subset)),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "eligible": int(eligible),
                "auc": float(metrics["auc"]),
                "aupr": float(metrics["aupr"]),
                "f1": float(metrics["f1"]),
                "acc": float(metrics["acc"]),
            }
        )
    return rows
