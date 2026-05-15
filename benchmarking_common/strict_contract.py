from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch


STRICT_BENCHMARK_NAME = "3OmicsStrictBenchmarking"
STRICT_ENABLED_MODELS = (
    "FUSECDR",
    "GraphCDR",
    "RedCDR",
    "GADRP",
    "DeepTTC",
    "GraphDRP",
)
STRICT_DISABLED_MODELS: tuple[str, ...] = ()
STRICT_PREDICTIVE_INPUTS = (
    "genomics_mutation.csv",
    "transcriptomics_expression.csv",
    "epigenomics_methylation.csv",
    "drug_graph_feat/",
)
STRICT_GRAPH_INPUTS = (
    "similarity.csv",
    "physicochemical.csv",
    "train_pairs",
)


def validate_strict_prepared_metadata(prepared_dir: str, metadata: dict) -> None:
    if metadata.get("benchmark") != STRICT_BENCHMARK_NAME:
        raise ValueError(
            f"{prepared_dir} is not tagged as {STRICT_BENCHMARK_NAME}. "
            f"Found benchmark={metadata.get('benchmark')!r}."
        )
    predictive_inputs = tuple(metadata.get("strict_predictive_inputs", []))
    if predictive_inputs != STRICT_PREDICTIVE_INPUTS:
        raise ValueError(
            f"{prepared_dir} has stale strict predictive-input metadata. "
            f"Expected {STRICT_PREDICTIVE_INPUTS}, found {predictive_inputs}."
        )
    graph_inputs = tuple(metadata.get("strict_graph_inputs", []))
    if graph_inputs != STRICT_GRAPH_INPUTS:
        raise ValueError(
            f"{prepared_dir} has stale strict graph-input metadata. "
            f"Expected {STRICT_GRAPH_INPUTS}, found {graph_inputs}."
        )
    if metadata.get("response_graph_source") != "train_pairs":
        raise ValueError(
            f"{prepared_dir} must declare response_graph_source='train_pairs'. "
            f"Found {metadata.get('response_graph_source')!r}."
        )

    declared_models = tuple(metadata.get("models", []))
    if tuple(sorted(declared_models)) != tuple(sorted(STRICT_ENABLED_MODELS)):
        raise ValueError(
            f"{prepared_dir} declares stale strict models. "
            f"Expected {STRICT_ENABLED_MODELS}, found {declared_models}."
        )

    disabled = tuple(metadata.get("disabled_models_pending_strict_alignment", []))
    if tuple(sorted(disabled)) != tuple(sorted(STRICT_DISABLED_MODELS)):
        raise ValueError(
            f"{prepared_dir} declares stale disabled strict models. "
            f"Expected {STRICT_DISABLED_MODELS}, found {disabled}."
        )


def validate_strict_model_contract(
    model_name: str,
    *,
    predictive_inputs: Sequence[str],
    graph_inputs: Sequence[str],
) -> None:
    if tuple(sorted(predictive_inputs)) != tuple(sorted(STRICT_PREDICTIVE_INPUTS)):
        raise ValueError(
            f"{model_name} violates the strict predictive-input contract. "
            f"Expected {STRICT_PREDICTIVE_INPUTS}, found {tuple(predictive_inputs)}."
        )
    if tuple(sorted(graph_inputs)) != tuple(sorted(STRICT_GRAPH_INPUTS)):
        raise ValueError(
            f"{model_name} violates the strict graph-input contract. "
            f"Expected {STRICT_GRAPH_INPUTS}, found {tuple(graph_inputs)}."
        )


def _expected_pairs_for_scope(
    cell_ids: Sequence[str],
    drug_ids: Sequence[str],
    train_pairs: pd.DataFrame,
) -> set[tuple[str, str]]:
    cell_set = set(map(str, cell_ids))
    drug_set = set(map(str, drug_ids))
    expected: set[tuple[str, str]] = set()
    for row in train_pairs.itertuples(index=False):
        cell_id = str(row.cell_id)
        drug_id = str(row.drug_id)
        if cell_id in cell_set and drug_id in drug_set:
            expected.add((cell_id, drug_id))
    return expected


def validate_strict_response_edge_index(
    *,
    cell_ids: Sequence[str],
    drug_ids: Sequence[str],
    train_pairs: pd.DataFrame,
    response_edge_index: torch.Tensor,
) -> None:
    expected = _expected_pairs_for_scope(cell_ids, drug_ids, train_pairs)
    edge_np = response_edge_index.detach().cpu().numpy()
    if edge_np.ndim != 2 or edge_np.shape[0] != 2:
        raise ValueError(
            f"Strict response-edge index must have shape (2, E). "
            f"Found {tuple(edge_np.shape)}."
        )
    actual = {
        (str(cell_ids[int(src)]), str(drug_ids[int(dst)]))
        for src, dst in edge_np.T.tolist()
    }
    if actual != expected:
        raise ValueError(
            "Strict response-edge leakage or drift detected. "
            f"Expected {len(expected)} train pairs, found {len(actual)}."
        )
    if edge_np.shape[1] != len(expected):
        raise ValueError(
            "Strict response-edge graph contains duplicate or missing edges. "
            f"Expected {len(expected)} edges, found {edge_np.shape[1]}."
        )


def validate_strict_train_edge_array(
    *,
    cell_ids: Sequence[str],
    drug_ids: Sequence[str],
    train_pairs: pd.DataFrame,
    train_edge: np.ndarray | torch.Tensor,
    mirrored: bool,
) -> None:
    expected = _expected_pairs_for_scope(cell_ids, drug_ids, train_pairs)
    if isinstance(train_edge, torch.Tensor):
        edge_np = train_edge.detach().cpu().numpy()
    else:
        edge_np = np.asarray(train_edge)
    if edge_np.ndim != 2 or edge_np.shape[1] < 2:
        raise ValueError(
            f"Strict train-edge array must have shape (E, >=2). Found {tuple(edge_np.shape)}."
        )

    decoded: list[tuple[str, str]] = []
    boundary = len(cell_ids)
    for src, dst in edge_np[:, :2].tolist():
        if src < boundary and dst >= boundary:
            decoded.append((str(cell_ids[int(src)]), str(drug_ids[int(dst - boundary)])))
        elif dst < boundary and src >= boundary:
            decoded.append((str(cell_ids[int(dst)]), str(drug_ids[int(src - boundary)])))
        else:
            raise ValueError(
                "Strict train-edge array contains an invalid edge orientation. "
                f"Found src={src}, dst={dst}, boundary={boundary}."
            )

    actual = set(decoded)
    if actual != expected:
        raise ValueError(
            "Strict train-edge array drift detected. "
            f"Expected {len(expected)} train pairs, found {len(actual)}."
        )
    expected_count = len(expected) * (2 if mirrored else 1)
    if len(decoded) != expected_count:
        raise ValueError(
            "Strict train-edge array contains duplicate or missing mirrored edges. "
            f"Expected {expected_count} rows, found {len(decoded)}."
        )


def validate_required_strict_files(prepared_dir: str) -> None:
    missing = []
    for relative_path in STRICT_PREDICTIVE_INPUTS[:-1] + STRICT_GRAPH_INPUTS[:-1]:
        if not os.path.isfile(os.path.join(prepared_dir, relative_path)):
            missing.append(relative_path)
    if not os.path.isdir(os.path.join(prepared_dir, "drug_graph_feat")):
        missing.append("drug_graph_feat/")
    if missing:
        raise FileNotFoundError(
            f"{prepared_dir} is missing required strict benchmark inputs: {sorted(missing)}"
        )
