import copy
import os
from typing import Dict, List

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
    validate_strict_train_edge_array,
)
from benchmark_wrappers.common import (
    build_flat_mask,
    build_prediction_rows_from_mask,
    build_pyg_graphs,
    build_shared_similarity_graphs,
    build_strict_train_edge_array,
    load_external_module,
    load_fold_bundle_tables,
    load_hkl_features,
    load_prepared_dataset,
    resolve_fold_ids,
    scope_entities_for_split,
    sorted_cell_drug_ids,
)


def _load_redcdr_model(root_dir: str):
    model_dir = os.path.join(root_dir, "benchmark models", "RedCDR-main")
    return load_external_module(
        module_name="benchmark_redcdr_shared_graph_model",
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


def _build_scope(
    dataset,
    graph_dir: str,
    scope_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    device: torch.device,
    top_k: int,
) -> Dict:
    cell_ids, drug_ids = sorted_cell_drug_ids(scope_pairs)
    mutation_df = dataset["tables"]["genomics_mutation"].loc[cell_ids]
    expression_df = dataset["tables"]["transcriptomics_expression"].loc[cell_ids]
    methylation_df = dataset["tables"]["epigenomics_methylation"].loc[cell_ids]
    similarity_df = dataset["tables"]["similarity"].loc[cell_ids]
    physicochemical_df = dataset["tables"]["physicochemical"].loc[drug_ids]

    mutation = torch.from_numpy(mutation_df.to_numpy(dtype="float32")).unsqueeze(1).unsqueeze(1).to(device)
    gexpr = torch.from_numpy(expression_df.to_numpy(dtype="float32")).to(device)
    methylation = torch.from_numpy(methylation_df.to_numpy(dtype="float32")).to(device)

    graph_features = load_hkl_features(graph_dir, drug_ids)
    graphs = build_pyg_graphs(drug_ids, graph_features)
    drug_batch = Batch.from_data_list(graphs).to(device)
    atom_shape = graphs[0].x.shape[-1]

    graph_edge_index_dict = build_shared_similarity_graphs(
        cell_similarity_features=similarity_df.to_numpy(dtype=np.float32),
        drug_similarity_features=physicochemical_df.to_numpy(dtype=np.float32),
        top_k=top_k,
        device=device,
    )

    label_pos = torch.zeros(len(cell_ids) * len(drug_ids), dtype=torch.float32)
    positive_pairs = scope_pairs[scope_pairs["label"] == 1]
    if not positive_pairs.empty:
        label_pos = torch.from_numpy(
            (
                pd.crosstab(positive_pairs["cell_id"], positive_pairs["drug_id"])
                .reindex(index=cell_ids, columns=drug_ids, fill_value=0)
                .to_numpy(dtype=np.float32)
            ).reshape(-1)
        )

    train_mask = build_flat_mask(cell_ids, drug_ids, train_pairs)
    val_mask = build_flat_mask(cell_ids, drug_ids, val_pairs)
    test_mask = build_flat_mask(cell_ids, drug_ids, test_pairs)

    train_edge = build_strict_train_edge_array(
        cell_ids,
        drug_ids,
        train_pairs,
        mirror=True,
        sort_by_label=True,
    )
    validate_strict_train_edge_array(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        train_pairs=train_pairs,
        train_edge=train_edge,
        mirrored=True,
    )

    return {
        "cell_ids": cell_ids,
        "drug_ids": drug_ids,
        "expression_df": expression_df,
        "methylation_df": methylation_df,
        "mutation": mutation,
        "gexpr": gexpr,
        "methylation": methylation,
        "drug_batch": drug_batch,
        "atom_shape": atom_shape,
        "label_pos": label_pos.to(device),
        "train_mask": train_mask.to(device),
        "val_mask": val_mask.to(device),
        "test_mask": test_mask.to(device),
        "train_edge": train_edge,
        "cell_similarity_edge": graph_edge_index_dict[("cell", "similar_to", "cell")],
        "drug_similarity_edge": graph_edge_index_dict[("drug", "similar_to", "drug")],
    }


def _build_model(module, scope: Dict, dim_feat: int, numk: int, layers: int, dropout: float, alpha: float, device_str: str):
    return module.RedCDR(
        scope["atom_shape"],
        [256, 256, 256],
        scope["expression_df"].shape[-1],
        scope["methylation_df"].shape[-1],
        dim_feat,
        100,
        len(scope["cell_ids"]),
        len(scope["drug_ids"]),
        numk,
        layers,
        dropout,
        alpha,
        True,
        True,
        False,
        0.2,
        device_str,
    )


def _copy_train_state_to_eval(train_model, eval_model, train_scope: Dict, eval_scope: Dict) -> None:
    eval_state = eval_model.state_dict()
    train_state = train_model.state_dict()
    for key, value in train_state.items():
        if key in {"C_emb", "D_emb"}:
            continue
        if key in eval_state and eval_state[key].shape == value.shape:
            eval_state[key] = value.detach().clone()
    eval_model.load_state_dict(eval_state, strict=False)

    eval_cell_map = {cell_id: idx for idx, cell_id in enumerate(eval_scope["cell_ids"])}
    eval_drug_map = {drug_id: idx for idx, drug_id in enumerate(eval_scope["drug_ids"])}
    with torch.no_grad():
        for idx, cell_id in enumerate(train_scope["cell_ids"]):
            target_idx = eval_cell_map.get(cell_id)
            if target_idx is not None:
                eval_model.C_emb[target_idx].copy_(train_model.C_emb[idx].detach())
        for idx, drug_id in enumerate(train_scope["drug_ids"]):
            target_idx = eval_drug_map.get(drug_id)
            if target_idx is not None:
                eval_model.D_emb[target_idx].copy_(train_model.D_emb[idx].detach())


def _forward_predictions(model, scope: Dict):
    predictions, rd_loss, pd_loss = model(
        scope["drug_batch"].x,
        scope["drug_batch"].edge_index,
        scope["drug_batch"].ptr,
        scope["mutation"],
        scope["gexpr"],
        scope["methylation"],
        scope["train_edge"],
        scope["cell_similarity_edge"],
        scope["drug_similarity_edge"],
    )
    return predictions, rd_loss, pd_loss


def _sanitize_predictions(predictions: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 400,
    lr: float = 0.001,
    dropout: float = 0.4,
    numk: int = 5,
    dim_feat: int = 100,
    layers: int = 2,
    rd: float = 0.5,
    pd_weight: float = 1.5,
    alpha: float = 8.0,
    top_k: int = 10,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)
    runtime_device_str = runtime_device.type if runtime_device.index is None else f"{runtime_device.type}:{runtime_device.index}"
    module = _load_redcdr_model(root_dir)
    dataset = load_prepared_dataset(prepared_dir)
    validate_strict_prepared_metadata(prepared_dir, dataset["metadata"])
    validate_strict_model_contract(
        "RedCDR",
        predictive_inputs=(
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ),
        graph_inputs=("similarity.csv", "physicochemical.csv", "train_pairs"),
    )
    graph_dir = os.path.join(prepared_dir, "drug_graph_feat")

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
        protocol = bundle["manifest"].get("protocol", "random")
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == "random":
            full_pairs = dataset["response_pairs"]
            train_scope = _build_scope(dataset, graph_dir, full_pairs, train_pairs, val_pairs, test_pairs, runtime_device, top_k)
            val_scope = train_scope
            test_scope = train_scope
        else:
            scope_entities_for_split(bundle, "train")
            scope_entities_for_split(bundle, "val")
            scope_entities_for_split(bundle, "test")

            train_scope = _build_scope(dataset, graph_dir, train_pairs, train_pairs, _empty_pairs(), _empty_pairs(), runtime_device, top_k)
            val_scope = _build_scope(
                dataset,
                graph_dir,
                _concat_pairs(train_pairs, val_pairs),
                train_pairs,
                val_pairs,
                _empty_pairs(),
                runtime_device,
                top_k,
            )
            test_scope = _build_scope(
                dataset,
                graph_dir,
                _concat_pairs(train_pairs, test_pairs),
                train_pairs,
                _empty_pairs(),
                test_pairs,
                runtime_device,
                top_k,
            )

        train_model = _build_model(module, train_scope, dim_feat, numk, layers, dropout, alpha, runtime_device_str).to(runtime_device)
        optimizer = torch.optim.Adam(train_model.parameters(), lr=lr, weight_decay=0.00001)
        criterion = nn.BCELoss().to(runtime_device)

        best_val_auc = -1.0
        best_state = None
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            train_model.train()
            optimizer.zero_grad()
            predictions, rd_loss, pd_loss = _forward_predictions(train_model, train_scope)
            pos_loss = criterion(predictions[train_scope["train_mask"]], train_scope["label_pos"][train_scope["train_mask"]])
            loss = pos_loss + rd * rd_loss + pd_weight * pd_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if protocol == "random":
                    train_model.eval()
                    val_predictions, _, _ = _forward_predictions(train_model, val_scope)
                    val_predictions = _sanitize_predictions(val_predictions)
                    val_metrics = compute_binary_metrics(
                        val_scope["label_pos"][val_scope["val_mask"]],
                        val_predictions[val_scope["val_mask"]],
                    )
                    if val_metrics["auc"] > best_val_auc:
                        best_val_auc = val_metrics["auc"]
                        best_state = copy.deepcopy(train_model.state_dict())
                else:
                    val_model = _build_model(module, val_scope, dim_feat, numk, layers, dropout, alpha, runtime_device_str).to(runtime_device)
                    _copy_train_state_to_eval(train_model, val_model, train_scope, val_scope)
                    val_model.eval()
                    val_predictions, _, _ = _forward_predictions(val_model, val_scope)
                    val_predictions = _sanitize_predictions(val_predictions)
                    val_metrics = compute_binary_metrics(
                        val_scope["label_pos"][val_scope["val_mask"]],
                        val_predictions[val_scope["val_mask"]],
                    )
                    if val_metrics["auc"] > best_val_auc:
                        best_val_auc = val_metrics["auc"]
                        best_state = copy.deepcopy(train_model.state_dict())

        if best_state is None:
            raise RuntimeError("RedCDR failed to capture a best checkpoint")

        train_model.load_state_dict(best_state)
        if protocol == "random":
            train_model.eval()
            predictions, _, _ = _forward_predictions(train_model, test_scope)
            best_predictions = _sanitize_predictions(predictions).detach().cpu()
        else:
            test_model = _build_model(module, test_scope, dim_feat, numk, layers, dropout, alpha, runtime_device_str).to(runtime_device)
            _copy_train_state_to_eval(train_model, test_model, train_scope, test_scope)
            test_model.eval()
            test_predictions, _, _ = _forward_predictions(test_model, test_scope)
            best_predictions = _sanitize_predictions(test_predictions).detach().cpu()

        test_metrics = compute_binary_metrics(
            test_scope["label_pos"][test_scope["test_mask"]],
            best_predictions[test_scope["test_mask"].cpu()],
        )
        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "auc": test_metrics["auc"],
            "aupr": test_metrics["aupr"],
            "f1": test_metrics["f1"],
            "acc": test_metrics["acc"],
        }
        prediction_rows = build_prediction_rows_from_mask(
            cell_ids=test_scope["cell_ids"],
            drug_ids=test_scope["drug_ids"],
            flat_mask=test_scope["test_mask"].cpu(),
            flat_labels=test_scope["label_pos"].cpu(),
            flat_predictions=best_predictions,
        )
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "RedCDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "graph_builder": "topk_directed_cosine",
            "graph_inputs": {
                "cell_graph": "similarity.csv",
                "drug_similarity_graph": "physicochemical.csv",
                "drug_structure": "drug_graph_feat",
            },
            "config": {
                "lr": lr,
                "rd": rd,
                "pd_weight": pd_weight,
                "top_k": top_k,
            },
        },
    )
