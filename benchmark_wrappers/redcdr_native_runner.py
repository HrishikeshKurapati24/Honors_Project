import copy
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from benchmarking_common import resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import save_model_outputs
from benchmark_wrappers.common import (
    build_flat_mask,
    build_prediction_rows_from_mask,
    build_redcdr_allpairs,
    filter_pairs_by_scope,
    load_external_module,
    load_fold_bundle_tables,
    load_prepared_dataset,
    protocol_from_bundle,
    resolve_fold_ids,
    scope_entities_for_split,
    sorted_cell_drug_ids,
)


def _load_redcdr_model(root_dir: str):
    model_dir = os.path.join(root_dir, "benchmark models", "RedCDR-main")
    return load_external_module(
        module_name="benchmark_redcdr_native_model",
        file_path=os.path.join(model_dir, "model_native.py"),
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
    dataset: Dict,
    scope_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    device: torch.device,
) -> Dict:
    cell_ids, drug_ids = sorted_cell_drug_ids(scope_pairs)
    expression_df = dataset["tables"]["transcriptomics_expression"].loc[cell_ids]
    cnv_df = dataset["tables"]["genomics_cnv"].loc[cell_ids]
    fingerprint_df = dataset["tables"]["drug_fingerprint"].loc[drug_ids]

    gexpr = torch.from_numpy(expression_df.to_numpy(dtype="float32")).to(device)
    cnv = torch.from_numpy(cnv_df.to_numpy(dtype="float32")).to(device)
    fingerprint = torch.from_numpy(fingerprint_df.to_numpy(dtype="float32")).to(device)

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

    train_edge = build_redcdr_allpairs(cell_ids, drug_ids, train_pairs)
    if train_edge.size:
        train_edge = np.vstack((train_edge, train_edge[:, [1, 0, 2]]))

    return {
        "cell_ids": cell_ids,
        "drug_ids": drug_ids,
        "expression_df": expression_df,
        "cnv_df": cnv_df,
        "fingerprint_df": fingerprint_df,
        "gexpr": gexpr,
        "cnv": cnv,
        "fingerprint": fingerprint,
        "label_pos": label_pos.to(device),
        "train_mask": train_mask.to(device),
        "val_mask": val_mask.to(device),
        "test_mask": test_mask.to(device),
        "train_edge": train_edge,
    }


def _build_model(module, scope: Dict, dim_feat: int, numk: int, layers: int, dropout: float, alpha: float, device_str: str):
    return module.RedCDR(
        scope["fingerprint_df"].shape[-1],
        scope["expression_df"].shape[-1],
        scope["cnv_df"].shape[-1],
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
        scope["fingerprint"],
        scope["gexpr"],
        scope["cnv"],
        scope["train_edge"],
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
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)
    runtime_device_str = runtime_device.type if runtime_device.index is None else f"{runtime_device.type}:{runtime_device.index}"
    module = _load_redcdr_model(root_dir)
    dataset = load_prepared_dataset(prepared_dir)

    fold_metrics = []
    prediction_rows_by_fold = {}

    for fold in resolve_fold_ids(split_dir, fold_ids):
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = protocol_from_bundle(bundle)
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == "random":
            full_pairs = dataset["response_pairs"]
            train_scope = _build_scope(dataset, full_pairs, train_pairs, val_pairs, test_pairs, runtime_device)
            val_scope = train_scope
            test_scope = train_scope
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            train_pairs_scope = filter_pairs_by_scope(train_pairs, train_cells, train_drugs)
            val_pairs_scope = filter_pairs_by_scope(val_pairs, val_cells, val_drugs)
            test_pairs_scope = filter_pairs_by_scope(test_pairs, test_cells, test_drugs)

            train_scope = _build_scope(dataset, train_pairs_scope, train_pairs_scope, _empty_pairs(), _empty_pairs(), runtime_device)
            val_scope = _build_scope(
                dataset,
                _concat_pairs(train_pairs_scope, val_pairs_scope),
                train_pairs_scope,
                val_pairs_scope,
                _empty_pairs(),
                runtime_device,
            )
            test_scope = _build_scope(
                dataset,
                _concat_pairs(train_pairs_scope, test_pairs_scope),
                train_pairs_scope,
                _empty_pairs(),
                test_pairs_scope,
                runtime_device,
            )

        train_model = _build_model(module, train_scope, dim_feat, numk, layers, dropout, alpha, runtime_device_str).to(runtime_device)
        optimizer = torch.optim.Adam(train_model.parameters(), lr=lr, weight_decay=0.00001)
        criterion = nn.BCELoss().to(runtime_device)

        best_val_auc = -1.0
        best_state = None
        best_predictions = None

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
                        best_predictions = val_predictions.detach().cpu()
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
                        test_model = _build_model(module, test_scope, dim_feat, numk, layers, dropout, alpha, runtime_device_str).to(runtime_device)
                        _copy_train_state_to_eval(train_model, test_model, train_scope, test_scope)
                        test_model.eval()
                        test_predictions, _, _ = _forward_predictions(test_model, test_scope)
                        best_predictions = _sanitize_predictions(test_predictions).detach().cpu()

        if best_state is None or best_predictions is None:
            raise RuntimeError("RedCDR native runner failed to capture a best checkpoint")

        if protocol == "random":
            train_model.load_state_dict(best_state)
            train_model.eval()
            predictions, _, _ = _forward_predictions(train_model, test_scope)
            best_predictions = _sanitize_predictions(predictions).detach().cpu()

        test_metrics = compute_binary_metrics(
            test_scope["label_pos"][test_scope["test_mask"]],
            best_predictions[test_scope["test_mask"].cpu()],
        )
        fold_metrics.append(
            {
                "fold": fold,
                "best_val_auc": float(best_val_auc),
                "auc": test_metrics["auc"],
                "aupr": test_metrics["aupr"],
                "f1": test_metrics["f1"],
                "acc": test_metrics["acc"],
            }
        )
        prediction_rows_by_fold[fold] = build_prediction_rows_from_mask(
            cell_ids=test_scope["cell_ids"],
            drug_ids=test_scope["drug_ids"],
            flat_mask=test_scope["test_mask"].cpu(),
            flat_labels=test_scope["label_pos"].cpu(),
            flat_predictions=best_predictions,
        )

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "RedCDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "config": {"lr": lr, "numk": numk, "rd": rd, "pd_weight": pd_weight},
            "native_inputs": ["transcriptomics_expression", "genomics_cnv", "drug_fingerprint"],
        },
    )
