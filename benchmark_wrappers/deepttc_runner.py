import copy
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from benchmarking_common import resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import save_model_outputs
from benchmark_wrappers.common import (
    build_prediction_rows,
    load_external_module,
    load_fold_bundle_tables,
    load_prepared_dataset,
    resolve_fold_ids,
)


def _load_deepttc_module(root_dir: str):
    model_dir = os.path.join(root_dir, "benchmark models", "DeepTTC-main", "benchmark implementation")
    return load_external_module(
        module_name="benchmark_deepttc_model",
        file_path=os.path.join(model_dir, "model.py"),
        extra_paths=[model_dir],
    )


class DeepTTCDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        expression: pd.DataFrame,
        token_ids: pd.DataFrame,
        attention_mask: pd.DataFrame,
        target_gene_dim: int = 17737,
    ):
        self.pairs = pairs.reset_index(drop=True)
        self.expression = expression
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.target_gene_dim = target_gene_dim

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        cell_id = row["cell_id"]
        drug_id = row["drug_id"]
        label = float(row["label"])

        expr = self.expression.loc[cell_id].to_numpy(dtype=np.float32)
        if expr.shape[0] < self.target_gene_dim:
            expr = np.pad(expr, (0, self.target_gene_dim - expr.shape[0]), mode="constant")
        elif expr.shape[0] > self.target_gene_dim:
            expr = expr[: self.target_gene_dim]

        token_row = self.token_ids.loc[drug_id].to_numpy(dtype=np.int64)
        mask_row = self.attention_mask.loc[drug_id].to_numpy(dtype=np.float32)

        return (
            torch.from_numpy(token_row),
            torch.from_numpy(mask_row),
            torch.from_numpy(expr),
            torch.tensor([label], dtype=torch.float32),
        )


def _predict(model, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    probs = []
    with torch.no_grad():
        for token_ids, masks, expr, _labels in loader:
            token_ids = token_ids.to(device)
            masks = masks.to(device)
            expr = expr.to(device)
            logits = model(token_ids, masks, expr).view(-1)
            probs.append(torch.sigmoid(logits).cpu())
    if not probs:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(probs, dim=0)


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)
    module = _load_deepttc_module(root_dir)
    dataset = load_prepared_dataset(prepared_dir)

    expression = dataset["tables"]["transcriptomics_expression"]
    token_ids = dataset["aux"]["smiles_token_ids"]
    attention_mask = dataset["aux"]["smiles_attention_mask"]

    fold_metrics = []
    prediction_rows_by_fold = {}

    for fold in resolve_fold_ids(split_dir, fold_ids):
        bundle = load_fold_bundle_tables(split_dir, fold)
        train_pairs = bundle["train"].reset_index(drop=True)
        val_pairs = bundle["val"].reset_index(drop=True)
        test_pairs = bundle["test"].reset_index(drop=True)

        train_loader = DataLoader(
            DeepTTCDataset(train_pairs, expression, token_ids, attention_mask),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            DeepTTCDataset(val_pairs, expression, token_ids, attention_mask),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            DeepTTCDataset(test_pairs, expression, token_ids, attention_mask),
            batch_size=batch_size,
            shuffle=False,
        )

        model = module.DeepTTC_Model().to(runtime_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_val_auc = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for token_batch, mask_batch, expr_batch, label_batch in train_loader:
                token_batch = token_batch.to(runtime_device)
                mask_batch = mask_batch.to(runtime_device)
                expr_batch = expr_batch.to(runtime_device)
                label_batch = label_batch.to(runtime_device).view(-1)

                optimizer.zero_grad()
                logits = model(token_batch, mask_batch, expr_batch).view(-1)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

            val_scores = _predict(model, val_loader, runtime_device)
            val_metrics = compute_binary_metrics(val_pairs["label"].to_numpy(dtype=float), val_scores)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("DeepTTC runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        test_scores = _predict(model, test_loader, runtime_device)
        test_metrics = compute_binary_metrics(test_pairs["label"].to_numpy(dtype=float), test_scores)

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
        prediction_rows_by_fold[fold] = build_prediction_rows(test_pairs, test_scores.numpy())

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "DeepTTC",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "config": {
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            },
        },
    )
