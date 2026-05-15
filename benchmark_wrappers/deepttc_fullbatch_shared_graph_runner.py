import copy
from typing import Dict, List

import torch
import torch.nn as nn

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
)
from benchmark_wrappers import deepttc_shared_graph_runner as base_runner
from benchmark_wrappers.common import build_prediction_rows, load_fold_bundle_tables, protocol_from_bundle, resolve_fold_ids


MODEL_KEY = "DeepTTC_fullbatch"


def _train_epoch_fullbatch(
    model,
    scope: Dict,
    pair_indices: torch.Tensor,
    labels: torch.Tensor,
    optimizer,
    scaler,
    criterion,
    use_amp: bool,
) -> None:
    if pair_indices.numel() == 0:
        return
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        logits = base_runner._predict_logits(model, scope, pair_indices)
        loss = criterion(logits, labels)
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 500,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_size: int = 64,
    top_k: int = 10,
    patience: int | None = 100,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    runtime_device = resolve_device(device)
    use_amp = runtime_device.type == "cuda"
    module = base_runner._load_deepttc_module(root_dir)
    dataset = base_runner.load_prepared_dataset(prepared_dir)
    validate_strict_prepared_metadata(prepared_dir, dataset["metadata"])
    validate_strict_model_contract(
        MODEL_KEY,
        predictive_inputs=(
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ),
        graph_inputs=("similarity.csv", "physicochemical.csv", "train_pairs"),
    )

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
        protocol = protocol_from_bundle(bundle)
        train_pairs = bundle["train"].reset_index(drop=True)
        val_pairs = bundle["val"].reset_index(drop=True)
        test_pairs = bundle["test"].reset_index(drop=True)

        if protocol == "random":
            scope_pairs = dataset["response_pairs"]
            train_scope = base_runner._build_scope(dataset, scope_pairs, train_pairs, runtime_device, top_k)
            val_scope = train_scope
            test_scope = train_scope
        else:
            train_scope = base_runner._build_scope(
                dataset,
                base_runner._concat_pairs(train_pairs),
                train_pairs,
                runtime_device,
                top_k,
            )
            val_scope = base_runner._build_scope(
                dataset,
                base_runner._concat_pairs(train_pairs, val_pairs),
                train_pairs,
                runtime_device,
                top_k,
            )
            test_scope = base_runner._build_scope(
                dataset,
                base_runner._concat_pairs(train_pairs, test_pairs),
                train_pairs,
                runtime_device,
                top_k,
            )

        train_indices, train_labels = base_runner._pairs_to_local_tensors(
            train_pairs,
            train_scope["cell_ids"],
            train_scope["drug_ids"],
            runtime_device,
        )
        val_indices, val_labels = base_runner._pairs_to_local_tensors(
            val_pairs,
            val_scope["cell_ids"],
            val_scope["drug_ids"],
            runtime_device,
        )
        test_indices, test_labels = base_runner._pairs_to_local_tensors(
            test_pairs,
            test_scope["cell_ids"],
            test_scope["drug_ids"],
            runtime_device,
        )

        model = module.DeepTTCSharedGraph(
            gene_input_dim=train_scope["expression"].shape[1],
            atom_dim=train_scope["drug_tokens"].shape[2],
            max_tokens=train_scope["drug_tokens"].shape[1],
        ).to(runtime_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_val_auc = -1.0
        best_state = None
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_trained = 0
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            model.train()
            _train_epoch_fullbatch(
                model=model,
                scope=train_scope,
                pair_indices=train_indices,
                labels=train_labels,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                use_amp=use_amp,
            )

            model.eval()
            with torch.no_grad():
                val_scores = base_runner._predict_scores_chunked(
                    model,
                    val_scope,
                    val_indices,
                    batch_size=batch_size,
                    use_amp=use_amp,
                )
                val_metrics = compute_binary_metrics(val_labels, val_scores)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epochs_trained = epoch + 1
            should_stop = patience is not None and epochs_without_improvement >= patience
            if should_stop:
                print(
                    f"> Fold {fold} - Early stop at epoch {epochs_trained}/{epochs} "
                    f"(patience={patience}, best_epoch={best_epoch}, best_val_auc={best_val_auc:.4f})",
                    flush=True,
                )
                break

        if best_state is None:
            raise RuntimeError(f"{MODEL_KEY} strict runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_scores = base_runner._predict_scores_chunked(
                model,
                test_scope,
                test_indices,
                batch_size=batch_size,
                use_amp=use_amp,
            )
        test_metrics = compute_binary_metrics(test_labels, test_scores)

        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "auc": test_metrics["auc"],
            "aupr": test_metrics["aupr"],
            "f1": test_metrics["f1"],
            "acc": test_metrics["acc"],
        }
        prediction_rows = build_prediction_rows(test_pairs, test_scores.detach().cpu().numpy())
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": MODEL_KEY,
            "base_model": "DeepTTC",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "training_regime": "per_epoch_full_batch",
            "graph_builder": "topk_directed_cosine",
            "graph_inputs": {
                "cell_graph": "similarity.csv",
                "drug_similarity_graph": "physicochemical.csv",
                "response_graph": "train_pairs",
                "drug_structure": "drug_graph_feat",
            },
            "config": {
                "epochs": epochs,
                "patience": patience,
                "lr": lr,
                "weight_decay": weight_decay,
                "eval_chunk_size": batch_size,
                "top_k": top_k,
            },
        },
    )
