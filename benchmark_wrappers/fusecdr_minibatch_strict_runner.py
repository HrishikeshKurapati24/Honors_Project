import copy
import os
from typing import Dict, List

import torch
import torch.nn as nn

from benchmarking_common import read_json, resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import (
    load_completed_folds,
    load_saved_predictions,
    save_fold_result,
    save_model_outputs,
)
from benchmarking_common.splits import PROTOCOL_RANDOM
from benchmarking_common.strict_contract import validate_strict_model_contract, validate_strict_prepared_metadata
from benchmark_wrappers.common import build_prediction_rows_from_mask, load_fold_bundle_tables, resolve_fold_ids, scope_entities_for_split
from benchmark_wrappers import fusecdr_strict_runner as base_runner


MODEL_KEY = "FUSECDR_minibatch"


def _assert_minibatch_loader(processed, train_drug_batch_size: int, scope_name: str) -> None:
    num_batches = len(processed.drug_loader)
    if train_drug_batch_size <= 0:
        raise ValueError(
            f"{MODEL_KEY} requires a positive train_drug_batch_size. "
            f"Found {train_drug_batch_size} for {scope_name}."
        )
    if num_batches <= 1:
        raise ValueError(
            f"{MODEL_KEY} expected more than one drug batch for {scope_name}, "
            f"but the configured train_drug_batch_size={train_drug_batch_size} produced {num_batches} batch(es)."
        )


def _assert_full_batch_loader(processed, scope_name: str) -> None:
    num_batches = len(processed.drug_loader)
    if num_batches != 1:
        raise ValueError(
            f"{MODEL_KEY} evaluation expects one full drug batch for {scope_name}, "
            f"but found {num_batches} batches."
        )


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 400,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    fusion_channels: int | None = None,
    hidden_channels: int = 256,
    output_channels: int = 64,
    dropout: float = 0.2,
    num_layers: int = 2,
    heads: int = 4,
    drug_num_gnn_layers: int = 3,
    top_k: int = 10,
    contrastive_weight: float = 0.005,
    temperature: float = 0.05,
    warmup_epochs: int = 10,
    max_contrastive_pairs: int = 2048,
    train_drug_batch_size: int = 64,
    eval_drug_batch_size: int = 0,
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    module = base_runner._load_fusecdr_module(root_dir)
    runtime_device = resolve_device(device)
    metadata_path = os.path.join(prepared_dir, "metadata.json")
    prepared_metadata = read_json(metadata_path) if os.path.isfile(metadata_path) else {}
    validate_strict_prepared_metadata(prepared_dir, prepared_metadata)
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
    selected_omics = prepared_metadata.get("omics_for_fusecdr")
    if not selected_omics:
        selected_omics = [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ]
    fusion_dim = fusion_channels if fusion_channels is not None else hidden_channels

    loaded = module.dataload_flexible(prepared_dir, selected_omics=selected_omics)
    metadata = (
        ["drug", "cell"],
        [
            ("drug", "responds_to", "cell"),
            ("cell", "similar_to", "cell"),
            ("drug", "similar_to", "drug"),
        ],
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
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = bundle["manifest"].get("protocol", PROTOCOL_RANDOM)
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == PROTOCOL_RANDOM:
            split_tables = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
            processed_train = module.process_flexible(
                loaded=loaded,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=train_drug_batch_size,
                split_tables=split_tables,
            )
            processed_eval = module.process_flexible(
                loaded=loaded,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=eval_drug_batch_size,
                split_tables=split_tables,
            )
            processed_val = processed_eval
            processed_test = processed_eval
            prediction_cell_ids = sorted({item[0] for item in loaded.data_new})
            prediction_drug_ids = sorted({item[1] for item in loaded.data_new})
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            loaded_train = base_runner._subset_loaded(loaded, train_cells, train_drugs, train_pairs)
            loaded_val = base_runner._subset_loaded(loaded, val_cells, val_drugs, base_runner._concat_pairs(train_pairs, val_pairs))
            loaded_test = base_runner._subset_loaded(loaded, test_cells, test_drugs, base_runner._concat_pairs(train_pairs, test_pairs))

            processed_train = module.process_flexible(
                loaded=loaded_train,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=train_drug_batch_size,
                split_tables={"train": train_pairs, "val": base_runner._empty_pairs(), "test": base_runner._empty_pairs()},
            )
            processed_val = module.process_flexible(
                loaded=loaded_val,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=eval_drug_batch_size,
                split_tables={"train": train_pairs, "val": val_pairs, "test": base_runner._empty_pairs()},
            )
            processed_test = module.process_flexible(
                loaded=loaded_test,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=eval_drug_batch_size,
                split_tables={"train": train_pairs, "val": base_runner._empty_pairs(), "test": test_pairs},
            )
            prediction_cell_ids = sorted({item[0] for item in loaded_test.data_new})
            prediction_drug_ids = sorted({item[1] for item in loaded_test.data_new})

        _assert_minibatch_loader(processed_train, train_drug_batch_size, "train")
        _assert_full_batch_loader(processed_val, "val")
        _assert_full_batch_loader(processed_test, "test")

        encoder_configs = module.build_encoder_configs(
            omics_tensors=processed_train.omics_tensors,
            fusion_dim=fusion_dim,
        )
        model = module.FUSECDR(
            atom_shape=processed_train.atom_shape,
            encoder_configs=encoder_configs,
            metadata=metadata,
            hidden_dim=hidden_channels,
            output_dim=output_channels,
            fusion_dim=fusion_dim,
            dropout=dropout,
            num_layers=num_layers,
            heads=heads,
            drug_num_gnn_layers=drug_num_gnn_layers,
        ).to(runtime_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        contrastive_criterion = module.SupConLoss(temperature=temperature)

        train_runtime = base_runner._prepare_runtime(processed_train, runtime_device, top_k)
        val_runtime = base_runner._prepare_runtime(processed_val, runtime_device, top_k)
        test_runtime = base_runner._prepare_runtime(processed_test, runtime_device, top_k)

        best_val_auc = -1.0
        best_state = None
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            module.train_one_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                drug_loader=processed_train.drug_loader,
                omics_data_device=train_runtime["omics_data_device"],
                train_edge_tensor=train_runtime["train_edge_tensor"],
                label_pos=train_runtime["label_pos"],
                train_mask=train_runtime["train_mask"],
                nb_celllines=processed_train.nb_celllines,
                nb_drugs=processed_train.nb_drugs,
                global_edge_index_dict=train_runtime["global_edge_index_dict"],
                contrastive_weight=contrastive_weight,
                warmup_epochs=warmup_epochs,
                epoch=epoch,
                max_contrastive_pairs=max_contrastive_pairs,
                device=runtime_device,
            )
            val_auc, _, _, _, _, _ = module.evaluate_split(
                model=model,
                drug_loader=processed_val.drug_loader,
                omics_data_device=val_runtime["omics_data_device"],
                train_edge_tensor=val_runtime["train_edge_tensor"],
                eval_mask=val_runtime["val_mask"],
                label_pos=val_runtime["label_pos"],
                nb_celllines=processed_val.nb_celllines,
                nb_drugs=processed_val.nb_drugs,
                global_edge_index_dict=val_runtime["global_edge_index_dict"],
                device=runtime_device,
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError(f"{MODEL_KEY} failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        best_test_predictions = base_runner._predict_full_matrix(
            module=module,
            model=model,
            processed=processed_test,
            runtime_payload=test_runtime,
            device=runtime_device,
        )
        metrics = compute_binary_metrics(
            processed_test.label_pos[processed_test.test_mask].cpu().numpy(),
            best_test_predictions[test_runtime["test_mask"].cpu()].numpy(),
        )
        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "auc": metrics["auc"],
            "aupr": metrics["aupr"],
            "f1": metrics["f1"],
            "acc": metrics["acc"],
        }
        prediction_rows = build_prediction_rows_from_mask(
            cell_ids=prediction_cell_ids,
            drug_ids=prediction_drug_ids,
            flat_mask=processed_test.test_mask,
            flat_labels=processed_test.label_pos,
            flat_predictions=best_test_predictions,
        )
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": MODEL_KEY,
            "base_model": "FUSECDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "selected_omics": selected_omics,
            "training_regime": "drug_minibatch",
            "train_drug_batch_size": train_drug_batch_size,
            "eval_drug_batch_size": eval_drug_batch_size,
            "graph_builder": "topk_directed_cosine",
            "graph_inputs": {
                "cell_graph": "similarity.csv",
                "drug_similarity_graph": "physicochemical.csv",
                "drug_structure": "drug_graph_feat",
            },
            "config": {
                "lr": lr,
                "fusion_channels": fusion_dim,
                "hidden_channels": hidden_channels,
                "output_channels": output_channels,
                "top_k": top_k,
                "contrastive_weight": contrastive_weight,
                "temperature": temperature,
            },
        },
    )
