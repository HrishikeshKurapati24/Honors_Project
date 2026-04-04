import copy
import os
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

from benchmarking_common import read_json, resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import save_model_outputs
from benchmarking_common.splits import PROTOCOL_RANDOM
from benchmark_wrappers.common import (
    build_prediction_rows_from_mask,
    load_external_module,
    load_fold_bundle_tables,
    resolve_fold_ids,
    scope_entities_for_split,
)


def _load_soulcdr_module(root_dir: str):
    scalable_dir = os.path.join(root_dir, "scalable model")
    return load_external_module(
        module_name="benchmark_soulcdr_main",
        file_path=os.path.join(scalable_dir, "main_scalable.py"),
        extra_paths=[scalable_dir],
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=["cell_id", "drug_id", "label"])


def _concat_pairs(*tables: pd.DataFrame) -> pd.DataFrame:
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return _empty_pairs()
    return pd.concat(non_empty, ignore_index=True).drop_duplicates(["cell_id", "drug_id"], keep="first")


def _subset_loaded(loaded, cell_ids: List[str], drug_ids: List[str], response_pairs: pd.DataFrame) -> SimpleNamespace:
    omics_features = {
        category: {subtype: df.loc[cell_ids] for subtype, df in subtype_map.items()}
        for category, subtype_map in loaded.omics_features.items()
    }
    similarity_feature = loaded.similarity_feature.loc[cell_ids]
    physicochemical_feature = {drug_id: loaded.physicochemical_feature[drug_id] for drug_id in drug_ids}
    drug_feature = {drug_id: loaded.drug_feature[drug_id] for drug_id in drug_ids}
    data_new = [(row.cell_id, row.drug_id, int(row.label)) for row in response_pairs.itertuples(index=False)]
    return SimpleNamespace(
        drug_feature=drug_feature,
        omics_features=omics_features,
        similarity_feature=similarity_feature,
        data_new=data_new,
        nb_celllines=len(cell_ids),
        nb_drugs=len(drug_ids),
        physicochemical_feature=physicochemical_feature,
        selected_omics_stems=getattr(loaded, "selected_omics_stems", []),
    )


def _prepare_runtime(module, processed, device: torch.device, top_k: int) -> Dict:
    return {
        "omics_data_device": module.move_omics_to_device(processed.omics_tensors, device),
        "label_pos": processed.label_pos.to(device),
        "train_mask": processed.train_mask.to(device),
        "val_mask": processed.val_mask.to(device),
        "test_mask": processed.test_mask.to(device),
        "train_edge_tensor": torch.tensor(processed.train_edge, dtype=torch.long, device=device),
        "global_edge_index_dict": module.build_hetero_global_graph(
            cell_similarity_tensor=processed.similarity_tensor.to(device),
            drug_phys_tensor=processed.physicochemical_tensor.to(device),
            top_k=top_k,
            device=device,
        ),
    }


def _predict_full_matrix(module, model, processed, runtime_payload, device: torch.device):
    model.eval()
    with torch.no_grad():
        drug_batch = next(iter(processed.drug_loader)).to(device)
        batch_drug_indices = drug_batch.drug_idx.long().to(device)
        batch_edge_index_dict = module.get_batch_hetero_graph(
            global_edge_index_dict=runtime_payload["global_edge_index_dict"],
            batch_drug_indices=batch_drug_indices,
            train_edge_subset=runtime_payload["train_edge_tensor"],
            device=device,
        )
        out = model(
            drug_feature=drug_batch.x,
            drug_adj=drug_batch.edge_index,
            ibatch=drug_batch.batch,
            omics_data=runtime_payload["omics_data_device"],
            hetero_graph_edge_index_dict=batch_edge_index_dict,
        )
        z_d = out["drug_embeddings"]
        z_c = out["cell_embeddings"]
        z_c_rep = z_c.repeat_interleave(processed.nb_drugs, dim=0)
        z_d_rep = z_d.repeat(processed.nb_celllines, 1)
        logits = model.predictor(torch.cat([z_d_rep, z_c_rep], dim=1)).view(-1)
        return torch.sigmoid(logits).detach().cpu()


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
    fold_ids: List[int] | None = None,
) -> Dict:
    set_seed(seed)
    module = _load_soulcdr_module(root_dir)
    runtime_device = resolve_device(device)
    metadata_path = os.path.join(prepared_dir, "metadata.json")
    prepared_metadata = read_json(metadata_path) if os.path.isfile(metadata_path) else {}
    selected_omics = prepared_metadata.get("omics_for_soulcdr")
    if not selected_omics:
        selected_omics = (
            ["genomics_mutation", "transcriptomics_expression", "epigenomics_methylation"]
            if os.path.isfile(os.path.join(prepared_dir, "genomics_mutation.csv"))
            else ["transcriptomics_expression"]
            if os.path.isfile(os.path.join(prepared_dir, "transcriptomics_expression.csv"))
            else ["pathway"]
        )
    fusion_dim = fusion_channels if fusion_channels is not None else hidden_channels

    loaded = module.dataload_scalable(prepared_dir, selected_omics=selected_omics)
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

    for fold in resolve_fold_ids(split_dir, fold_ids):
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = bundle["manifest"].get("protocol", PROTOCOL_RANDOM)
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == PROTOCOL_RANDOM:
            split_tables = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
            processed_train = module.process_scalable(
                loaded=loaded,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=0,
                split_tables=split_tables,
            )
            processed_val = processed_train
            processed_test = processed_train
            prediction_cell_ids = sorted({item[0] for item in loaded.data_new})
            prediction_drug_ids = sorted({item[1] for item in loaded.data_new})
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            loaded_train = _subset_loaded(loaded, train_cells, train_drugs, train_pairs)
            loaded_val = _subset_loaded(loaded, val_cells, val_drugs, _concat_pairs(train_pairs, val_pairs))
            loaded_test = _subset_loaded(loaded, test_cells, test_drugs, _concat_pairs(train_pairs, test_pairs))

            processed_train = module.process_scalable(
                loaded=loaded_train,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=0,
                split_tables={"train": train_pairs, "val": _empty_pairs(), "test": _empty_pairs()},
            )
            processed_val = module.process_scalable(
                loaded=loaded_val,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=0,
                split_tables={"train": train_pairs, "val": val_pairs, "test": _empty_pairs()},
            )
            processed_test = module.process_scalable(
                loaded=loaded_test,
                k_folds=5,
                current_fold=fold - 1,
                data_split_seed=seed,
                drug_batch_size=0,
                split_tables={"train": train_pairs, "val": _empty_pairs(), "test": test_pairs},
            )
            prediction_cell_ids = sorted({item[0] for item in loaded_test.data_new})
            prediction_drug_ids = sorted({item[1] for item in loaded_test.data_new})

        encoder_configs = module.build_encoder_configs(
            omics_tensors=processed_train.omics_tensors,
            fusion_dim=fusion_dim,
        )
        model = module.SOULCDR(
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

        train_runtime = _prepare_runtime(module, processed_train, runtime_device, top_k)
        val_runtime = _prepare_runtime(module, processed_val, runtime_device, top_k)
        test_runtime = _prepare_runtime(module, processed_test, runtime_device, top_k)

        best_val_auc = -1.0
        best_state = None
        best_test_predictions = None

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
                best_test_predictions = _predict_full_matrix(
                    module=module,
                    model=model,
                    processed=processed_test,
                    runtime_payload=test_runtime,
                    device=runtime_device,
                )

        if best_state is None or best_test_predictions is None:
            raise RuntimeError("SOULCDR failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        metrics = compute_binary_metrics(
            processed_test.label_pos[processed_test.test_mask].cpu().numpy(),
            best_test_predictions[test_runtime["test_mask"].cpu()].numpy(),
        )
        fold_metrics.append(
            {
                "fold": fold,
                "best_val_auc": float(best_val_auc),
                "auc": metrics["auc"],
                "aupr": metrics["aupr"],
                "f1": metrics["f1"],
                "acc": metrics["acc"],
            }
        )
        prediction_rows_by_fold[fold] = build_prediction_rows_from_mask(
            cell_ids=prediction_cell_ids,
            drug_ids=prediction_drug_ids,
            flat_mask=processed_test.test_mask,
            flat_labels=processed_test.label_pos,
            flat_predictions=best_test_predictions,
        )

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "SOULCDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "selected_omics": selected_omics,
            "config": {
                "lr": lr,
                "fusion_channels": fusion_dim,
                "hidden_channels": hidden_channels,
                "output_channels": output_channels,
            },
        },
    )
