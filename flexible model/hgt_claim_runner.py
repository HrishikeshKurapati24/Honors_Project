import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_flexible import LoadedFlexibleData, dataload_flexible, process_flexible
from hgt_claim_model import HGTClaimFUSECDR
from hgt_claim_utils import (
    build_prediction_rows_from_mask,
    build_prediction_rows_from_pairs,
    empty_edge_index,
    filter_similarity_edges,
    fold_is_complete,
    load_completed_fold_metrics,
    load_fold_tables,
    sample_negative_pairs,
    save_fold_metrics_table,
    save_fold_result,
    save_summary,
    sparsify_positive_train_pairs,
    sorted_entity_ids,
)
from main_flexible import (
    SupConLoss,
    build_encoder_configs,
    build_hetero_global_graph,
    metrics_graph,
    move_omics_to_device,
    resolve_device,
    set_seed,
    train_one_epoch,
)
from flexibility_utils import read_json


HGT_METADATA = (
    ["drug", "cell"],
    [
        ("drug", "responds_to", "cell"),
        ("cell", "similar_to", "cell"),
        ("drug", "similar_to", "drug"),
    ],
)


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def predict_all_pairs(
    *,
    model: HGTClaimFUSECDR,
    processed,
    omics_data_device: Dict[str, Dict[str, torch.Tensor]],
    train_edge_tensor: torch.Tensor,
    global_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    score_matrix = np.zeros((processed.nb_celllines, processed.nb_drugs), dtype=np.float32)

    from main_flexible import get_batch_hetero_graph

    with torch.no_grad():
        for drug_batch in processed.drug_loader:
            drug_batch = drug_batch.to(device)
            batch_drug_indices = drug_batch.drug_idx.long().to(device)
            mask_drug_in_batch = torch.isin(train_edge_tensor[:, 1], batch_drug_indices)
            batch_train_edges = train_edge_tensor[mask_drug_in_batch]

            batch_edge_index_dict = get_batch_hetero_graph(
                global_edge_index_dict=global_edge_index_dict,
                batch_drug_indices=batch_drug_indices,
                train_edge_subset=batch_train_edges,
                device=device,
            )

            out = model(
                drug_feature=drug_batch.x,
                drug_adj=drug_batch.edge_index,
                ibatch=drug_batch.batch,
                omics_data=omics_data_device,
                hetero_graph_edge_index_dict=batch_edge_index_dict,
            )
            z_d = out["drug_embeddings"]
            z_c = out["cell_embeddings"]
            curr_bs = int(len(batch_drug_indices))

            z_c_rep = z_c.repeat_interleave(curr_bs, dim=0)
            z_d_rep = z_d.repeat(processed.nb_celllines, 1)
            z_pair = torch.cat([z_d_rep, z_c_rep], dim=1)
            logits = model.predictor(z_pair).view(processed.nb_celllines, curr_bs)
            batch_scores = torch.sigmoid(logits).detach().cpu().numpy()
            score_matrix[:, batch_drug_indices.detach().cpu().numpy()] = batch_scores
    return score_matrix.reshape(-1)


def _pair_predictions_from_full_scores(
    *,
    pairs_df: pd.DataFrame,
    cell_ids: Sequence[str],
    drug_ids: Sequence[str],
    flat_scores: np.ndarray,
) -> np.ndarray:
    cell_to_idx = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    drug_to_idx = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    nb_drugs = len(drug_ids)
    predictions = []
    for row in pairs_df.itertuples(index=False):
        flat_idx = cell_to_idx[str(row.cell_id)] * nb_drugs + drug_to_idx[str(row.drug_id)]
        predictions.append(float(flat_scores[flat_idx]))
    return np.asarray(predictions, dtype=np.float32)


def _base_fold_summary(
    *,
    fold_id: int,
    best_val_auc: float,
    test_metrics: Dict[str, float],
    parameter_count: int,
    fold_elapsed_seconds: float,
    edge_mode: str,
    variant_name: str,
    local_layers: int,
    global_layers: int,
) -> Dict:
    return {
        "fold": int(fold_id),
        "best_val_auc": float(best_val_auc),
        "test_auc": float(test_metrics["auc"]),
        "test_aupr": float(test_metrics["aupr"]),
        "test_f1": float(test_metrics["f1"]),
        "test_acc": float(test_metrics["acc"]),
        "parameter_count": int(parameter_count),
        "fold_elapsed_seconds": float(fold_elapsed_seconds),
        "edge_mode": edge_mode,
        "variant_name": variant_name,
        "local_layers": int(local_layers),
        "global_layers": int(global_layers),
    }


def run_hgt_training_experiment(
    *,
    dataset_root: Path | str,
    split_dir: Path | str,
    results_dir: Path | str,
    variant_name: str,
    use_local_branch: bool,
    use_global_branch: bool,
    num_local_layers: int = 2,
    num_global_layers: int = 2,
    edge_mode: str = "full_graph",
    response_sparsity_fraction: float = 0.0,
    device: str = "auto",
    seed: int = 0,
    fold_ids: Optional[Sequence[int]] = None,
    epochs: int = 120,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_channels: int = 256,
    output_channels: int = 64,
    fusion_dim: int = 512,
    dropout: float = 0.2,
    heads: int = 4,
    drug_num_gnn_layers: int = 3,
    top_k: int = 10,
    contrastive_weight: float = 0.005,
    temperature: float = 0.05,
    warmup_epochs: int = 10,
    max_contrastive_pairs: int = 2048,
) -> Dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    loaded: LoadedFlexibleData = dataload_flexible(str(Path(dataset_root).resolve()))
    cell_ids, drug_ids = sorted_entity_ids(loaded.data_new)
    runtime_device = resolve_device(device)
    criterion = nn.BCEWithLogitsLoss()
    contrastive_criterion = SupConLoss(temperature=temperature)

    completed = {
        int(row["fold"]): row
        for row in load_completed_fold_metrics(results_dir)
        if fold_is_complete(results_dir, int(row["fold"]), extra_required_suffixes=["config"])
    }
    fold_metrics: List[Dict] = list(completed.values())
    resolved_folds = list(fold_ids) if fold_ids else sorted(
        int(path.name.split("_", 1)[1])
        for path in Path(split_dir).glob("fold_*")
        if path.is_dir()
    )

    for fold_id in resolved_folds:
        if fold_id in completed:
            continue

        set_seed(seed + fold_id * 1000)
        fold_start = time.time()
        split_tables = load_fold_tables(split_dir, fold_id)
        original_train = split_tables["train"].copy()
        observed_train, withheld_pos = sparsify_positive_train_pairs(
            original_train,
            fraction=response_sparsity_fraction,
            seed=seed + fold_id * 1000 + 17,
        )
        processed = process_flexible(
            loaded=loaded,
            k_folds=len(resolved_folds),
            current_fold=max(fold_id - 1, 0),
            data_split_seed=seed,
            drug_batch_size=0,
            split_tables={
                "train": observed_train,
                "val": split_tables["val"],
                "test": split_tables["test"],
            },
        )

        encoder_configs = build_encoder_configs(
            omics_tensors=processed.omics_tensors,
            fusion_dim=fusion_dim,
        )
        model = HGTClaimFUSECDR(
            atom_shape=processed.atom_shape,
            encoder_configs=encoder_configs,
            metadata=HGT_METADATA,
            hidden_dim=hidden_channels,
            output_dim=output_channels,
            fusion_dim=fusion_dim,
            dropout=dropout,
            num_local_layers=num_local_layers,
            num_global_layers=num_global_layers,
            heads=heads,
            drug_num_gnn_layers=drug_num_gnn_layers,
            use_local_branch=use_local_branch,
            use_global_branch=use_global_branch,
        ).to(runtime_device)
        parameter_count = _count_trainable_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        omics_data_device = move_omics_to_device(processed.omics_tensors, runtime_device)
        label_pos = processed.label_pos.to(runtime_device)
        train_mask = processed.train_mask.to(runtime_device)
        val_mask = processed.val_mask.to(runtime_device)
        test_mask = processed.test_mask.to(runtime_device)
        train_edge_tensor = torch.tensor(processed.train_edge, dtype=torch.long, device=runtime_device)
        global_edge_index_dict = build_hetero_global_graph(
            cell_similarity_tensor=processed.similarity_tensor.to(runtime_device),
            drug_phys_tensor=processed.physicochemical_tensor.to(runtime_device),
            top_k=top_k,
            device=runtime_device,
        )
        global_edge_index_dict = filter_similarity_edges(
            global_edge_index_dict,
            mode=edge_mode,
            device=runtime_device,
        )

        best_val_auc = -1.0
        best_state_dict = None
        for epoch in range(epochs):
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                drug_loader=processed.drug_loader,
                omics_data_device=omics_data_device,
                train_edge_tensor=train_edge_tensor,
                label_pos=label_pos,
                train_mask=train_mask,
                nb_celllines=processed.nb_celllines,
                nb_drugs=processed.nb_drugs,
                global_edge_index_dict=global_edge_index_dict,
                contrastive_weight=contrastive_weight,
                warmup_epochs=warmup_epochs,
                epoch=epoch,
                max_contrastive_pairs=max_contrastive_pairs,
                device=runtime_device,
            )

            flat_scores = predict_all_pairs(
                model=model,
                processed=processed,
                omics_data_device=omics_data_device,
                train_edge_tensor=train_edge_tensor,
                global_edge_index_dict=global_edge_index_dict,
                device=runtime_device,
            )
            val_mask_np = val_mask.detach().cpu().numpy().astype(bool)
            val_labels_np = label_pos.detach().cpu().numpy()
            val_auc, _, _, _ = metrics_graph(val_labels_np[val_mask_np], flat_scores[val_mask_np])
            if val_auc > best_val_auc:
                best_val_auc = float(val_auc)
                best_state_dict = copy.deepcopy(model.state_dict())

        if best_state_dict is None:
            raise RuntimeError("Failed to capture a best model state during HGT-claim training.")
        model.load_state_dict(best_state_dict)
        flat_scores = predict_all_pairs(
            model=model,
            processed=processed,
            omics_data_device=omics_data_device,
            train_edge_tensor=train_edge_tensor,
            global_edge_index_dict=global_edge_index_dict,
            device=runtime_device,
        )
        flat_labels_np = label_pos.detach().cpu().numpy()
        test_mask_np = test_mask.detach().cpu().numpy().astype(bool)
        test_auc, test_aupr, test_f1, test_acc = metrics_graph(
            flat_labels_np[test_mask_np],
            flat_scores[test_mask_np],
        )
        test_metrics = {
            "auc": test_auc,
            "aupr": test_aupr,
            "f1": test_f1,
            "acc": test_acc,
        }
        prediction_rows = build_prediction_rows_from_mask(
            cell_ids=cell_ids,
            drug_ids=drug_ids,
            flat_mask=test_mask_np,
            flat_labels=flat_labels_np,
            flat_predictions=flat_scores,
        )

        extra_payloads = {
            "config": pd.DataFrame(
                [
                    {
                        "variant_name": variant_name,
                        "use_local_branch": use_local_branch,
                        "use_global_branch": use_global_branch,
                        "num_local_layers": num_local_layers,
                        "num_global_layers": num_global_layers,
                        "edge_mode": edge_mode,
                        "response_sparsity_fraction": response_sparsity_fraction,
                    }
                ]
            )
        }

        fold_metric = _base_fold_summary(
            fold_id=fold_id,
            best_val_auc=best_val_auc,
            test_metrics=test_metrics,
            parameter_count=parameter_count,
            fold_elapsed_seconds=time.time() - fold_start,
            edge_mode=edge_mode,
            variant_name=variant_name,
            local_layers=num_local_layers,
            global_layers=num_global_layers,
        )

        if response_sparsity_fraction > 0 and not withheld_pos.empty:
            sampled_neg = sample_negative_pairs(
                original_train,
                count=len(withheld_pos),
                seed=seed + fold_id * 1000 + 29,
            )
            recovery_pairs = pd.concat([withheld_pos, sampled_neg], ignore_index=True)
            recovery_pairs = recovery_pairs.sample(frac=1.0, random_state=seed + fold_id * 1000 + 31).reset_index(drop=True)
            recovery_predictions = _pair_predictions_from_full_scores(
                pairs_df=recovery_pairs,
                cell_ids=cell_ids,
                drug_ids=drug_ids,
                flat_scores=flat_scores,
            )
            rec_auc, rec_aupr, rec_f1, rec_acc = metrics_graph(
                recovery_pairs["label"].to_numpy(dtype=np.int64),
                recovery_predictions,
            )
            fold_metric.update(
                {
                    "recovery_auc": float(rec_auc),
                    "recovery_aupr": float(rec_aupr),
                    "recovery_f1": float(rec_f1),
                    "recovery_acc": float(rec_acc),
                    "withheld_positive_count": int(len(withheld_pos)),
                    "sampled_negative_count": int(len(sampled_neg)),
                }
            )
            extra_payloads["hidden_link_predictions"] = pd.DataFrame(
                build_prediction_rows_from_pairs(
                    pairs_df=recovery_pairs,
                    predictions=recovery_predictions,
                )
            )

        save_fold_result(
            results_dir=results_dir,
            fold_id=fold_id,
            metrics=fold_metric,
            prediction_rows=prediction_rows,
            extra_payloads=extra_payloads,
        )
        fold_metrics.append(fold_metric)
        save_fold_metrics_table(results_dir, fold_metrics)
        save_summary(
            results_dir,
            fold_metrics,
            metadata={
                "variant_name": variant_name,
                "use_local_branch": use_local_branch,
                "use_global_branch": use_global_branch,
                "num_local_layers": num_local_layers,
                "num_global_layers": num_global_layers,
                "edge_mode": edge_mode,
                "response_sparsity_fraction": response_sparsity_fraction,
                "dataset_root": str(Path(dataset_root).resolve()),
                "split_dir": str(Path(split_dir).resolve()),
                "epochs": epochs,
                "seed": seed,
                "device": device,
            },
        )

    return read_json(results_dir / "summary.json")
