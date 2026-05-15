"""
main_flexible.py

Locked flexible training script for a single best FUSE-CDR architecture:
- Drug GNN: GIN
- Graph type: heterogenous
- Local branch: GraphSAGE (SAGEConv in HeteroConv)
- Global branch: Graph Transformer (HGT)
- Contrastive learning: always enabled

Data root is fixed to: final_dataset/
"""

import argparse
import copy
import csv
import datetime
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from data_flexible import dataload_flexible, list_available_omics, process_flexible
from model_flexible import EncoderConfig, OmicsEncoderRegistry, FUSECDR


class Logger:
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SupConLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device

        if features.dim() == 2:
            features = features.unsqueeze(1)
        if features.dim() < 3:
            raise ValueError(f"features must be >=3D after unsqueeze, got shape={features.shape}")

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels does not match number of features")

        mask = torch.eq(labels, labels.T).float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = features.view(batch_size * contrast_count, -1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
        contrast_feature = nn.functional.normalize(contrast_feature, dim=1)

        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


def metrics_graph(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = float(-np.trapz(precision, recall))

    if len(np.unique(y_true)) < 2:
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true, y_pred))

    real_score = np.atleast_2d(np.asarray(y_true).flatten())
    predict_score = np.atleast_2d(np.asarray(y_pred).flatten())

    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    if sorted_predict_score.size == 0:
        return auc, aupr, 0.0, 0.0

    threshold_indices = np.int32(sorted_predict_score.size * np.arange(1, 1000) / 1000)
    threshold_indices = np.clip(threshold_indices, 0, sorted_predict_score.size - 1)
    thresholds = sorted_predict_score[threshold_indices]
    thresholds = np.atleast_2d(np.asarray(thresholds).flatten())

    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    tp = predict_score_matrix.dot(real_score.T)
    fp = predict_score_matrix.sum(axis=1, keepdims=True) - tp
    fn = real_score.sum() - tp
    total = real_score.shape[1]
    tn = total - tp - fp - fn

    f1_scores = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0,
    )
    acc_scores = (tp + tn) / total

    max_index = int(np.argmax(f1_scores))
    f1_score = float(np.clip(f1_scores.flat[max_index], 0.0, 1.0))
    accuracy = float(np.clip(acc_scores.flat[max_index], 0.0, 1.0))

    return auc, aupr, f1_score, accuracy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_encoder_configs(
    omics_tensors: Dict[str, Dict[str, torch.Tensor]],
    fusion_dim: int,
) -> List[EncoderConfig]:
    configs: List[EncoderConfig] = []
    for category, subtype_map in omics_tensors.items():
        for subtype, tensor in subtype_map.items():
            input_dim = int(tensor.shape[1])
            encoder_type = OmicsEncoderRegistry.resolve_encoder_type(category, subtype)

            configs.append(
                EncoderConfig(
                    category=category,
                    subtype=subtype,
                    encoder_type=encoder_type,
                    input_dim=input_dim,
                    output_dim=fusion_dim,
                )
            )
    return configs


def move_omics_to_device(
    omics_tensors: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        category: {subtype: tensor.to(device) for subtype, tensor in subtype_map.items()}
        for category, subtype_map in omics_tensors.items()
    }


def _topk_directed_edges(feature_tensor: torch.Tensor, top_k: int, device: torch.device) -> torch.Tensor:
    if feature_tensor is None or feature_tensor.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    feats = feature_tensor.detach().cpu().numpy()
    sim = cosine_similarity(feats)
    np.fill_diagonal(sim, 0)

    n_nodes = sim.shape[0]
    if n_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    k = max(1, min(top_k, n_nodes - 1))
    topk_indices = np.argsort(sim, axis=1)[:, -k:]

    src, dst = [], []
    for i in range(n_nodes):
        for j in topk_indices[i]:
            src.append(i)
            dst.append(j)

    return torch.tensor([src, dst], dtype=torch.long, device=device)


def build_hetero_global_graph(
    cell_similarity_tensor: torch.Tensor,
    drug_phys_tensor: torch.Tensor,
    top_k: int,
    device: torch.device,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    cell_edge_index = _topk_directed_edges(cell_similarity_tensor, top_k, device)
    drug_edge_index = _topk_directed_edges(drug_phys_tensor, top_k, device)

    return {
        ("cell", "similar_to", "cell"): cell_edge_index,
        ("drug", "similar_to", "drug"): drug_edge_index,
    }


def get_batch_hetero_graph(
    global_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    batch_drug_indices: torch.Tensor,
    train_edge_subset: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    batch_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

    # Keep full cell-cell graph
    batch_edge_index_dict[("cell", "similar_to", "cell")] = global_edge_index_dict[
        ("cell", "similar_to", "cell")
    ]

    # Induced subgraph over current batch drugs for drug-drug edges
    d_idx_map = {global_id.item(): local_id for local_id, global_id in enumerate(batch_drug_indices)}
    global_dd_index = global_edge_index_dict[("drug", "similar_to", "drug")]

    if global_dd_index.numel() == 0:
        batch_dd_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        mask_src = torch.isin(global_dd_index[0], batch_drug_indices)
        mask_dst = torch.isin(global_dd_index[1], batch_drug_indices)
        mask = mask_src & mask_dst
        subset_dd_index = global_dd_index[:, mask]

        if subset_dd_index.numel() == 0:
            batch_dd_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            new_src = torch.tensor(
                [d_idx_map[i.item()] for i in subset_dd_index[0]],
                dtype=torch.long,
                device=device,
            )
            new_dst = torch.tensor(
                [d_idx_map[i.item()] for i in subset_dd_index[1]],
                dtype=torch.long,
                device=device,
            )
            batch_dd_index = torch.stack([new_src, new_dst])

    batch_edge_index_dict[("drug", "similar_to", "drug")] = batch_dd_index

    # Construct drug->cell responds edges from current training subset (positive edges only)
    if train_edge_subset is not None and train_edge_subset.numel() > 0:
        pos_edges = train_edge_subset[train_edge_subset[:, 2] == 1]
        if pos_edges.numel() > 0:
            valid_rows = [idx for idx, row in enumerate(pos_edges) if row[1].item() in d_idx_map]
            if len(valid_rows) > 0:
                pos_edges = pos_edges[valid_rows]
                batch_d_ids = torch.tensor(
                    [d_idx_map[d.item()] for d in pos_edges[:, 1]],
                    dtype=torch.long,
                    device=device,
                )
                batch_c_ids = pos_edges[:, 0].long().to(device)
                responds = torch.stack([batch_d_ids, batch_c_ids])
            else:
                responds = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            responds = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        responds = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_edge_index_dict[("drug", "responds_to", "cell")] = responds
    return batch_edge_index_dict


def train_one_epoch(
    model: FUSECDR,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    contrastive_criterion: SupConLoss,
    drug_loader,
    omics_data_device: Dict[str, Dict[str, torch.Tensor]],
    train_edge_tensor: torch.Tensor,
    label_pos: torch.Tensor,
    train_mask: torch.Tensor,
    nb_celllines: int,
    nb_drugs: int,
    global_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    contrastive_weight: float,
    warmup_epochs: int,
    epoch: int,
    max_contrastive_pairs: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    loss_total = 0.0
    cls_total = 0.0
    cont_total = 0.0

    lbl_2d = label_pos.view(nb_celllines, nb_drugs)
    mask_2d = train_mask.view(nb_celllines, nb_drugs)

    for drug_batch in drug_loader:
        optimizer.zero_grad()

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
        z_d_rep = z_d.repeat(nb_celllines, 1)
        z_pair = torch.cat([z_d_rep, z_c_rep], dim=1)
        logits = model.predictor(z_pair).view(-1)

        batch_labels = lbl_2d[:, batch_drug_indices].reshape(-1)
        batch_mask = mask_2d[:, batch_drug_indices].reshape(-1)

        if batch_mask.sum() == 0:
            continue

        cls_loss = criterion(logits[batch_mask], batch_labels[batch_mask])

        cont_loss = torch.tensor(0.0, device=device)
        if epoch >= warmup_epochs:
            z_proj = model.projection_head(z_pair)[batch_mask]
            labels_cl = batch_labels[batch_mask].long()

            unique_labels = torch.unique(labels_cl)
            if len(unique_labels) >= 2:
                counts = torch.stack([(labels_cl == c).sum() for c in unique_labels])
                if torch.all(counts >= 2):
                    if z_proj.shape[0] > max_contrastive_pairs:
                        perm = torch.randperm(z_proj.shape[0], device=device)[:max_contrastive_pairs]
                        z_proj = z_proj[perm]
                        labels_cl = labels_cl[perm]
                    cont_loss = contrastive_criterion(z_proj, labels_cl)

        loss = cls_loss + contrastive_weight * cont_loss
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        cls_total += float(cls_loss.item())
        cont_total += float(cont_loss.item())

    num_batches = len(drug_loader)
    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return loss_total / num_batches, cls_total / num_batches, cont_total / num_batches


def evaluate_split(
    model: FUSECDR,
    drug_loader,
    omics_data_device: Dict[str, Dict[str, torch.Tensor]],
    train_edge_tensor: torch.Tensor,
    eval_mask: torch.Tensor,
    label_pos: torch.Tensor,
    nb_celllines: int,
    nb_drugs: int,
    global_edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    device: torch.device,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_lbls = []

    lbl_2d = label_pos.view(nb_celllines, nb_drugs)
    mask_2d = eval_mask.view(nb_celllines, nb_drugs)

    with torch.no_grad():
        for drug_batch in drug_loader:
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
            z_d_rep = z_d.repeat(nb_celllines, 1)
            z_pair = torch.cat([z_d_rep, z_c_rep], dim=1)
            logits = model.predictor(z_pair).view(-1)

            batch_lbl = lbl_2d[:, batch_drug_indices].reshape(-1)
            batch_eval_mask = mask_2d[:, batch_drug_indices].reshape(-1)

            if batch_eval_mask.sum() > 0:
                all_preds.append(torch.sigmoid(logits[batch_eval_mask]).cpu().numpy())
                all_lbls.append(batch_lbl[batch_eval_mask].cpu().numpy())

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.array([]), np.array([])

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_lbls)
    auc, aupr, f1, acc = metrics_graph(y_true, y_pred)
    return auc, aupr, f1, acc, y_true, y_pred


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_fold_metrics(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    records = frame.to_dict(orient="records")
    out: List[Dict] = []
    for record in records:
        normalized = dict(record)
        if "fold" in normalized:
            normalized["fold"] = int(normalized["fold"])
        for key in [
            "best_val_auc",
            "test_auc",
            "test_aupr",
            "test_f1",
            "test_acc",
            "parameter_count",
            "fold_elapsed_seconds",
            "peak_gpu_memory_bytes",
        ]:
            if key in normalized and pd.notna(normalized[key]):
                if key in {"parameter_count", "peak_gpu_memory_bytes"}:
                    normalized[key] = int(normalized[key])
                else:
                    normalized[key] = float(normalized[key])
        for key in ["checkpoint", "history_csv", "predictions_npz"]:
            if key in normalized and pd.isna(normalized[key]):
                normalized[key] = ""
        out.append(normalized)
    return out


def fold_result_is_complete(fold_result: Dict, require_checkpoint: bool = False) -> bool:
    history_csv = str(fold_result.get("history_csv", "") or "")
    predictions_npz = str(fold_result.get("predictions_npz", "") or "")
    checkpoint = str(fold_result.get("checkpoint", "") or "")
    required_paths = [history_csv, predictions_npz]
    if require_checkpoint and checkpoint:
        required_paths.append(checkpoint)
    return all(path and os.path.isfile(path) for path in required_paths)


def save_partial_fold_metrics(path: str, fold_metrics: List[Dict]) -> None:
    if not fold_metrics:
        return
    write_csv(
        path,
        fieldnames=[
            "fold",
            "best_val_auc",
            "test_auc",
            "test_aupr",
            "test_f1",
            "test_acc",
            "parameter_count",
            "fold_elapsed_seconds",
            "peak_gpu_memory_bytes",
            "checkpoint",
            "history_csv",
            "predictions_npz",
        ],
        rows=sorted(fold_metrics, key=lambda row: int(row["fold"])),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locked flexible FUSE-CDR trainer (best single architecture)."
    )

    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--fusion_dim", type=int, default=512, help="Intermediate fusion dimension for Cell Line Module")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--drug_num_gnn_layers", type=int, default=3)

    parser.add_argument("--drug_batch_size", type=int, default=0, help="0 for full-batch (matching regular model)")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--contrastive_weight", type=float, default=0.005)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--max_contrastive_pairs", type=int, default=2048)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to a FUSECDR-compatible dataset directory. Defaults to final_dataset at the repository root.",
    )
    parser.add_argument(
        "--omics",
        nargs="+",
        default=None,
        help=(
            "Subset of omics inputs for this run. "
            "Each token can be a full CSV stem like 'genomics_mutation' "
            "or a category like 'genomics' to include all its subtypes."
        ),
    )
    parser.add_argument(
        "--list_omics",
        action="store_true",
        help="List available omics selectors discovered in the selected dataset_root and exit.",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save per-fold best model checkpoints. Keep disabled unless a downstream experiment needs them.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Optional directory containing fold_<n>/{train,val,test}.csv files to use instead of internal random splits.",
    )
    parser.add_argument(
        "--fold_ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit fold IDs to run when --split_dir is provided.",
    )

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def resolve_external_fold_ids(split_dir: str, requested_fold_ids: Optional[List[int]]) -> List[int]:
    if requested_fold_ids:
        return sorted(set(int(fold_id) for fold_id in requested_fold_ids))

    fold_ids: List[int] = []
    for entry in os.listdir(split_dir):
        if not entry.startswith("fold_"):
            continue
        try:
            fold_ids.append(int(entry.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(set(fold_ids))


def load_external_split_tables(split_dir: str, fold_id: int) -> Dict[str, pd.DataFrame]:
    fold_dir = os.path.join(split_dir, f"fold_{fold_id}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    tables = {}
    for split_name in ("train", "val", "test"):
        path = os.path.join(fold_dir, f"{split_name}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing split table: {path}")
        tables[split_name] = pd.read_csv(path)
    return tables


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset_root = args.dataset_root or os.path.join(parent_dir, "final_dataset")
    dataset_root = os.path.abspath(dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset directory not found at: {dataset_root}")

    split_dir = None
    fold_ids: Optional[List[int]] = None
    if args.split_dir:
        split_dir = os.path.abspath(args.split_dir)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found at: {split_dir}")
        fold_ids = resolve_external_fold_ids(split_dir, args.fold_ids)
        if not fold_ids:
            raise ValueError(f"No fold directories found in split_dir={split_dir}")
    elif args.fold_ids:
        raise ValueError("--fold_ids can only be used together with --split_dir")

    if args.list_omics:
        entries = list_available_omics(dataset_root)
        print("Available omics selectors (use with --omics ...):")
        for entry in entries:
            print(
                f"  - stem={entry['stem']:<28} "
                f"(category={entry['category']}, subtype={entry['subtype']}, file={entry['file']})"
            )
        print("\nCategory selectors are also supported, e.g.: --omics genomics proteomics")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"flexible_best_{timestamp}"
    outputs_root = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_root, exist_ok=True)
    run_dir = os.path.join(outputs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "run.log")
    sys.stdout = Logger(log_file)

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Command: {' '.join(sys.argv)}")
    if split_dir:
        print(f"External splits: {split_dir} | fold_ids={fold_ids}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    start_time = time.time()
    folds_path = os.path.join(run_dir, "fold_metrics.csv")
    existing_completed = {
        int(row["fold"]): row
        for row in load_existing_fold_metrics(folds_path)
        if fold_result_is_complete(row, require_checkpoint=args.save_checkpoints)
    }

    print(f"Loading aligned data from: {dataset_root}")
    loaded = dataload_flexible(dataset_root, selected_omics=args.omics)
    print(
        f"Loaded data: pairs={len(loaded.data_new)}, cells={loaded.nb_celllines}, drugs={loaded.nb_drugs}"
    )
    print(f"Selected omics: {', '.join(loaded.selected_omics_stems)}")

    metadata = (
        ["drug", "cell"],
        [
            ("drug", "responds_to", "cell"),
            ("cell", "similar_to", "cell"),
            ("drug", "similar_to", "drug"),
        ],
    )

    active_fold_ids = fold_ids if fold_ids is not None else list(range(1, args.k_fold + 1))
    all_fold_metrics = []

    for fold_index, fold_id in enumerate(active_fold_ids):
        if fold_id in existing_completed:
            print(f"Fold {fold_id} already complete. Reusing saved outputs.")
            all_fold_metrics.append(existing_completed[fold_id])
            continue

        print("\n" + "=" * 80)
        print(f"Fold {fold_index + 1}/{len(active_fold_ids)} (fold_id={fold_id})")
        print("=" * 80)

        split_tables = load_external_split_tables(split_dir, fold_id) if split_dir else None
        fold_start_time = time.time()

        processed = process_flexible(
            loaded=loaded,
            k_folds=args.k_fold,
            current_fold=max(fold_id - 1, 0),
            data_split_seed=args.seed,
            drug_batch_size=args.drug_batch_size,
            split_tables=split_tables,
        )

        encoder_configs = build_encoder_configs(
            omics_tensors=processed.omics_tensors,
            fusion_dim=args.fusion_dim,
        )

        model = FUSECDR(
            atom_shape=processed.atom_shape,
            encoder_configs=encoder_configs,
            metadata=metadata,
            hidden_dim=args.hidden_channels,
            output_dim=args.output_channels,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            num_layers=args.num_layers,
            heads=args.heads,
            drug_num_gnn_layers=args.drug_num_gnn_layers,
        ).to(device)
        parameter_count = count_trainable_parameters(model)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        contrastive_criterion = SupConLoss(temperature=args.temperature)

        omics_data_device = move_omics_to_device(processed.omics_tensors, device)
        label_pos = processed.label_pos.to(device)
        train_mask = processed.train_mask.to(device)
        val_mask = processed.val_mask.to(device)
        test_mask = processed.test_mask.to(device)
        train_edge_tensor = torch.tensor(processed.train_edge, dtype=torch.long, device=device)

        global_edge_index_dict = build_hetero_global_graph(
            cell_similarity_tensor=processed.similarity_tensor.to(device),
            drug_phys_tensor=processed.physicochemical_tensor.to(device),
            top_k=args.top_k,
            device=device,
        )

        best_val_auc = -1.0
        best_state_dict = None
        history_rows = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            train_loss, cls_loss, cont_loss = train_one_epoch(
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
                contrastive_weight=args.contrastive_weight,
                warmup_epochs=args.warmup_epochs,
                epoch=epoch,
                max_contrastive_pairs=args.max_contrastive_pairs,
                device=device,
            )
            epoch_elapsed = time.time() - epoch_start_time

            val_auc, val_aupr, val_f1, val_acc, _, _ = evaluate_split(
                model=model,
                drug_loader=processed.drug_loader,
                omics_data_device=omics_data_device,
                train_edge_tensor=train_edge_tensor,
                eval_mask=val_mask,
                label_pos=label_pos,
                nb_celllines=processed.nb_celllines,
                nb_drugs=processed.nb_drugs,
                global_edge_index_dict=global_edge_index_dict,
                device=device,
            )

            history_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_cls_loss": cls_loss,
                    "train_contrastive_loss": cont_loss,
                    "val_auc": val_auc,
                    "val_aupr": val_aupr,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "epoch_seconds": float(epoch_elapsed),
                }
            )

            print(
                f"Epoch {epoch + 1:03d}/{args.epoch} | "
                f"TrainLoss={train_loss:.4f} (Cls={cls_loss:.4f}, Cont={cont_loss:.4f}) | "
                f"Val AUC={val_auc:.4f} AUPR={val_aupr:.4f} F1={val_f1:.4f} ACC={val_acc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state_dict = copy.deepcopy(model.state_dict())

        if best_state_dict is None:
            raise RuntimeError("No best model state captured during training")

        ckpt_path = ""
        if args.save_checkpoints:
            ckpt_path = os.path.join(run_dir, f"fold_{fold_id}_best.pt")
            torch.save(best_state_dict, ckpt_path)
        model.load_state_dict(best_state_dict)

        test_auc, test_aupr, test_f1, test_acc, y_true, y_pred = evaluate_split(
            model=model,
            drug_loader=processed.drug_loader,
            omics_data_device=omics_data_device,
            train_edge_tensor=train_edge_tensor,
            eval_mask=test_mask,
            label_pos=label_pos,
            nb_celllines=processed.nb_celllines,
            nb_drugs=processed.nb_drugs,
            global_edge_index_dict=global_edge_index_dict,
            device=device,
        )

        history_path = os.path.join(run_dir, f"fold_{fold_id}_history.csv")
        write_csv(
            history_path,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_cls_loss",
                "train_contrastive_loss",
                "val_auc",
                "val_aupr",
                "val_f1",
                "val_acc",
                "epoch_seconds",
            ],
            rows=history_rows,
        )

        pred_path = os.path.join(run_dir, f"fold_{fold_id}_test_predictions.npz")
        np.savez(pred_path, y_true=y_true, y_pred=y_pred)
        fold_elapsed = time.time() - fold_start_time
        peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )

        fold_result = {
            "fold": fold_id,
            "best_val_auc": float(best_val_auc),
            "test_auc": float(test_auc),
            "test_aupr": float(test_aupr),
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "parameter_count": parameter_count,
            "fold_elapsed_seconds": float(fold_elapsed),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "checkpoint": ckpt_path,
            "history_csv": history_path,
            "predictions_npz": pred_path,
        }
        all_fold_metrics.append(fold_result)
        save_partial_fold_metrics(folds_path, all_fold_metrics)

        print(
            f"Fold {fold_id} complete | "
            f"Best Val AUC={best_val_auc:.4f} | "
            f"Test AUC={test_auc:.4f} AUPR={test_aupr:.4f} F1={test_f1:.4f} ACC={test_acc:.4f} | "
            f"FoldTime={fold_elapsed:.2f}s"
        )

    metric_keys = ["test_auc", "test_aupr", "test_f1", "test_acc"]
    mean_metrics = {
        key: float(np.mean([row[key] for row in all_fold_metrics])) for key in metric_keys
    }
    std_metrics = {
        key: float(np.std([row[key] for row in all_fold_metrics])) for key in metric_keys
    }

    elapsed = time.time() - start_time

    all_fold_metrics = sorted(all_fold_metrics, key=lambda row: int(row["fold"]))
    save_partial_fold_metrics(folds_path, all_fold_metrics)

    summary = {
        "run_dir": run_dir,
        "dataset_root": dataset_root,
        "split_dir": split_dir,
        "elapsed_seconds": float(elapsed),
        "k_fold": int(len(active_fold_ids)),
        "fold_ids": active_fold_ids,
        "mean": mean_metrics,
        "std": std_metrics,
        "system": {
            "mean_fold_elapsed_seconds": float(np.mean([row["fold_elapsed_seconds"] for row in all_fold_metrics])),
            "max_peak_gpu_memory_bytes": int(max([row["peak_gpu_memory_bytes"] for row in all_fold_metrics], default=0)),
            "parameter_count": int(all_fold_metrics[0]["parameter_count"]) if all_fold_metrics else 0,
        },
        "folds": all_fold_metrics,
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("Locked Flexible FUSE-CDR Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Elapsed: {elapsed:.2f} s\n")
        f.write(f"K-Fold: {args.k_fold}\n")
        f.write(f"Mean AUC:  {mean_metrics['test_auc']:.4f} +/- {std_metrics['test_auc']:.4f}\n")
        f.write(f"Mean AUPR: {mean_metrics['test_aupr']:.4f} +/- {std_metrics['test_aupr']:.4f}\n")
        f.write(f"Mean F1:   {mean_metrics['test_f1']:.4f} +/- {std_metrics['test_f1']:.4f}\n")
        f.write(f"Mean ACC:  {mean_metrics['test_acc']:.4f} +/- {std_metrics['test_acc']:.4f}\n")

    print("\n" + "=" * 80)
    print("Cross-validation complete")
    print(f"Mean AUC:  {mean_metrics['test_auc']:.4f} +/- {std_metrics['test_auc']:.4f}")
    print(f"Mean AUPR: {mean_metrics['test_aupr']:.4f} +/- {std_metrics['test_aupr']:.4f}")
    print(f"Mean F1:   {mean_metrics['test_f1']:.4f} +/- {std_metrics['test_f1']:.4f}")
    print(f"Mean ACC:  {mean_metrics['test_acc']:.4f} +/- {std_metrics['test_acc']:.4f}")
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
