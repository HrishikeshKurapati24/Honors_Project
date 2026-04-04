import torch
import torch.nn as nn
import time
import argparse
import numpy as np

from model import GraphCDR, Encoder, Summary, NodeRepresentation
import sys
import os
from data_load import dataload
from data_process import process
from utils import *

# Import baseline modules with aliases to avoid conflicts
from model_baseline import GraphCDR as BaselineGraphCDR, Encoder as BaselineEncoder, Summary as BaselineSummary, NodeRepresentation as BaselineNodeRepresentation

FIXED_SEED = 0


def set_fixed_seed():
    """Use a fixed torch seed across runs to match baseline-style behavior."""
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# REQUIRED COLUMNS IN THE RESPONSE FILE: cell_line_id, pubchem_id, label, label is 1 if the cell line is sensitive to the drug, -1 if the cell line is resistant to the drug.
# REQUIRED COLUMNS IN THE Omics FILES: index_col=[0](cell line index), columns are the features of the omics data
# REQUIRED COLUMNS IN THE DRUG FILE: index_col=[0](drug index), columns are the features of the drug
# DRUG FEATURE FILE IS A HKL FILE NAMED <pubchem>.hkl
# DRUG FEATURE FILE CONTAINS: feat_mat, adj_list, degree_list
# FEAT_MAT IS A 2D ARRAY OF FEATURES, ROWS ARE THE ATOMS, COLUMNS ARE THE FEATURES
# ADJ_LIST IS A LIST OF ADJACENCY LISTS, EACH LIST IS A LIST OF ATOMS THAT ARE CONNECTED TO THE ATOM
# DEGREE_LIST IS A LIST OF DEGREES OF THE ATOMS


def print_active_modalities(args):
    print("\n=== Active Modalities ===")
    print(f"Genomics: {'✓' if args.use_genomics else '✗'}")
    print(f"Epigenomics: {'✓' if args.use_epigenomics else '✗'}")
    print(f"Transcriptomics: {'✓' if args.use_transcriptomics else '✗'}")
    print(f"Proteomics: {'✓' if args.use_proteomics else '✗'}")
    print(f"Metabolomics: {'✓' if args.use_metabolomics else '✗'}")
    print(f"Pathway: {'✓' if args.use_pathway else '✗'}")
    print(f"Enhanced Drug Representation: {'✓' if args.active else '✗'}")
    print(f"Transformer Drug Architecture (2GIN+1Transformer): {'✓' if args.use_transformer_drug else '✗'}")
    print(f"Cell Line Module Variation: {args.cell_line_module_variation}")
    print("========================\n")


def _normalize_feature_matrix(feature):
    """Treat empty DataFrames as disabled modality inputs."""
    if feature is None:
        return None
    if hasattr(feature, "empty") and feature.empty:
        return None
    return feature


def _apply_modality_flag(feature, enabled, name, verbose=False):
    feature = _normalize_feature_matrix(feature)
    if not enabled:
        if verbose:
            print(f"{name} modality disabled by flag")
        return None
    if feature is None and verbose:
        print(f"{name} modality enabled but matrix is empty; treating as disabled")
    return feature


def _feature_dim(feature):
    feature = _normalize_feature_matrix(feature)
    return feature.shape[1] if feature is not None else 0


def _resolve_runtime_device(args):
    cached_device = getattr(args, "_runtime_device", None)
    if cached_device is not None:
        return cached_device

    requested = getattr(args, "device", "auto")
    require_gpu = bool(getattr(args, "require_gpu", False))
    cuda_available = torch.cuda.is_available()

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested via --device cuda, but no CUDA device is available.")
        device = torch.device("cuda")
    elif requested == "cpu":
        if require_gpu:
            raise ValueError("--require_gpu cannot be combined with --device cpu.")
        device = torch.device("cpu")
    else:
        if cuda_available:
            device = torch.device("cuda")
        else:
            if require_gpu:
                raise RuntimeError("--require_gpu was set, but CUDA is not available.")
            device = torch.device("cpu")

    setattr(args, "_runtime_device", device)
    return device


def _resolve_pin_memory(args, device):
    return bool(getattr(args, "pin_memory", False) and device.type == "cuda")


def _resolve_non_blocking(device, pin_memory):
    return device.type == "cuda" and pin_memory


def _precompute_edge_tensors(edge, device):
    if isinstance(edge, (tuple, list)) and len(edge) == 2:
        pos_edge, neg_edge = edge
        return (
            pos_edge.to(device=device, dtype=torch.long).contiguous(),
            neg_edge.to(device=device, dtype=torch.long).contiguous(),
        )

    if isinstance(edge, torch.Tensor):
        edge_tensor = edge.to(device=device, dtype=torch.long)
    else:
        edge_tensor = torch.as_tensor(edge, dtype=torch.long, device=device)

    if edge_tensor.dim() != 2 or edge_tensor.size(1) < 3:
        raise ValueError("Expected edge array/tensor with shape [N, 3] where column 2 is the label.")

    pos_edge = edge_tensor[edge_tensor[:, 2] == 1, 0:2].t().contiguous()
    neg_edge = edge_tensor[edge_tensor[:, 2] == -1, 0:2].t().contiguous()
    return pos_edge, neg_edge


def _move_batch_to_device(drug, cell, device, non_blocking=False):
    drug = drug.to(device, non_blocking=non_blocking)
    cell = [c.to(device, non_blocking=non_blocking) if c is not None else None for c in cell]
    return drug, cell


def _assert_tensor_device(name, tensor, expected_device):
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.device.type != expected_device.type:
        raise RuntimeError(f"{name} is on {tensor.device}, expected {expected_device}.")


def run_modified_experiment(args, drug_feature, genomics_feature,
                            epigenomics_feature, transcriptomics_feature, proteomics_feature,
                            metabolomics_feature, pathway_feature, data_new, nb_celllines, 
                            nb_drugs, physicochemical_feature=None, verbose=True, k_folds=1, current_fold=0):
    """Run a single experiment for the modified architecture"""
    set_fixed_seed()
    
    device = _resolve_runtime_device(args)
    pin_memory = _resolve_pin_memory(args, device)
    non_blocking = _resolve_non_blocking(device, pin_memory)
    if verbose:
        print(f"Using device: {device} (pin_memory={pin_memory}, num_workers={args.num_workers})")
    
    # Resolve feature matrices from flags and empty DataFrames.
    genomics_feature = _apply_modality_flag(genomics_feature, args.use_genomics, "Genomics", verbose)
    epigenomics_feature = _apply_modality_flag(epigenomics_feature, args.use_epigenomics, "Epigenomics", verbose)
    transcriptomics_feature = _apply_modality_flag(transcriptomics_feature, args.use_transcriptomics, "Transcriptomics", verbose)
    proteomics_feature = _apply_modality_flag(proteomics_feature, args.use_proteomics, "Proteomics", verbose)
    metabolomics_feature = _apply_modality_flag(metabolomics_feature, args.use_metabolomics, "Metabolomics", verbose)
    pathway_feature = _apply_modality_flag(pathway_feature, args.use_pathway, "Pathway", verbose)

    # Build loaders, masks, and edges
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        val_mask,
        test_mask,
        atom_shape,
        physicochemical_tensor,
    ) = process(
        drug_feature,
        genomics_feature,
        epigenomics_feature,
        transcriptomics_feature,
        proteomics_feature,
        metabolomics_feature,
        pathway_feature,
        data_new,
        nb_celllines,
        nb_drugs,
        independent_test=False,
        test_ratio=0.1,
        data_split_seed=FIXED_SEED,
        k_folds=k_folds,
        current_fold=current_fold,
        physicochemical_feature=physicochemical_feature,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    train_edge_pair = _precompute_edge_tensors(train_edge, device)

    # Infer input dims for NodeRepresentation
    genomics_dim = _feature_dim(genomics_feature)
    epigenomics_in_channels = _feature_dim(epigenomics_feature)
    transcriptomics_dim = _feature_dim(transcriptomics_feature)
    proteomics_dim = _feature_dim(proteomics_feature)
    metabolomics_dim = _feature_dim(metabolomics_feature)
    pathway_dim = _feature_dim(pathway_feature)

    if verbose:
        print("Pathway dimensions: " + str(pathway_dim))

    # Build model
    model = GraphCDR(
        hidden_channels=args.hidden_channels,
        encoder=Encoder(args.output_channels, args.hidden_channels),
        summary=Summary(args.output_channels, args.hidden_channels),
        feat=NodeRepresentation(
            atom_shape,
            genomics_dim,
            epigenomics_in_channels,
            transcriptomics_dim,
            proteomics_dim,
            metabolomics_dim,
            pathway_dim,
            args.gnn_type,
            args.output_channels,
            active=args.active,
            variation=args.cell_line_module_variation,
            use_transformer_drug=args.use_transformer_drug,
        ),
        index=nb_celllines,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    myloss = nn.BCELoss()
    
    # Move long-lived tensors to the target device once.
    train_mask = train_mask.to(device, non_blocking=non_blocking)
    val_mask = val_mask.to(device, non_blocking=non_blocking)
    test_mask = test_mask.to(device, non_blocking=non_blocking)
    label_pos = label_pos.to(device, non_blocking=non_blocking)
    if physicochemical_tensor is not None:
        physicochemical_tensor = physicochemical_tensor.to(device, non_blocking=non_blocking)

    first_batch_checked = False

    def train():
        nonlocal first_batch_checked
        model.train()
        loss_temp = 0
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            optimizer.zero_grad()
            drug, cell = _move_batch_to_device(drug, cell, device, non_blocking=non_blocking)
            
            model_kwargs = {
                'proteomics_data': cell[3] if cell[3] is not None else None,
                'metabolomics_data': cell[4] if cell[4] is not None else None,
                'pathway_data': cell[5] if cell[5] is not None else None,
            }
            if args.active:
                model_kwargs['physicochemical_features'] = physicochemical_tensor

            if args.device_assert and device.type == "cuda" and not first_batch_checked:
                _assert_tensor_device("drug.x", drug.x, device)
                _assert_tensor_device("drug.edge_index", drug.edge_index, device)
                _assert_tensor_device("cell.genomics", cell[0], device)
                _assert_tensor_device("cell.epigenomics", cell[1], device)
                _assert_tensor_device("cell.transcriptomics", cell[2], device)
                _assert_tensor_device("train_mask", train_mask, device)
                _assert_tensor_device("label_pos", label_pos, device)
                _assert_tensor_device("train_edge.pos", train_edge_pair[0], device)
                _assert_tensor_device("train_edge.neg", train_edge_pair[1], device)
                if args.active:
                    _assert_tensor_device("physicochemical_tensor", physicochemical_tensor, device)
                if verbose:
                    print("Device assert passed on first modified-model batch.")
                first_batch_checked = True
            
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                drug.x,
                drug.edge_index,
                drug.batch,
                cell[0],
                cell[1],
                cell[2],
                train_edge_pair,
                **model_kwargs
            )
            dgi_pos = model.loss(pos_z, neg_z, summary_pos)
            dgi_neg = model.loss(neg_z, pos_z, summary_neg)
            pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
            loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
            if args.cell_line_module_variation == 'AE':
                ae_recon = model.get_autoencoder_loss()
                if ae_recon is not None:
                    loss = loss + args.ae_recon_weight * ae_recon
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
        if verbose:
            print('train loss: ', str(round(loss_temp, 4)))

    def evaluate(eval_mask, split_name):
        model.eval()
        with torch.no_grad():
            for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
                drug, cell = _move_batch_to_device(drug, cell, device, non_blocking=non_blocking)
                
                model_kwargs = {
                    'proteomics_data': cell[3] if cell[3] is not None else None,
                    'metabolomics_data': cell[4] if cell[4] is not None else None,
                    'pathway_data': cell[5] if cell[5] is not None else None,
                }
                if args.active:
                    model_kwargs['physicochemical_features'] = physicochemical_tensor
                
                _, _, _, _, pre_adj = model(
                    drug.x,
                    drug.edge_index,
                    drug.batch,
                    cell[0],
                    cell[1],
                    cell[2],
                    train_edge_pair,
                    **model_kwargs
                )
                loss_temp = myloss(pre_adj[eval_mask], label_pos[eval_mask])
            yp = pre_adj[eval_mask].detach().cpu().numpy()
            ytest = label_pos[eval_mask].detach().cpu().numpy()
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            if verbose:
                print(f'{split_name} loss: ', str(round(loss_temp.item(), 4)))
                print(
                    f'{split_name} auc: ' + str(round(AUC, 4)) +
                    '  ' + f'{split_name} aupr: ' + str(round(AUPR, 4)) +
                    '  ' + f'{split_name} f1: ' + str(round(F1, 4)) +
                    '  ' + f'{split_name} acc: ' + str(round(ACC, 4))
                )
        return AUC, AUPR, F1, ACC

    import copy
    best_val_auc = -1.0
    best_model_weights = None

    # Main training loop
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        val_AUC, val_AUPR, val_F1, val_ACC = evaluate(val_mask, "val")
        
        if val_AUC > best_val_auc:
            best_val_auc = val_AUC
            # Deepcopy to save the actual tensor data, not just references
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best weights before final test 
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        if verbose:
            print(f"\n[Model Selection] Loaded best model weights (Validation AUC: {best_val_auc:.4f})")

    final_AUC, final_AUPR, final_F1, final_ACC = evaluate(test_mask, "test")
    return final_AUC, final_AUPR, final_F1, final_ACC


def run_baseline_experiment(args, drug_feature, genomics_feature,
                            epigenomics_feature, transcriptomics_feature, data_new, nb_celllines,
                            nb_drugs, verbose=True, k_folds=1, current_fold=0):
    """Run a single experiment for the baseline (graphCDR) architecture"""
    set_fixed_seed()
    
    device = _resolve_runtime_device(args)
    pin_memory = _resolve_pin_memory(args, device)
    non_blocking = _resolve_non_blocking(device, pin_memory)
    if verbose:
        print(f"Using device: {device} (pin_memory={pin_memory}, num_workers={args.num_workers})")

    genomics_feature = _normalize_feature_matrix(genomics_feature)
    epigenomics_feature = _normalize_feature_matrix(epigenomics_feature)
    transcriptomics_feature = _normalize_feature_matrix(transcriptomics_feature)
    if genomics_feature is None or epigenomics_feature is None or transcriptomics_feature is None:
        raise ValueError(
            "Baseline graphCDR requires non-empty genomics, epigenomics, and transcriptomics features. "
            "Enable --use_genomics --use_epigenomics --use_transcriptomics."
        )
    
    # Process data using unified processing to guarantee exact subset matched evaluation
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        val_mask,
        test_mask,
        atom_shape,
        _physico_ignored,
    ) = process(
        drug_feature,
        genomics_feature,
        epigenomics_feature,
        transcriptomics_feature,
        None,  # proteomics - not used
        None,  # metabolomics - not used
        None,  # pathway - not used
        data_new,
        nb_celllines,
        nb_drugs,
        independent_test=False,
        test_ratio=0.1,
        data_split_seed=FIXED_SEED,
        k_folds=k_folds,
        current_fold=current_fold,
        physicochemical_feature=None,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    train_edge_pair = _precompute_edge_tensors(train_edge, device)
    
    # Map the unified tensor batches correctly to Baseline components (Mutation, GExp, Methy)
    # The unified DataLoader packs: (genomics_tensor, epigenomics_tensor, transcriptomics_tensor, ...)
    # Baseline expects: mutation(1), gexpr(expr), methy(epi)
    # But cellline_set packs them as (0:genomics, 1:epi, 2:trans, 3:prot, 4:meta, 5:pathway)
    # So we need to rebuild the cellline_set with only 3 specific tensors for baseline loops
    
    # Actually, the baseline process expected (drug, mutation, expr, epi).
    # Since we use the Unified loader, we unpack the tuples in train/test loops accordingly.
    
    model = BaselineGraphCDR(
        hidden_channels=args.hidden_channels,
        encoder=BaselineEncoder(args.output_channels, args.hidden_channels),
        summary=BaselineSummary(args.output_channels, args.hidden_channels),
        feat=BaselineNodeRepresentation(atom_shape, transcriptomics_feature.shape[-1], epigenomics_feature.shape[-1], args.output_channels),
        index=nb_celllines
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    myloss = nn.BCELoss()
    
    train_mask = train_mask.to(device, non_blocking=non_blocking)
    val_mask = val_mask.to(device, non_blocking=non_blocking)
    test_mask = test_mask.to(device, non_blocking=non_blocking)
    label_pos = label_pos.to(device, non_blocking=non_blocking)

    first_batch_checked = False
    
    def train():
        nonlocal first_batch_checked
        model.train()
        loss_temp = 0
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            optimizer.zero_grad()
            drug, cell = _move_batch_to_device(drug, cell, device, non_blocking=non_blocking)
            
            # Baseline expects mutation to be 4D for Conv2D (N, 1, 1, Features)
            mutation_data = torch.unsqueeze(cell[0], dim=1)
            mutation_data = torch.unsqueeze(mutation_data, dim=1)

            if args.device_assert and device.type == "cuda" and not first_batch_checked:
                _assert_tensor_device("drug.x", drug.x, device)
                _assert_tensor_device("drug.edge_index", drug.edge_index, device)
                _assert_tensor_device("cell.genomics", cell[0], device)
                _assert_tensor_device("cell.epigenomics", cell[1], device)
                _assert_tensor_device("cell.transcriptomics", cell[2], device)
                _assert_tensor_device("train_mask", train_mask, device)
                _assert_tensor_device("label_pos", label_pos, device)
                _assert_tensor_device("train_edge.pos", train_edge_pair[0], device)
                _assert_tensor_device("train_edge.neg", train_edge_pair[1], device)
                if verbose:
                    print("Device assert passed on first baseline batch.")
                first_batch_checked = True
            
            # Baseline expects: cell[0](mutation), cell[1](gexpr), cell[2](methylation)
            # Unified loader returns: cell[0](genomics/mutation), cell[1](epigenomics/methylation), cell[2](transcript/gexpr)
            # Therefore we map: mutation=mutation_data, gexpr=cell[2], methylation=cell[1]
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                drug.x, drug.edge_index, drug.batch, mutation_data, cell[2], cell[1], train_edge
            )
            dgi_pos = model.loss(pos_z, neg_z, summary_pos)
            dgi_neg = model.loss(neg_z, pos_z, summary_neg)
            pos_loss = myloss(pos_adj[train_mask], label_pos[train_mask])
            loss = (1 - args.alph - args.beta) * pos_loss + args.alph * dgi_pos + args.beta * dgi_neg
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
        if verbose:
            print('train loss: ', str(round(loss_temp, 4)))
    
    def evaluate(eval_mask, split_name):
        model.eval()
        with torch.no_grad():
            for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
                drug, cell = _move_batch_to_device(drug, cell, device, non_blocking=non_blocking)
                
                # Baseline expects mutation to be 4D for Conv2D (N, 1, 1, Features)
                mutation_data = torch.unsqueeze(cell[0], dim=1)
                mutation_data = torch.unsqueeze(mutation_data, dim=1)
                
                _, _, _, _, pre_adj = model(
                    drug.x, drug.edge_index, drug.batch, mutation_data, cell[2], cell[1], train_edge
                )
                loss_temp = myloss(pre_adj[eval_mask], label_pos[eval_mask])
            yp = pre_adj[eval_mask].detach().cpu().numpy()
            ytest = label_pos[eval_mask].detach().cpu().numpy()
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            if verbose:
                print(f'{split_name} loss: ', str(round(loss_temp.item(), 4)))
                print(
                    f'{split_name} auc: ' + str(round(AUC, 4)) +
                    '  ' + f'{split_name} aupr: ' + str(round(AUPR, 4)) +
                    '  ' + f'{split_name} f1: ' + str(round(F1, 4)) +
                    '  ' + f'{split_name} acc: ' + str(round(ACC, 4))
                )
        return AUC, AUPR, F1, ACC
    
    import copy
    best_val_auc = -1.0
    best_model_weights = None

    # Main training loop
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        val_AUC, val_AUPR, val_F1, val_ACC = evaluate(val_mask, "val")
        
        if val_AUC > best_val_auc:
            best_val_auc = val_AUC
            # Deepcopy to save the actual tensor data, not just references
            best_model_weights = copy.deepcopy(model.state_dict())

    # Load best weights before final test 
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        if verbose:
            print(f"\n[Model Selection] Loaded best model weights (Validation AUC: {best_val_auc:.4f})")

    final_AUC, final_AUPR, final_F1, final_ACC = evaluate(test_mask, "test")
    return final_AUC, final_AUPR, final_F1, final_ACC


def main():
    parser = argparse.ArgumentParser(description='GraphCDR with modified NodeRepresentation')
    parser.add_argument('--alph', type=float, default=0.30)
    parser.add_argument('--beta', type=float, default=0.30)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--output_channels', type=int, default=100)

    # Paths (placeholders — replace with your actual files/dirs)
    parser.add_argument('--drug_feature_dir', type=str, default='../../final_dataset/drug_graph_feat')
    parser.add_argument('--response_csv', type=str, default='../../final_dataset/response_pairs.csv')
    parser.add_argument('--genomics_csv', type=str, default='../../final_dataset/genomics_mutation.csv')
    parser.add_argument('--epigenomics_csv', type=str, default='../../final_dataset/epigenomics.csv')
    parser.add_argument('--transcriptomics_csv', type=str, default='../../final_dataset/transcriptomics.csv')
    parser.add_argument('--proteomics_csv', type=str, default='../../final_dataset/proteomics.csv')
    parser.add_argument('--metabolomics_csv', type=str, default='../../final_dataset/metabolomics.csv')
    parser.add_argument('--pathway_csv', type=str, default='../../final_dataset/pathway.csv')

    # Modality flags
    parser.add_argument('--use_genomics', action='store_true', default=False, help='Use genomics data')
    parser.add_argument('--use_epigenomics', action='store_true', default=False, help='Use epigenomics data')
    parser.add_argument('--use_transcriptomics', action='store_true', default=False, help='Use transcriptomics data')
    parser.add_argument('--use_proteomics', action='store_true', default=False, help='Use proteomics data')
    parser.add_argument('--use_metabolomics', action='store_true', default=False, help='Use metabolomics data')
    parser.add_argument('--use_pathway', action='store_true', default=False, help='Use pathway data')

    # GNN type
    parser.add_argument('--gnn_type', type=str, default='GIN', choices=['GIN', 'GCN', 'GraphSAGE', 'GAT'], help='Type of GNN layer for drug representation')

    # Drug representation enhancement
    parser.add_argument('--active', action='store_true', default=False, help='Use enhanced drug representation with physicochemical features and cross-attention')
    # Default to the consolidated final dataset physicochemical file
    parser.add_argument('--physicochemical_csv', type=str, default='../../final_dataset/physicochemical.csv', help='Path to physicochemical properties CSV')
    parser.add_argument('--use_transformer_drug', action='store_true', default=False, help='Use transformer-enhanced architecture for drug representation')

    # Cell line module variation
    parser.add_argument('--cell_line_module_variation', type=str, default='original', choices=['original', 'AE', 'FC'], help='Cell line processing method: original (FC+attention), AE (AE+attention), or FC (FC+FC fusion)')
    parser.add_argument('--ae_recon_weight', type=float, default=0.1, help='Weight for AE reconstruction loss when --cell_line_module_variation AE')

    # Execution  Architecture
    parser.add_argument('--execution_architecture', type=str, default='modified', choices=['modified', 'graphCDR'])
    parser.add_argument('--k_fold', type=int, default=1, help='Number of folds for cross-validation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Runtime device selection for modified and graphCDR experiments')
    parser.add_argument('--require_gpu', action='store_true', default=False,
                       help='Fail fast if CUDA is unavailable')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader worker count for modified and graphCDR experiments')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                       help='Enable pinned host memory for faster host-to-device copies')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                       help='Disable pinned host memory in DataLoaders')
    parser.add_argument('--device_assert', action='store_true', default=False,
                       help='Assert first training batch tensors are on the expected device')
    parser.set_defaults(pin_memory=False)

    args = parser.parse_args()
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if args.k_fold <= 0:
        raise ValueError("--k_fold must be >= 1")
    if args.device == "cpu" and args.require_gpu:
        raise ValueError("--require_gpu cannot be combined with --device cpu")
    runtime_device = _resolve_runtime_device(args)
    effective_pin_memory = _resolve_pin_memory(args, runtime_device)
    print(
        f"Runtime config -> requested_device={args.device}, resolved_device={runtime_device}, "
        f"require_gpu={args.require_gpu}, num_workers={args.num_workers}, pin_memory={effective_pin_memory}"
    )

    # Validate minimum modalities requirement
    enabled_modalities = sum([
        args.use_genomics,
        args.use_epigenomics,
        args.use_transcriptomics,
        args.use_proteomics,
        args.use_metabolomics,
        args.use_pathway
    ])

    print(f"\nUsing {enabled_modalities} modalities for training\n")
    
    # Print active modalities
    print_active_modalities(args)
    
    start_time = time.time()

    # ----- Load data (drug HKLs + multi-omics + response labels)
    (
        drug_feature,
        genomics_feature,
        epigenomics_feature,
        transcriptomics_feature,
        proteomics_feature,
        metabolomics_feature,
        pathway_feature,
        data_new,
        nb_celllines,
        nb_drugs,
    ) = dataload(
        args.drug_feature_dir,
        args.response_csv,
        args.genomics_csv,
        args.epigenomics_csv,
        args.transcriptomics_csv,
        args.proteomics_csv,
        args.metabolomics_csv,
        args.pathway_csv,
        use_genomics=args.use_genomics,
        use_epigenomics=args.use_epigenomics,
        use_transcriptomics=args.use_transcriptomics,
        use_proteomics=args.use_proteomics,
        use_metabolomics=args.use_metabolomics,
        use_pathway=args.use_pathway,
    )

    # ----- Load physicochemical features if active=True
    physicochemical_feature = None
    if args.active:
        import os
        import pandas as pd
        # Resolve physicochemical CSV path with robust fallbacks relative to this script
        script_dir = os.path.dirname(__file__)
        candidate_paths = [
            args.physicochemical_csv,
            os.path.join(script_dir, '../../final_dataset/physicochemical.csv'),
            os.path.join(script_dir, '../final_dataset/physicochemical.csv'),
        ]
        resolved_csv = None
        for p in candidate_paths:
            if os.path.exists(p):
                resolved_csv = p
                break
        if resolved_csv is None:
            searched_list = "\n - " + "\n - ".join(candidate_paths)
            raise FileNotFoundError(
                f"Physicochemical CSV not found while --active is enabled.\n"
                f"Searched paths:{searched_list}\n"
                f"Provide --physicochemical_csv <path> or place the file under new_data/GDSC/Processed data."
            )
        print(f"Loading physicochemical features from: {resolved_csv}")
        physicochemical_df = pd.read_csv(resolved_csv, sep=',', header=0)
        # Check if column is named 'pubchem_cid' or similar
        cid_col = [col for col in physicochemical_df.columns if 'pubchem' in col.lower() or 'cid' in col.lower()]
        if not cid_col:
            raise ValueError("Could not find pubchem ID column in physicochemical CSV")
        physicochemical_df = physicochemical_df.set_index(cid_col[0])
        physicochemical_df = physicochemical_df.astype(float)
        print(f"Loaded {len(physicochemical_df)} drugs with {physicochemical_df.shape[1]} physicochemical features")
        # Convert to dict for easy lookup
        physicochemical_feature = physicochemical_df.to_dict('index')

    def _run_kfold(experiment_fn, architecture_name):
        fold_metrics = []
        for fold in range(args.k_fold):
            if args.k_fold > 1:
                print(f"\n--- Running Fold {fold + 1}/{args.k_fold} for {architecture_name} ---")
            metrics = experiment_fn(current_fold=fold)
            fold_metrics.append(metrics)

        fold_metrics = np.array(fold_metrics, dtype=float)
        mean_metrics = np.mean(fold_metrics, axis=0)
        std_metrics = np.std(fold_metrics, axis=0)
        return mean_metrics, std_metrics

    if args.execution_architecture == 'modified':
        mean_metrics, std_metrics = _run_kfold(
            lambda current_fold: run_modified_experiment(
                args,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                proteomics_feature=proteomics_feature,
                metabolomics_feature=metabolomics_feature,
                pathway_feature=pathway_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                physicochemical_feature=physicochemical_feature,
                verbose=True,
                k_folds=args.k_fold,
                current_fold=current_fold,
            ),
            "modified",
        )
        output_file = "results.txt"
    elif args.execution_architecture == 'graphCDR':
        print("\n=== Running Baseline GraphCDR ===")
        mean_metrics, std_metrics = _run_kfold(
            lambda current_fold: run_baseline_experiment(
                args,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                verbose=True,
                k_folds=args.k_fold,
                current_fold=current_fold,
            ),
            "graphCDR",
        )
        output_file = "baseline_results.txt"
    else:
        raise ValueError(
            f"Unsupported execution architecture: {args.execution_architecture}. "
            "Supported: modified, graphCDR"
        )

    elapsed = time.time() - start_time
    final_AUC, final_AUPR, final_F1, final_ACC = mean_metrics
    std_AUC, std_AUPR, std_F1, std_ACC = std_metrics

    print(
        f"\nFinal Results ({args.execution_architecture}) -> "
        f"AUC={final_AUC:.4f}, AUPR={final_AUPR:.4f}, F1={final_F1:.4f}, ACC={final_ACC:.4f}"
    )

    with open(output_file, 'a') as f:
        f.write('---------------------------------------\n')
        f.write(f'Elapsed time: {round(elapsed, 4)}\n')
        f.write(f'Fixed torch seed: {FIXED_SEED}\n')
        if args.k_fold > 1:
            f.write(f'K-Fold CV: {args.k_fold} folds\n')
            f.write('Final_AUC: ' + str(round(final_AUC, 4)) + f' +/- {round(std_AUC, 4)}' +
                    '  Final_AUPR: ' + str(round(final_AUPR, 4)) + f' +/- {round(std_AUPR, 4)}' +
                    '  Final_F1: ' + str(round(final_F1, 4)) + f' +/- {round(std_F1, 4)}' +
                    '  Final_ACC: ' + str(round(final_ACC, 4)) + f' +/- {round(std_ACC, 4)}' + '\n')
        else:
            f.write('Final_AUC: ' + str(round(final_AUC, 4)) +
                    '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                    '  Final_F1: ' + str(round(final_F1, 4)) +
                    '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
        f.write('---------------------------------------\n')

# ==========================================
# Logging Implementation
# ==========================================
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # Needed for python 3 compatibility
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':
    # Initialize logging immediately
    import datetime
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Generate log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join('logs', f'node_experiment_{timestamp}_{os.getpid()}.log')
    
    # Redirect stdout to Logger
    sys.stdout = Logger(log_file)
    
    print(f"Logging started: {log_file}")
    print(f"Command: {' '.join(sys.argv)}")
    print("==========================================")
    main()