import torch
import torch.nn as nn
import time
import argparse
import random
import numpy as np

from model import GraphCDR, Encoder, Summary, NodeRepresentation
from model_GPDRP import NodeRepresentationGPDRP
from model_GCLM_CDR import GCLMCDR
import importlib.util
import sys
import os
from data_load import dataload
from data_process import process
from utils import *

# Import baseline modules with aliases to avoid conflicts
from model_baseline import GraphCDR as BaselineGraphCDR, Encoder as BaselineEncoder, Summary as BaselineSummary, NodeRepresentation as BaselineNodeRepresentation
from data_process_baseline import process as baseline_process


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def run_modified_experiment(args, data_split_seed, model_seed, drug_feature, genomics_feature, 
                            epigenomics_feature, transcriptomics_feature, proteomics_feature,
                            metabolomics_feature, pathway_feature, data_new, nb_celllines, 
                            nb_drugs, physicochemical_feature=None, verbose=True):
    """Run a single experiment for the modified architecture"""
    # Set seed for model initialization
    set_seed(model_seed)
    
    # Detect and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Disable modalities based on flags
    if not args.use_genomics:
        genomics_feature = None
        if verbose:
            print("Genomics modality disabled by flag")
    if not args.use_epigenomics:
        epigenomics_feature = None
        if verbose:
            print("Epigenomics modality disabled by flag")
    if not args.use_transcriptomics:
        transcriptomics_feature = None
        if verbose:
            print("Transcriptomics modality disabled by flag")
    if not args.use_proteomics:
        proteomics_feature = None
        if verbose:
            print("Proteomics modality disabled by flag")
    if not args.use_metabolomics:
        metabolomics_feature = None
        if verbose:
            print("Metabolomics modality disabled by flag")
    if not args.use_pathway:
        pathway_feature = None
        if verbose:
            print("Pathway modality disabled by flag")

    # Build loaders, masks, and edges
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
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
        independent_test=True,
        test_ratio=0.1,
        data_split_seed=data_split_seed,
        physicochemical_feature=physicochemical_feature,
    )

    # Infer input dims for NodeRepresentation
    genomics_dim = genomics_feature.shape[1] if genomics_feature is not None else 1
    epigenomics_in_channels = 1 if epigenomics_feature is not None else 1
    transcriptomics_dim = transcriptomics_feature.shape[1] if transcriptomics_feature is not None else 1
    proteomics_dim = proteomics_feature.shape[1] if proteomics_feature is not None else 1
    metabolomics_dim = metabolomics_feature.shape[1] if metabolomics_feature is not None else 1
    pathway_dim = pathway_feature.shape[1] if pathway_feature is not None else 1

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
    
    # Move data tensors to device
    # Note: train_edge is kept as numpy array - the model's forward method converts it internally
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    label_pos = label_pos.to(device)
    if physicochemical_tensor is not None:
        physicochemical_tensor = physicochemical_tensor.to(device)

    def train():
        model.train()
        loss_temp = 0
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            optimizer.zero_grad()
            # Move data to device
            drug.x = drug.x.to(device)
            drug.edge_index = drug.edge_index.to(device)
            drug.batch = drug.batch.to(device)
            cell[0] = cell[0].to(device)
            cell[1] = cell[1].to(device)
            cell[2] = cell[2].to(device)
            
            model_kwargs = {
                'proteomics_data': cell[3].to(device) if cell[3] is not None else None,
                'metabolomics_data': cell[4].to(device) if cell[4] is not None else None,
                'pathway_data': cell[5].to(device) if cell[5] is not None else None,
            }
            if args.active:
                model_kwargs['physicochemical_features'] = physicochemical_tensor
            
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                drug.x,
                drug.edge_index,
                drug.batch,
                cell[0],
                cell[1],
                cell[2],
                train_edge,
                **model_kwargs
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

    def test():
        model.eval()
        with torch.no_grad():
            for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
                # Move data to device
                drug.x = drug.x.to(device)
                drug.edge_index = drug.edge_index.to(device)
                drug.batch = drug.batch.to(device)
                cell[0] = cell[0].to(device)
                cell[1] = cell[1].to(device)
                cell[2] = cell[2].to(device)
                
                model_kwargs = {
                    'proteomics_data': cell[3].to(device) if cell[3] is not None else None,
                    'metabolomics_data': cell[4].to(device) if cell[4] is not None else None,
                    'pathway_data': cell[5].to(device) if cell[5] is not None else None,
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
                    train_edge,
                    **model_kwargs
                )
                loss_temp = myloss(pre_adj[test_mask], label_pos[test_mask])
            yp = pre_adj[test_mask].detach().cpu().numpy()
            ytest = label_pos[test_mask].detach().cpu().numpy()
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            if verbose:
                print('test loss: ', str(round(loss_temp.item(), 4)))
                print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
                    '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
        return AUC, AUPR, F1, ACC

    # Main training loop
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        AUC, AUPR, F1, ACC = test()
        if AUC > final_AUC:
            final_AUC = AUC; final_AUPR = AUPR; final_F1 = F1; final_ACC = ACC

    return final_AUC, final_AUPR, final_F1, final_ACC


def run_baseline_experiment(args, data_split_seed, model_seed, drug_feature, genomics_feature,
                            epigenomics_feature, transcriptomics_feature, data_new, nb_celllines,
                            nb_drugs, verbose=True):
    """Run a single experiment for the baseline (graphCDR) architecture"""
    # Set seed for model initialization
    set_seed(model_seed)
    
    # Detect and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Map loaded data to baseline format
    mutation_feature = genomics_feature
    gexpr_feature = transcriptomics_feature
    methylation_feature = epigenomics_feature
    
    # Process data using baseline processing
    drug_set, cellline_set, train_edge, label_pos, train_mask, test_mask, atom_shape = baseline_process(
        drug_feature, mutation_feature, gexpr_feature, methylation_feature,
        data_new, nb_celllines, nb_drugs, data_split_seed=data_split_seed
    )
    
    # Build baseline model
    model = BaselineGraphCDR(
        hidden_channels=args.hidden_channels,
        encoder=BaselineEncoder(args.output_channels, args.hidden_channels),
        summary=BaselineSummary(args.output_channels, args.hidden_channels),
        feat=BaselineNodeRepresentation(atom_shape, gexpr_feature.shape[-1], methylation_feature.shape[-1], args.output_channels),
        index=nb_celllines
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    myloss = nn.BCELoss()
    
    # Move data tensors to device
    # Note: train_edge is kept as numpy array - the model's forward method converts it internally
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    label_pos = label_pos.to(device)
    
    def train():
        model.train()
        loss_temp = 0
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            optimizer.zero_grad()
            # Move data to device
            drug.x = drug.x.to(device)
            drug.edge_index = drug.edge_index.to(device)
            drug.batch = drug.batch.to(device)
            cell[0] = cell[0].to(device)
            cell[1] = cell[1].to(device)
            cell[2] = cell[2].to(device)
            
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
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
    
    def test():
        model.eval()
        with torch.no_grad():
            for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
                # Move data to device
                drug.x = drug.x.to(device)
                drug.edge_index = drug.edge_index.to(device)
                drug.batch = drug.batch.to(device)
                cell[0] = cell[0].to(device)
                cell[1] = cell[1].to(device)
                cell[2] = cell[2].to(device)
                
                _, _, _, _, pre_adj = model(
                    drug.x, drug.edge_index, drug.batch, cell[0], cell[1], cell[2], train_edge
                )
                loss_temp = myloss(pre_adj[test_mask], label_pos[test_mask])
            yp = pre_adj[test_mask].detach().cpu().numpy()
            ytest = label_pos[test_mask].detach().cpu().numpy()
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            if verbose:
                print('test loss: ', str(round(loss_temp.item(), 4)))
                print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
                    '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
        return AUC, AUPR, F1, ACC
    
    # Main training loop
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        AUC, AUPR, F1, ACC = test()
        if AUC > final_AUC:
            final_AUC = AUC; final_AUPR = AUPR; final_F1 = F1; final_ACC = ACC
    
    return final_AUC, final_AUPR, final_F1, final_ACC


def run_gpdrp_experiment(args, data_split_seed, model_seed, drug_feature, genomics_feature,
                         epigenomics_feature, transcriptomics_feature, proteomics_feature,
                         metabolomics_feature, pathway_feature, data_new, nb_celllines,
                         nb_drugs, verbose=True):
    """Run a single experiment for the GPDRP architecture"""
    # Set seed for model initialization
    set_seed(model_seed)
    
    # Detect and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Build loaders, masks, and edges
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        test_mask,
        atom_shape,
        _physico_ignored,
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
        independent_test=True,
        test_ratio=0.1,
        data_split_seed=data_split_seed,
        physicochemical_feature=None,
    )

    # Input dims for GPDRP encoders: atom_shape and pathway_dim
    pathway_dim = pathway_feature.shape[1] if pathway_feature is not None else 0
    if pathway_dim <= 0:
        raise ValueError("GPDRP execution requires --use_pathway and a valid pathway CSV")

    # Build GraphCDR model but with GPDRP node representation
    model = GraphCDR(
        hidden_channels=args.hidden_channels,
        encoder=Encoder(args.output_channels, args.hidden_channels),
        summary=Summary(args.output_channels, args.hidden_channels),
        feat=NodeRepresentationGPDRP(
            atom_shape,
            genomics_dim=0,
            epigenomics_in_channels=0,
            transcriptomics_dim=0,
            proteomics_dim=0,
            metabolomics_dim=0,
            pathway_dim=pathway_dim,
            gnn_type=args.gnn_type,
            output=args.output_channels,
        ),
        index=nb_celllines,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    myloss = nn.BCELoss()
    
    # Move data tensors to device
    # Note: train_edge is kept as numpy array - the model's forward method converts it internally
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    label_pos = label_pos.to(device)

    def train():
        model.train()
        loss_temp = 0
        for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
            optimizer.zero_grad()
            # Move data to device
            drug.x = drug.x.to(device)
            drug.edge_index = drug.edge_index.to(device)
            drug.batch = drug.batch.to(device)
            cell[0] = cell[0].to(device)
            cell[1] = cell[1].to(device)
            cell[2] = cell[2].to(device)
            
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                drug.x,
                drug.edge_index,
                drug.batch,
                cell[0],
                cell[1],
                cell[2],
                train_edge,
                proteomics_data=cell[3].to(device) if cell[3] is not None else None,
                metabolomics_data=cell[4].to(device) if cell[4] is not None else None,
                pathway_data=cell[5].to(device) if cell[5] is not None else None,
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

    def test():
        model.eval()
        with torch.no_grad():
            for batch, (drug, cell) in enumerate(zip(drug_set, cellline_set)):
                # Move data to device
                drug.x = drug.x.to(device)
                drug.edge_index = drug.edge_index.to(device)
                drug.batch = drug.batch.to(device)
                cell[0] = cell[0].to(device)
                cell[1] = cell[1].to(device)
                cell[2] = cell[2].to(device)
                
                _, _, _, _, pre_adj = model(
                    drug.x,
                    drug.edge_index,
                    drug.batch,
                    cell[0],
                    cell[1],
                    cell[2],
                    train_edge,
                    proteomics_data=cell[3].to(device) if cell[3] is not None else None,
                    metabolomics_data=cell[4].to(device) if cell[4] is not None else None,
                    pathway_data=cell[5].to(device) if cell[5] is not None else None,
                )
                loss_temp = myloss(pre_adj[test_mask], label_pos[test_mask])
            yp = pre_adj[test_mask].detach().cpu().numpy()
            ytest = label_pos[test_mask].detach().cpu().numpy()
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            if verbose:
                print('test loss: ', str(round(loss_temp.item(), 4)))
                print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
                      '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
        return AUC, AUPR, F1, ACC

    # Main training loop
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        AUC, AUPR, F1, ACC = test()
        if AUC > final_AUC:
            final_AUC = AUC; final_AUPR = AUPR; final_F1 = F1; final_ACC = ACC

    return final_AUC, final_AUPR, final_F1, final_ACC


def run_gclmcdr_experiment(args, data_split_seed, model_seed, drug_feature, genomics_feature,
                           epigenomics_feature, transcriptomics_feature, data_new, nb_celllines,
                           nb_drugs, verbose=True):
    """Run a single experiment for the GCLM_CDR architecture"""
    # Set seed for model initialization
    set_seed(model_seed)
    
    # Detect and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # GCLM_CDR only uses genomics, epigenomics, transcriptomics
    # Build loaders, masks, and edges
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        test_mask,
        atom_shape,
        _physico_ignored,
    ) = process(
        drug_feature,
        genomics_feature,
        epigenomics_feature,
        transcriptomics_feature,
        None,  # proteomics - not used by GCLM_CDR
        None,  # metabolomics - not used
        None,  # pathway - not used
        data_new,
        nb_celllines,
        nb_drugs,
        independent_test=True,
        test_ratio=0.1,
        data_split_seed=data_split_seed,
        physicochemical_feature=None,
    )
    

    # Infer input dims from actual data
    genomics_dim = genomics_feature.shape[1] if genomics_feature is not None else 1
    epigenomics_dim = epigenomics_feature.shape[1] if epigenomics_feature is not None else 1
    transcriptomics_dim = transcriptomics_feature.shape[1] if transcriptomics_feature is not None else 1

    if verbose:
        print(f"GCLM_CDR input dimensions - Genomics: {genomics_dim}, Epigenomics: {epigenomics_dim}, Transcriptomics: {transcriptomics_dim}")

    # Build GCLM_CDR model
    model = GCLMCDR(
        genomics_dim=genomics_dim,
        epigenomics_dim=epigenomics_dim,
        transcriptomics_dim=transcriptomics_dim,
        atom_feature_dim=atom_shape,
        hidden_dim=args.hidden_channels,
        gat_heads=2,
        gat_layers=2,
        lambda_pos=0.3,
        lambda_neg=0.3,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    
    # Convert train_edge to tensor
    # train_edge already contains only training pairs (symmetrized, so each pair appears twice)
    if isinstance(train_edge, np.ndarray):
        train_edge_tensor = torch.from_numpy(train_edge).long().to(device)
    else:
        train_edge_tensor = train_edge.long().to(device)
    
    # Get all cell line and drug data (single batch from loaders)
    drug_batch = next(iter(drug_set))
    cell_batch = next(iter(cellline_set))
    
    # Move drug_batch and cell_batch to device
    drug_batch = drug_batch.to(device)
    cell_batch = tuple(c.to(device) if c is not None else None for c in cell_batch)
    
    # train_edge is symmetrized (each pair appears twice: [cell, drug] and [drug, cell])
    # We only need one direction, so filter to get only [cell, drug] pairs
    # For GCLM-CDR, we work with pairs where cell_idx < nb_celllines and drug_idx >= nb_celllines
    train_pairs_mask = train_edge_tensor[:, 0] < nb_celllines
    train_pairs = train_edge_tensor[train_pairs_mask]  # Get training pairs (only one direction)
    
    train_cell_indices = train_pairs[:, 0].long()
    train_drug_indices = (train_pairs[:, 1] - nb_celllines).long()  # Adjust drug indices (drugs start at nb_celllines)
    train_labels = (train_pairs[:, 2].float() + 1) / 2  # Convert {1, -1} to {0, 1}
    
    # Move test_mask and label_pos to device
    test_mask = test_mask.to(device)
    label_pos = label_pos.to(device)
    
    # Batch size for processing pairs
    pair_batch_size = 64  # Process pairs in batches
    
    def train():
        model.train()
        loss_temp = 0
        pred_loss_total = 0
        pos_loss_total = 0
        neg_loss_total = 0
        
        # Recompute drug embeddings each epoch (they may change during training)
        with torch.no_grad():
            all_drug_embeddings = model.drug_encoder(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        
        # Shuffle training pairs
        perm = torch.randperm(len(train_cell_indices), device=device)
        train_cell_indices_shuffled = train_cell_indices[perm]
        train_drug_indices_shuffled = train_drug_indices[perm]
        train_labels_shuffled = train_labels[perm]
        
        num_batches = (len(train_cell_indices) + pair_batch_size - 1) // pair_batch_size
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            batch_start = batch_idx * pair_batch_size
            batch_end = min((batch_idx + 1) * pair_batch_size, len(train_cell_indices))
            
            batch_cell_indices = train_cell_indices_shuffled[batch_start:batch_end]
            batch_drug_indices = train_drug_indices_shuffled[batch_start:batch_end]
            batch_labels = train_labels_shuffled[batch_start:batch_end]
            
            # Get cell line data for this batch (already on device)
            batch_cell_genomics = cell_batch[0][batch_cell_indices] if cell_batch[0] is not None else None
            batch_cell_epigenomics = cell_batch[1][batch_cell_indices] if cell_batch[1] is not None else None
            batch_cell_transcriptomics = cell_batch[2][batch_cell_indices] if cell_batch[2] is not None else None
            
            # Forward pass with pre-computed drug embeddings
            predictions, h_O, h_D, T_OD, graph_info = model(
                drug_batch.x,
                drug_batch.edge_index,
                drug_batch.batch,
                batch_cell_genomics,
                batch_cell_epigenomics,
                batch_cell_transcriptomics,
                batch_cell_indices,
                batch_drug_indices,
                batch_labels,
                all_drug_embeddings=all_drug_embeddings
            )
            
            # Compute loss
            total_loss, pred_loss, pos_loss, neg_loss = model.compute_loss(
                predictions, batch_labels, h_O, h_D, T_OD, graph_info
            )
            
            total_loss.backward()
            optimizer.step()
            
            loss_temp += total_loss.item()
            pred_loss_total += pred_loss.item()
            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
        
        if verbose and num_batches > 0:
            print('train loss: ', str(round(loss_temp / num_batches, 4)))
            print('  pred loss: ', str(round(pred_loss_total / num_batches, 4)))
            print('  pos loss: ', str(round(pos_loss_total / num_batches, 4)))
            print('  neg loss: ', str(round(neg_loss_total / num_batches, 4)))

    def test():
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Recompute drug embeddings for test (in case model changed)
        with torch.no_grad():
            all_drug_embeddings = model.drug_encoder(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        
        # Get test pairs from test_mask
        # test_mask is flattened (nb_celllines * nb_drugs,)
        # Reconstruct test pairs (already moved to device earlier)
        test_mask_2d = test_mask.view(nb_celllines, nb_drugs)
        test_cell_idx, test_drug_idx = torch.where(test_mask_2d)
        
        if len(test_cell_idx) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Get labels from label_pos
        test_labels_mask = label_pos.view(nb_celllines, nb_drugs)[test_cell_idx, test_drug_idx]
        test_labels = test_labels_mask.float()  # Already in {0, 1} format
        
        test_cell_indices = test_cell_idx
        test_drug_indices = test_drug_idx  # Already 0-indexed for drugs
        
        # Process test pairs in batches
        test_pair_batch_size = 64
        num_test_batches = (len(test_cell_indices) + test_pair_batch_size - 1) // test_pair_batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                batch_start = batch_idx * test_pair_batch_size
                batch_end = min((batch_idx + 1) * test_pair_batch_size, len(test_cell_indices))
                
                batch_test_cell_indices = test_cell_indices[batch_start:batch_end]
                batch_test_drug_indices = test_drug_indices[batch_start:batch_end]
                batch_test_labels = test_labels[batch_start:batch_end]
                
                # Get corresponding data (already on device)
                batch_test_cell_genomics = cell_batch[0][batch_test_cell_indices] if cell_batch[0] is not None else None
                batch_test_cell_epigenomics = cell_batch[1][batch_test_cell_indices] if cell_batch[1] is not None else None
                batch_test_cell_transcriptomics = cell_batch[2][batch_test_cell_indices] if cell_batch[2] is not None else None
                
                # Forward pass with pre-computed drug embeddings
                predictions, _, _, _, _ = model(
                    drug_batch.x,
                    drug_batch.edge_index,
                    drug_batch.batch,
                    batch_test_cell_genomics,
                    batch_test_cell_epigenomics,
                    batch_test_cell_transcriptomics,
                    batch_test_cell_indices,
                    batch_test_drug_indices,
                    batch_test_labels,
                    all_drug_embeddings=all_drug_embeddings
                )
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_test_labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        if all_predictions:
            yp = np.concatenate(all_predictions)
            ytest = np.concatenate(all_labels)
            
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            
            if verbose:
                print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
                      '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
            
            return AUC, AUPR, F1, ACC
        else:
            return 0.0, 0.0, 0.0, 0.0

    # Main training loop
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        AUC, AUPR, F1, ACC = test()
        if AUC > final_AUC:
            final_AUC = AUC; final_AUPR = AUPR; final_F1 = F1; final_ACC = ACC

    return final_AUC, final_AUPR, final_F1, final_ACC


def run_gclmcdr_modified_experiment(args, data_split_seed, model_seed, drug_feature, genomics_feature,
                                     epigenomics_feature, transcriptomics_feature, proteomics_feature,
                                     metabolomics_feature, pathway_feature, data_new, nb_celllines,
                                     nb_drugs, verbose=True):
    """Run a single experiment for the GCLM_CDR_modified architecture"""
    # Set seed for model initialization
    set_seed(model_seed)
    
    # Detect and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Import GCLMCDRModifiable
    from model_GCLM_CDR import GCLMCDRModifiable
    
    # Build loaders, masks, and edges
    (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        test_mask,
        atom_shape,
        _physico_ignored,
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
        independent_test=True,
        test_ratio=0.1,
        data_split_seed=data_split_seed,
        physicochemical_feature=None,
    )
    
    # Infer input dims from actual data
    genomics_dim = genomics_feature.shape[1] if genomics_feature is not None else 1
    epigenomics_dim = epigenomics_feature.shape[1] if epigenomics_feature is not None else 1
    transcriptomics_dim = transcriptomics_feature.shape[1] if transcriptomics_feature is not None else 1
    proteomics_dim = proteomics_feature.shape[1] if proteomics_feature is not None else 0
    metabolomics_dim = metabolomics_feature.shape[1] if metabolomics_feature is not None else 0
    pathway_dim = pathway_feature.shape[1] if pathway_feature is not None else 0
    
    if verbose:
        print(f"GCLM_CDR_modified input dimensions - Genomics: {genomics_dim}, Epigenomics: {epigenomics_dim}, "
              f"Transcriptomics: {transcriptomics_dim}, Proteomics: {proteomics_dim}, "
              f"Metabolomics: {metabolomics_dim}, Pathway: {pathway_dim}")
    
    # Build GCLM_CDR_modified model with custom modules
    model = GCLMCDRModifiable(
        genomics_dim=genomics_dim,
        epigenomics_dim=epigenomics_dim,
        transcriptomics_dim=transcriptomics_dim,
        atom_feature_dim=atom_shape,
        proteomics_dim=proteomics_dim,
        metabolomics_dim=metabolomics_dim,
        pathway_dim=pathway_dim,
        hidden_dim=args.hidden_channels,
        gat_heads=2,
        gat_layers=2,
        lambda_pos=0.3,
        lambda_neg=0.3,
        dropout=0.2,
        use_custom_modules=True,  # Use custom modules
        gnn_type=args.gnn_type if hasattr(args, 'gnn_type') else 'GIN'
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    
    # Convert train_edge to tensor
    if isinstance(train_edge, np.ndarray):
        train_edge_tensor = torch.from_numpy(train_edge).long().to(device)
    else:
        train_edge_tensor = train_edge.long().to(device)
    
    # Get all cell line and drug data (single batch from loaders)
    drug_batch = next(iter(drug_set))
    cell_batch = next(iter(cellline_set))
    
    # Move drug_batch and cell_batch to device
    drug_batch = drug_batch.to(device)
    cell_batch = tuple(c.to(device) if c is not None else None for c in cell_batch)
    
    # train_edge is symmetrized (each pair appears twice: [cell, drug] and [drug, cell])
    # Filter to get only [cell, drug] pairs
    train_pairs_mask = train_edge_tensor[:, 0] < nb_celllines
    train_pairs = train_edge_tensor[train_pairs_mask]  # Get training pairs (only one direction)
    
    train_cell_indices = train_pairs[:, 0].long()
    train_drug_indices = (train_pairs[:, 1] - nb_celllines).long()  # Adjust drug indices
    train_labels = (train_pairs[:, 2].float() + 1) / 2  # Convert {1, -1} to {0, 1}
    
    # Move test_mask and label_pos to device
    test_mask = test_mask.to(device)
    label_pos = label_pos.to(device)
    
    # Batch size for processing pairs
    pair_batch_size = 64  # Process pairs in batches
    
    def train():
        model.train()
        loss_temp = 0
        pred_loss_total = 0
        pos_loss_total = 0
        neg_loss_total = 0
        
        # Pre-compute all drug and cell embeddings once per epoch
        with torch.no_grad():
            from model_GCLM_CDR import convert_pyg_to_user_format
            drug_feature, drug_adj, ibatch = convert_pyg_to_user_format(
                drug_batch.x, drug_batch.edge_index, drug_batch.batch
            )
            all_drug_embeddings = model.drug_module(drug_feature, drug_adj, ibatch)
            
            # Pre-compute all cell embeddings
            all_cell_embeddings = model.cell_module(
                genomics_data=cell_batch[0] if len(cell_batch) > 0 and cell_batch[0] is not None else None,
                epigenomics_data=cell_batch[1] if len(cell_batch) > 1 and cell_batch[1] is not None else None,
                transcriptomics_data=cell_batch[2] if len(cell_batch) > 2 and cell_batch[2] is not None else None,
                proteomics_data=cell_batch[3] if len(cell_batch) > 3 and cell_batch[3] is not None else None,
                metabolomics_data=cell_batch[4] if len(cell_batch) > 4 and cell_batch[4] is not None else None,
                pathway_data=cell_batch[5] if len(cell_batch) > 5 and cell_batch[5] is not None else None
            )
        
        # Shuffle training pairs
        perm = torch.randperm(len(train_cell_indices), device=device)
        train_cell_indices_shuffled = train_cell_indices[perm]
        train_drug_indices_shuffled = train_drug_indices[perm]
        train_labels_shuffled = train_labels[perm]
        
        num_batches = (len(train_cell_indices) + pair_batch_size - 1) // pair_batch_size
        
        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            
            batch_start = batch_idx * pair_batch_size
            batch_end = min((batch_idx + 1) * pair_batch_size, len(train_cell_indices))
            
            batch_cell_indices = train_cell_indices_shuffled[batch_start:batch_end]
            batch_drug_indices = train_drug_indices_shuffled[batch_start:batch_end]
            batch_labels = train_labels_shuffled[batch_start:batch_end]
            
            # Get cell line data for this batch
            batch_cell_genomics = cell_batch[0][batch_cell_indices] if cell_batch[0] is not None else None
            batch_cell_epigenomics = cell_batch[1][batch_cell_indices] if len(cell_batch) > 1 and cell_batch[1] is not None else None
            batch_cell_transcriptomics = cell_batch[2][batch_cell_indices] if len(cell_batch) > 2 and cell_batch[2] is not None else None
            batch_cell_proteomics = cell_batch[3][batch_cell_indices] if len(cell_batch) > 3 and cell_batch[3] is not None else None
            batch_cell_metabolomics = cell_batch[4][batch_cell_indices] if len(cell_batch) > 4 and cell_batch[4] is not None else None
            batch_cell_pathway = cell_batch[5][batch_cell_indices] if len(cell_batch) > 5 and cell_batch[5] is not None else None
            
            # Forward pass with pre-computed embeddings
            predictions, h_O, h_D, T_OD, graph_info = model(
                drug_batch.x,
                drug_batch.edge_index,
                drug_batch.batch,
                batch_cell_genomics,
                batch_cell_epigenomics,
                batch_cell_transcriptomics,
                batch_cell_indices,
                batch_drug_indices,
                batch_labels,
                proteomics_data=batch_cell_proteomics,
                metabolomics_data=batch_cell_metabolomics,
                pathway_data=batch_cell_pathway,
                all_drug_embeddings=all_drug_embeddings,
                all_cell_embeddings=all_cell_embeddings
            )
            
            # Compute loss
            total_loss, pred_loss, pos_loss, neg_loss = model.compute_loss(
                predictions, batch_labels, h_O, h_D, T_OD, graph_info
            )
            
            total_loss.backward()
            optimizer.step()
            
            loss_temp += total_loss.item()
            pred_loss_total += pred_loss.item()
            pos_loss_total += pos_loss.item()
            neg_loss_total += neg_loss.item()
        
        if verbose and num_batches > 0:
            print('train loss: ', str(round(loss_temp / num_batches, 4)))
            print('  pred loss: ', str(round(pred_loss_total / num_batches, 4)))
            print('  pos loss: ', str(round(pos_loss_total / num_batches, 4)))
            print('  neg loss: ', str(round(neg_loss_total / num_batches, 4)))
    
    def test():
        model.eval()
        all_predictions = []
        all_labels = []
        
        # Pre-compute all embeddings for test
        with torch.no_grad():
            from model_GCLM_CDR import convert_pyg_to_user_format
            drug_feature, drug_adj, ibatch = convert_pyg_to_user_format(
                drug_batch.x, drug_batch.edge_index, drug_batch.batch
            )
            all_drug_embeddings = model.drug_module(drug_feature, drug_adj, ibatch)
            
            all_cell_embeddings = model.cell_module(
                genomics_data=cell_batch[0] if len(cell_batch) > 0 and cell_batch[0] is not None else None,
                epigenomics_data=cell_batch[1] if len(cell_batch) > 1 and cell_batch[1] is not None else None,
                transcriptomics_data=cell_batch[2] if len(cell_batch) > 2 and cell_batch[2] is not None else None,
                proteomics_data=cell_batch[3] if len(cell_batch) > 3 and cell_batch[3] is not None else None,
                metabolomics_data=cell_batch[4] if len(cell_batch) > 4 and cell_batch[4] is not None else None,
                pathway_data=cell_batch[5] if len(cell_batch) > 5 and cell_batch[5] is not None else None
            )
        
        # Get test pairs from test_mask (already moved to device earlier)
        test_mask_2d = test_mask.view(nb_celllines, nb_drugs)
        test_cell_idx, test_drug_idx = torch.where(test_mask_2d)
        
        if len(test_cell_idx) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Get labels from label_pos
        test_labels_mask = label_pos.view(nb_celllines, nb_drugs)[test_cell_idx, test_drug_idx]
        test_labels = test_labels_mask.float()  # Already in {0, 1} format
        
        test_cell_indices = test_cell_idx
        test_drug_indices = test_drug_idx  # Already 0-indexed for drugs
        
        # Process test pairs in batches
        test_pair_batch_size = 64
        num_test_batches = (len(test_cell_indices) + test_pair_batch_size - 1) // test_pair_batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_test_batches):
                batch_start = batch_idx * test_pair_batch_size
                batch_end = min((batch_idx + 1) * test_pair_batch_size, len(test_cell_indices))
                
                batch_test_cell_indices = test_cell_indices[batch_start:batch_end]
                batch_test_drug_indices = test_drug_indices[batch_start:batch_end]
                batch_test_labels = test_labels[batch_start:batch_end]
                
                # Get corresponding data
                batch_test_cell_genomics = cell_batch[0][batch_test_cell_indices] if cell_batch[0] is not None else None
                batch_test_cell_epigenomics = cell_batch[1][batch_test_cell_indices] if len(cell_batch) > 1 and cell_batch[1] is not None else None
                batch_test_cell_transcriptomics = cell_batch[2][batch_test_cell_indices] if len(cell_batch) > 2 and cell_batch[2] is not None else None
                batch_test_cell_proteomics = cell_batch[3][batch_test_cell_indices] if len(cell_batch) > 3 and cell_batch[3] is not None else None
                batch_test_cell_metabolomics = cell_batch[4][batch_test_cell_indices] if len(cell_batch) > 4 and cell_batch[4] is not None else None
                batch_test_cell_pathway = cell_batch[5][batch_test_cell_indices] if len(cell_batch) > 5 and cell_batch[5] is not None else None
                
                # Forward pass with pre-computed embeddings
                predictions, _, _, _, _ = model(
                    drug_batch.x,
                    drug_batch.edge_index,
                    drug_batch.batch,
                    batch_test_cell_genomics,
                    batch_test_cell_epigenomics,
                    batch_test_cell_transcriptomics,
                    batch_test_cell_indices,
                    batch_test_drug_indices,
                    batch_test_labels,
                    proteomics_data=batch_test_cell_proteomics,
                    metabolomics_data=batch_test_cell_metabolomics,
                    pathway_data=batch_test_cell_pathway,
                    all_drug_embeddings=all_drug_embeddings,
                    all_cell_embeddings=all_cell_embeddings
                )
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_test_labels.cpu().numpy())
        
        # Concatenate all predictions and labels
        if all_predictions:
            yp = np.concatenate(all_predictions)
            ytest = np.concatenate(all_labels)
            
            AUC, AUPR, F1, ACC = metrics_graph(ytest, yp)
            
            if verbose:
                print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)) +
                      '  test f1: ' + str(round(F1, 4)) + '  test acc: ' + str(round(ACC, 4)))
            
            return AUC, AUPR, F1, ACC
        else:
            return 0.0, 0.0, 0.0, 0.0
    
    # Main training loop
    final_AUC = 0; final_AUPR = 0; final_F1 = 0; final_ACC = 0
    for epoch in range(args.epoch):
        if verbose:
            print('\nepoch: ' + str(epoch))
        train()
        AUC, AUPR, F1, ACC = test()
        if AUC > final_AUC:
            final_AUC = AUC; final_AUPR = AUPR; final_F1 = F1; final_ACC = ACC

    return final_AUC, final_AUPR, final_F1, final_ACC


def run_gclmcdr_modified_multiseed_experiment(args, seeds, drug_feature, genomics_feature, epigenomics_feature,
                                               transcriptomics_feature, proteomics_feature, metabolomics_feature,
                                               pathway_feature, data_new, nb_celllines, nb_drugs, verbose=True):
    """
    Run GCLM-CDR modified experiment with multiple seeds and report statistics.
    This is a wrapper function specifically for GCLM_CDR_modified architecture.
    """
    results = {
        'AUC': [],
        'AUPR': [],
        'F1': [],
        'ACC': []
    }
    
    print(f"\n{'='*60}")
    print(f"Running GCLM-CDR Modified experiments with {len(seeds)} seeds: {seeds}")
    print(f"{'='*60}\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed {i}/{len(seeds)}: {seed}")
        print(f"{'='*60}\n")
        
        # Run single-seed experiment
        AUC, AUPR, F1, ACC = run_gclmcdr_modified_experiment(
            args, data_split_seed=seed, model_seed=seed,
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
            verbose=verbose
        )
        
        results['AUC'].append(AUC)
        results['AUPR'].append(AUPR)
        results['F1'].append(F1)
        results['ACC'].append(ACC)
        
        print(f"\nSeed {seed} Results:")
        print(f"  AUC: {AUC:.4f}, AUPR: {AUPR:.4f}, F1: {F1:.4f}, ACC: {ACC:.4f}")
    
    # Calculate statistics
    stats = {}
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        values = np.array(results[metric])
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values.tolist()
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("GCLM-CDR Modified SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        s = stats[metric]
        print(f"{metric}:")
        print(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")
    
    # Save to file
    output_file = "GCLM_CDR_modified_multi_seed_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"GCLM-CDR Modified Multi-Seed Experiment Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Number of seeds: {len(seeds)}\n")
        f.write(f"{'='*60}\n\n")
        for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
            s = stats[metric]
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}\n")
            f.write(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]\n")
            f.write(f"  Values: {s['values']}\n\n")
    
    print(f"\nResults saved to {output_file}")
    
    return stats


def run_gclmcdr_multiseed_experiment(args, seeds, drug_feature, genomics_feature, epigenomics_feature,
                                     transcriptomics_feature, data_new, nb_celllines, nb_drugs, verbose=True):
    """
    Run GCLM-CDR experiment with multiple seeds and report statistics.
    This is a wrapper function specifically for GCLM_CDR architecture.
    
    Args:
        args: Argument parser object
        seeds: List of seeds to use
        drug_feature: Drug feature dictionary
        genomics_feature: Genomics feature array
        epigenomics_feature: Epigenomics feature array
        transcriptomics_feature: Transcriptomics feature array
        data_new: Path to response data CSV
        nb_celllines: Number of cell lines
        nb_drugs: Number of drugs
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    results = {
        'AUC': [],
        'AUPR': [],
        'F1': [],
        'ACC': []
    }
    
    print(f"\n{'='*60}")
    print(f"Running GCLM-CDR experiments with {len(seeds)} seeds: {seeds}")
    print(f"{'='*60}\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed {i}/{len(seeds)}: {seed}")
        print(f"{'='*60}\n")
        
        # Run single-seed experiment
        AUC, AUPR, F1, ACC = run_gclmcdr_experiment(
            args, data_split_seed=seed, model_seed=seed,
            drug_feature=drug_feature,
            genomics_feature=genomics_feature,
            epigenomics_feature=epigenomics_feature,
            transcriptomics_feature=transcriptomics_feature,
            data_new=data_new,
            nb_celllines=nb_celllines,
            nb_drugs=nb_drugs,
            verbose=verbose
        )
        
        results['AUC'].append(AUC)
        results['AUPR'].append(AUPR)
        results['F1'].append(F1)
        results['ACC'].append(ACC)
        
        print(f"\nSeed {seed} Results:")
        print(f"  AUC: {AUC:.4f}, AUPR: {AUPR:.4f}, F1: {F1:.4f}, ACC: {ACC:.4f}")
    
    # Calculate statistics
    stats = {}
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        values = np.array(results[metric])
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values.tolist()
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("GCLM-CDR SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        s = stats[metric]
        print(f"{metric}:")
        print(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")
    
    # Save to file
    output_file = "GCLM_CDR_multi_seed_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"GCLM-CDR Multi-Seed Experiment Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Number of seeds: {len(seeds)}\n")
        f.write(f"{'='*60}\n\n")
        for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
            s = stats[metric]
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}\n")
            f.write(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]\n")
            f.write(f"  Values: {s['values']}\n\n")
    
    print(f"\nResults saved to {output_file}")
    
    return stats


def run_multiple_seeds(args, seeds, drug_feature, genomics_feature, epigenomics_feature,
                       transcriptomics_feature, proteomics_feature, metabolomics_feature,
                       pathway_feature, data_new, nb_celllines, nb_drugs, physicochemical_feature=None):
    """
    Run experiment with multiple seeds and report statistics.
    
    Args:
        args: Argument parser object
        seeds: List of seeds to use
        ... (other data parameters)
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    results = {
        'AUC': [],
        'AUPR': [],
        'F1': [],
        'ACC': []
    }
    
    # Handle GCLM_CDR separately since it has its own dedicated multi-seed function
    if args.execution_architecture == 'GCLM_CDR':
        return run_gclmcdr_multiseed_experiment(
            args, seeds=seeds,
            drug_feature=drug_feature,
            genomics_feature=genomics_feature,
            epigenomics_feature=epigenomics_feature,
            transcriptomics_feature=transcriptomics_feature,
            data_new=data_new,
            nb_celllines=nb_celllines,
            nb_drugs=nb_drugs,
            verbose=True
        )
    
    if args.execution_architecture == 'GCLM_CDR_modified':
        return run_gclmcdr_modified_multiseed_experiment(
            args, seeds=seeds,
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
            verbose=True
        )
    
    print(f"\n{'='*60}")
    print(f"Running experiments with {len(seeds)} seeds: {seeds}")
    print(f"Architecture: {args.execution_architecture}")
    print(f"{'='*60}\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed {i}/{len(seeds)}: {seed}")
        print(f"{'='*60}\n")
        
        # Run experiment based on architecture
        if args.execution_architecture == 'modified':
            AUC, AUPR, F1, ACC = run_modified_experiment(
                args, data_split_seed=seed, model_seed=seed,
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
                verbose=True
            )
        elif args.execution_architecture == 'graphCDR':
            AUC, AUPR, F1, ACC = run_baseline_experiment(
                args, data_split_seed=seed, model_seed=seed,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                verbose=True
            )
        elif args.execution_architecture == 'GPDRP':
            AUC, AUPR, F1, ACC = run_gpdrp_experiment(
                args, data_split_seed=seed, model_seed=seed,
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
                verbose=True
            )
        else:
            raise ValueError(f"Multi-seed experiments only supported for 'modified', 'graphCDR', 'GPDRP', and 'GCLM_CDR' architectures")
        
        results['AUC'].append(AUC)
        results['AUPR'].append(AUPR)
        results['F1'].append(F1)
        results['ACC'].append(ACC)
        
        print(f"\nSeed {seed} Results:")
        print(f"  AUC: {AUC:.4f}, AUPR: {AUPR:.4f}, F1: {F1:.4f}, ACC: {ACC:.4f}")
    
    # Calculate statistics
    stats = {}
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        values = np.array(results[metric])
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values.tolist()
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL SEEDS")
    print(f"{'='*60}")
    for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
        s = stats[metric]
        print(f"{metric}:")
        print(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  Values: {[f'{v:.4f}' for v in s['values']]}")
    
    # Save to file
    output_file = f"{args.execution_architecture}_multi_seed_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Multi-Seed Experiment Results ({len(seeds)} seeds)\n")
        f.write(f"Architecture: {args.execution_architecture}\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"{'='*60}\n\n")
        
        for metric in ['AUC', 'AUPR', 'F1', 'ACC']:
            s = stats[metric]
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std: {s['mean']:.4f} ± {s['std']:.4f}\n")
            f.write(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]\n")
            f.write(f"  Values: {s['values']}\n\n")
    
    print(f"\nResults saved to {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='GraphCDR with modified NodeRepresentation')
    parser.add_argument('--alph', type=float, default=0.30)
    parser.add_argument('--beta', type=float, default=0.30)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--output_channels', type=int, default=100)

    # Paths (placeholders — replace with your actual files/dirs)
    parser.add_argument('--drug_feature_dir', type=str, default='../new_data/GDSC/Drug/drug_graph_feat')
    parser.add_argument('--response_csv', type=str, default='../new_data/GDSC/Processed data/response_labels.csv')
    parser.add_argument('--genomics_csv', type=str, default='../new_data/CCLE/Processed data/mutation_data.csv')
    parser.add_argument('--epigenomics_csv', type=str, default='../new_data/CCLE/Processed data/DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv')
    parser.add_argument('--transcriptomics_csv', type=str, default='../new_data/CCLE/Processed data/gene_expression_data.csv')
    parser.add_argument('--proteomics_csv', type=str, default='../new_data/CCLE/Processed data/reverse_phase_protein_array_data.csv')
    parser.add_argument('--metabolomics_csv', type=str, default='../new_data/CCLE/Processed data/metabolomics_data.csv')
    parser.add_argument('--pathway_csv', type=str, default='../new_data/CCLE/Processed data/cell_pathway_scores_from_gene_expression.csv')

    # Modality flags
    parser.add_argument('--use_genomics', action='store_true', default=True, help='Use genomics data')
    parser.add_argument('--use_epigenomics', action='store_true', default=True, help='Use epigenomics data')
    parser.add_argument('--use_transcriptomics', action='store_true', default=True, help='Use transcriptomics data')
    parser.add_argument('--use_proteomics', action='store_true', default=False, help='Use proteomics data')
    parser.add_argument('--use_metabolomics', action='store_true', default=False, help='Use metabolomics data')
    parser.add_argument('--use_pathway', action='store_true', default=False, help='Use pathway data')

    # GNN type
    parser.add_argument('--gnn_type', type=str, default='GIN', choices=['GIN', 'GCN', 'GraphSAGE', 'GAT'], help='Type of GNN layer for drug representation')

    # Drug representation enhancement
    parser.add_argument('--active', action='store_true', default=False, help='Use enhanced drug representation with physicochemical features and cross-attention')
    parser.add_argument('--physicochemical_csv', type=str, default='../new_data/GDSC/Processed data/pubchem_physiochemical_properties.csv', help='Path to physicochemical properties CSV')
    parser.add_argument('--use_transformer_drug', action='store_true', default=False, help='Use GPDRP-style 2GIN+1Transformer architecture for drug representation')

    # Cell line module variation
    parser.add_argument('--cell_line_module_variation', type=str, default='original', choices=['original', 'VAE', 'FC'], help='Cell line processing method: original (FC+attention), VAE (VAE+attention), or FC (FC+FC fusion)')

    # Execution Architecture
    parser.add_argument('--execution_architecture', type=str, default='modified', choices=['modified', 'graphCDR', 'GPDRP', 'GCLM_CDR', 'GCLM_CDR_modified'])
    
    # Multi-seed support (for modified, graphCDR, GPDRP, and GCLM_CDR)
    parser.add_argument('--multi_seed', action='store_true', default=False, 
                       help='Run experiment with multiple seeds and report statistics (for modified, graphCDR, GPDRP, and GCLM_CDR)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999],
                       help='List of seeds to use for multi-seed experiments (default: 42 123 456 789 999)')
    parser.add_argument('--single_seed', type=int, default=666,
                       help='Single seed to use if not running multi-seed (default: 666)')

    args = parser.parse_args()

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
    )

    # ----- Load physicochemical features if active=True
    physicochemical_feature = None
    if args.active:
        import pandas as pd
        print(f"Loading physicochemical features from: {args.physicochemical_csv}")
        physicochemical_df = pd.read_csv(args.physicochemical_csv, sep=',', header=0)
        # Check if column is named 'pubchem_cid' or similar
        cid_col = [col for col in physicochemical_df.columns if 'pubchem' in col.lower() or 'cid' in col.lower()]
        if not cid_col:
            raise ValueError("Could not find pubchem ID column in physicochemical CSV")
        physicochemical_df = physicochemical_df.set_index(cid_col[0])
        physicochemical_df = physicochemical_df.astype(float)
        print(f"Loaded {len(physicochemical_df)} drugs with {physicochemical_df.shape[1]} physicochemical features")
        # Convert to dict for easy lookup
        physicochemical_feature = physicochemical_df.to_dict('index')

    if args.execution_architecture == 'modified':
        if args.multi_seed:
            # Run with multiple seeds
            stats = run_multiple_seeds(
                args, seeds=args.seeds,
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
                physicochemical_feature=physicochemical_feature
            )
        else:
            # Single seed run (original behavior)
            set_seed(args.single_seed)
            final_AUC, final_AUPR, final_F1, final_ACC = run_modified_experiment(
                args, data_split_seed=args.single_seed, model_seed=args.single_seed,
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
                verbose=True
            )
            
            elapsed = time.time() - start_time
            output_file = "results.txt"
            with open(output_file, 'a') as f:
                f.write('---------------------------------------\n')
                f.write(f'Elapsed time: {round(elapsed, 4)}\n')
                f.write(f'Seed: {args.single_seed}\n')
                f.write('Final_AUC: ' + str(round(final_AUC, 4)) + 
                        '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                        '  Final_F1: ' + str(round(final_F1, 4)) + 
                        '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
                f.write('---------------------------------------\n')

    elif args.execution_architecture == 'graphCDR':
        print("\n=== Running Baseline GraphCDR ===")
        
        if args.multi_seed:
            # Run with multiple seeds
            stats = run_multiple_seeds(
                args, seeds=args.seeds,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                proteomics_feature=None,  # Baseline doesn't use these
                metabolomics_feature=None,
                pathway_feature=None,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                physicochemical_feature=None
            )
        else:
            # Single seed run (original behavior)
            set_seed(args.single_seed)
            final_AUC, final_AUPR, final_F1, final_ACC = run_baseline_experiment(
                args, data_split_seed=args.single_seed, model_seed=args.single_seed,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                verbose=True
            )
            
            elapsed = time.time() - start_time
            output_file = "baseline_results.txt"
            with open(output_file, 'a') as f:
                f.write('---------------------------------------\n')
                f.write(f'Elapsed time: {round(elapsed, 4)}\n')
                f.write(f'Seed: {args.single_seed}\n')
                f.write('Final_AUC: ' + str(round(final_AUC, 4)) + 
                        '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                        '  Final_F1: ' + str(round(final_F1, 4)) + 
                        '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
                f.write('---------------------------------------\n')
    elif args.execution_architecture == 'GPDRP':
        print("\n=== Running GPDRP ===")
        
        if args.multi_seed:
            # Run with multiple seeds
            stats = run_multiple_seeds(
                args, seeds=args.seeds,
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
                physicochemical_feature=None
            )
        else:
            # Single seed run (original behavior)
            set_seed(args.single_seed)
            final_AUC, final_AUPR, final_F1, final_ACC = run_gpdrp_experiment(
                args, data_split_seed=args.single_seed, model_seed=args.single_seed,
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
                verbose=True
            )
            
            elapsed = time.time() - start_time
            output_file = "modified_node_representation_results.txt"
            with open(output_file, 'a') as f:
                f.write('---------------------------------------\n')
                f.write(f'Elapsed time: {round(elapsed, 4)}\n')
                f.write(f'Seed: {args.single_seed}\n')
                f.write('Final_AUC: ' + str(round(final_AUC, 4)) +
                        '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                        '  Final_F1: ' + str(round(final_F1, 4)) +
                        '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
                f.write('---------------------------------------\n')
    elif args.execution_architecture == 'GCLM_CDR':
        print("\n=== Running GCLM_CDR ===")
        
        # GCLM_CDR only uses genomics, epigenomics, transcriptomics
        if args.multi_seed:
            # Run with multiple seeds using dedicated function
            stats = run_gclmcdr_multiseed_experiment(
                args, seeds=args.seeds,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                verbose=True
            )
        else:
            # Single seed run
            set_seed(args.single_seed)
            final_AUC, final_AUPR, final_F1, final_ACC = run_gclmcdr_experiment(
                args, data_split_seed=args.single_seed, model_seed=args.single_seed,
                drug_feature=drug_feature,
                genomics_feature=genomics_feature,
                epigenomics_feature=epigenomics_feature,
                transcriptomics_feature=transcriptomics_feature,
                data_new=data_new,
                nb_celllines=nb_celllines,
                nb_drugs=nb_drugs,
                verbose=True
            )
            
            elapsed = time.time() - start_time
            output_file = "GCLM_CDR_results.txt"
            with open(output_file, 'a') as f:
                f.write('---------------------------------------\n')
                f.write(f'Elapsed time: {round(elapsed, 4)}\n')
                f.write(f'Seed: {args.single_seed}\n')
                f.write('Final_AUC: ' + str(round(final_AUC, 4)) + 
                        '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                        '  Final_F1: ' + str(round(final_F1, 4)) + 
                        '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
                f.write('---------------------------------------\n')
    
    elif args.execution_architecture == 'GCLM_CDR_modified':
        print("\n=== Running GCLM_CDR_modified ===")
        
        # GCLM_CDR_modified uses all modalities (custom modules)
        if args.multi_seed:
            # Run with multiple seeds using dedicated function
            stats = run_gclmcdr_modified_multiseed_experiment(
                args, seeds=args.seeds,
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
                verbose=True
            )
        else:
            # Single seed run
            start_time = time.time()
            set_seed(args.single_seed)
            final_AUC, final_AUPR, final_F1, final_ACC = run_gclmcdr_modified_experiment(
                args, data_split_seed=args.single_seed, model_seed=args.single_seed,
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
                verbose=True
            )
            print(f"\nFinal Results: AUC={final_AUC:.4f}, AUPR={final_AUPR:.4f}, F1={final_F1:.4f}, ACC={final_ACC:.4f}")
            
            elapsed = time.time() - start_time
            output_file = "GCLM_CDR_modified_results.txt"
            with open(output_file, 'a') as f:
                f.write('---------------------------------------\n')
                f.write(f'Elapsed time: {round(elapsed, 4)}\n')
                f.write(f'Seed: {args.single_seed}\n')
                f.write('Final_AUC: ' + str(round(final_AUC, 4)) + 
                        '  Final_AUPR: ' + str(round(final_AUPR, 4)) +
                        '  Final_F1: ' + str(round(final_F1, 4)) + 
                        '  Final_ACC: ' + str(round(final_ACC, 4)) + '\n')
                f.write('---------------------------------------\n')
    
    else:
        raise ValueError(f"Unsupported execution architecture: {args.execution_architecture}")

if __name__ == '__main__':
    main()