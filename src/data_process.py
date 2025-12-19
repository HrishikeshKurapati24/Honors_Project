import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from scipy.sparse import coo_matrix

# We reuse the simple GraphDataset and collate utilities from the existing implementation pattern
from graphset import GraphDataset, collate  # adjust import path if needed


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature_dict):
    drug_data = [[] for _ in range(len(drug_feature_dict))]
    for i in range(len(drug_feature_dict)):
        feat_mat, adj_list, _ = drug_feature_dict.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0:int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask


def process(
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
    data_split_seed=666,
    physicochemical_feature=None,
):
    # ----- construct cell line - drug response pairs (index mapping)
    cell_ids = list(set([item[0] for item in data_new])); cell_ids.sort()
    drug_ids = list(set([item[1] for item in data_new])); drug_ids.sort()
    cellmap = list(zip(cell_ids, list(range(len(cell_ids)))))
    drugmap = list(zip(drug_ids, list(range(len(cell_ids), len(cell_ids) + len(drug_ids)))))

    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])
    drug_num = np.squeeze([[j[1] for j in drugmap if i[1] == j[0]] for i in data_new])
    label_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((cellline_num, drug_num, label_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    # ----- drug feature input (align order and build edge_index)
    ordered_drug_ids = [item[0] for item in drugmap]
    drug_feature_df = pd.DataFrame(drug_feature).T
    drug_feature_df = drug_feature_df.loc[ordered_drug_ids]
    atom_shape = drug_feature_df.iloc[0].iloc[0].shape[-1]  # Use iloc for positional access
    drug_data = FeatureExtract(drug_feature_df)

    # ----- cell line feature input (align to cell order)
    ordered_cell_ids = [item[0] for item in cellmap]
    
    # Handle None features - only align non-None features
    if genomics_feature is not None:
        genomics_feature = genomics_feature.loc[ordered_cell_ids]
    if epigenomics_feature is not None:
        epigenomics_feature = epigenomics_feature.loc[ordered_cell_ids]
    if transcriptomics_feature is not None:
        transcriptomics_feature = transcriptomics_feature.loc[ordered_cell_ids]
    if proteomics_feature is not None:
        proteomics_feature = proteomics_feature.loc[ordered_cell_ids]
    if metabolomics_feature is not None:
        metabolomics_feature = metabolomics_feature.loc[ordered_cell_ids]
    if pathway_feature is not None:
        pathway_feature = pathway_feature.loc[ordered_cell_ids]

    # Tensors - convert to torch tensors
    genomics = torch.from_numpy(np.array(genomics_feature, dtype='float32')) if genomics_feature is not None else None
    epigenomics = torch.from_numpy(np.array(epigenomics_feature, dtype='float32')) if epigenomics_feature is not None else None
    transcriptomics = torch.from_numpy(np.array(transcriptomics_feature, dtype='float32')) if transcriptomics_feature is not None else None
    proteomics = torch.from_numpy(np.array(proteomics_feature, dtype='float32')) if proteomics_feature is not None else None
    metabolomics = torch.from_numpy(np.array(metabolomics_feature, dtype='float32')) if metabolomics_feature is not None else None
    pathway = torch.from_numpy(np.array(pathway_feature, dtype='float32')) if pathway_feature is not None else None

    # --- compile loaders
    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data), collate_fn=collate, batch_size=nb_drugs, shuffle=False)

    # Create dataset with placeholder tensors for None modalities
    # Using zeros with minimal dimension for disabled modalities
    nb_cells = len(ordered_cell_ids)
    genomics_tensor = genomics if genomics is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    epigenomics_tensor = epigenomics if epigenomics is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    transcriptomics_tensor = transcriptomics if transcriptomics is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    proteomics_tensor = proteomics if proteomics is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    metabolomics_tensor = metabolomics if metabolomics is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    pathway_tensor = pathway if pathway is not None else torch.zeros(nb_cells, 1, dtype=torch.float32)
    
    # Physicochemical features for drugs
    physicochemical_tensor = None
    if physicochemical_feature is not None:
        # Create tensor for drugs in the same order as drug_data
        physicochemical_list = []
        for drug_id in ordered_drug_ids:
            if drug_id in physicochemical_feature:
                # Convert dict values to numpy array
                feature_dict = physicochemical_feature[drug_id]
                feature_values = np.array([feature_dict[key] for key in sorted(feature_dict.keys())], dtype='float32')
                physicochemical_list.append(feature_values)
            else:
                # If drug not in physicochemical data, use zeros
                physicochemical_list.append(np.zeros(64, dtype='float32'))
        physicochemical_tensor = torch.from_numpy(np.stack(physicochemical_list))
        print(f"Created physicochemical tensor with shape: {physicochemical_tensor.shape}")
    else:
        physicochemical_tensor = torch.zeros(nb_drugs, 64, dtype=torch.float32)
    
    # IMPORTANT ordering to match GraphCDR.forward → NodeRepresentation.forward mapping:
    # GraphCDR passes: (mutation_data, gexpr_data, methylation_data, proteomics, metabolomics, pathway)
    # NodeRepresentation expects:  (genomics, epigenomics, transcriptomics, proteomics, metabolomics, pathway)
    cellline_set = Data.DataLoader(
        dataset=Data.TensorDataset(genomics_tensor, epigenomics_tensor, transcriptomics_tensor, proteomics_tensor, metabolomics_tensor, pathway_tensor),
        batch_size=nb_celllines,
        shuffle=False,
    )

    # --- split train/test
    if independent_test:
        edge_mask = cmask(len(allpairs), test_ratio, data_split_seed)
        train = allpairs[edge_mask][:, 0:3]
        test = allpairs[~edge_mask][:, 0:3]
    else:
        CV_edgemask = cmask(len(allpairs), test_ratio, data_split_seed)
        cross_validation = allpairs[CV_edgemask][:, 0:3]
        vali_mask = cmask(len(cross_validation), 0.2, data_split_seed + 1)
        train = cross_validation[vali_mask][:, 0:3]
        test = cross_validation[~vali_mask][:, 0:3]

    # Shift drug indices to 0..D-1 for masks
    train[:, 1] -= nb_celllines
    test[:, 1] -= nb_celllines

    train_mask = coo_matrix((np.ones(train.shape[0], dtype=bool), (train[:, 0], train[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    test_mask = coo_matrix((np.ones(test.shape[0], dtype=bool), (test[:, 0], test[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    train_mask = torch.from_numpy(train_mask).view(-1)
    test_mask = torch.from_numpy(test_mask).view(-1)

    # Positive/negative edges for labels and contrastive
    if independent_test:
        pos_edge = allpairs[allpairs[:, 2] == 1, 0:2]
        neg_edge = allpairs[allpairs[:, 2] == -1, 0:2]
    else:
        pos_edge = cross_validation[cross_validation[:, 2] == 1, 0:2]
        neg_edge = cross_validation[cross_validation[:, 2] == -1, 0:2]
    pos_edge[:, 1] -= nb_celllines
    neg_edge[:, 1] -= nb_celllines

    label_pos = coo_matrix((np.ones(pos_edge.shape[0]), (pos_edge[:, 0], pos_edge[:, 1])), shape=(nb_celllines, nb_drugs)).toarray()
    label_pos = torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)

    # Train edges (symmetrize)
    if independent_test:
        train_edge = allpairs[edge_mask]
    else:
        train_edge = cross_validation[vali_mask]
    train_edge = np.vstack((train_edge, train_edge[:, [1, 0, 2]]))

    return (
        drug_set,
        cellline_set,
        train_edge,
        label_pos,
        train_mask,
        test_mask,
        atom_shape,
        physicochemical_tensor,
    )


