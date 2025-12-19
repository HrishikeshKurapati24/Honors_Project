import os
import csv
import numpy as np
import pandas as pd
import hickle as hkl

def dataload(
    drug_feature_dir,
    response_file,
    genomics_csv,
    epigenomics_csv,
    transcriptomics_csv,
    proteomics_csv,
    metabolomics_csv,
    pathway_csv,
):
    """
    Load precomputed drug graph features (HKL) and multi-omics CSVs for the new model.

    Expected response_file columns (placeholders; adapt to your data):
      - cell_line_id: str (must match omics row indices)
      - pubchem_id: str or int (must match HKL filenames without extension)
      - label: int in {1, -1}

    Returns:
      drug_feature           : dict[pubchem_id] -> [atom_features, adj_list, degree_list]
      genomics_feature       : pd.DataFrame (cells x features)
      epigenomics_feature    : pd.DataFrame (cells x length or channels*length flattened)
      transcriptomics_feature: pd.DataFrame (cells x features)
      proteomics_feature     : pd.DataFrame (cells x features)
      metabolomics_feature   : pd.DataFrame (cells x features)
      pathway_feature        : pd.DataFrame (cells x pathway_dims)
      data_new               : list of (cell_line_id, pubchem_id, label)
      nb_celllines           : int
      nb_drugs               : int
    """

    # ----- Load precomputed drug features (HKL files named <pubchem>.hkl)
    drug_pubchem_id_set = []
    drug_feature = {}
    print(f"Loading drug features from: {drug_feature_dir}")
    hkl_files = [f for f in os.listdir(drug_feature_dir) if f.endswith('.hkl')]
    print(f"Found {len(hkl_files)} HKL files")
    for each in hkl_files:
        pubchem_id = each.split('.')[0]
        drug_pubchem_id_set.append(pubchem_id)
        feat_mat, adj_list, degree_list = hkl.load(f"{drug_feature_dir}/{each}")
        drug_feature[pubchem_id] = [feat_mat, adj_list, degree_list]
    print(f"Loaded {len(drug_pubchem_id_set)} drug features")
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # ----- Load multi-omics CSVs (placeholders)
    genomics_feature = pd.read_csv(genomics_csv, sep=',', header=0, index_col=[0])
    epigenomics_feature = pd.read_csv(epigenomics_csv, sep=',', header=0, index_col=[0])
    transcriptomics_feature = pd.read_csv(transcriptomics_csv, sep=',', header=0, index_col=[0])
    proteomics_feature = pd.read_csv(proteomics_csv, sep=',', header=0, index_col=[0])
    metabolomics_feature = pd.read_csv(metabolomics_csv, sep=',', header=0, index_col=[0])
    # PATHWAY DATA LOADING COMMENTED OUT TO INCREASE DATASET SIZE
    # pathway_feature = pd.read_csv(pathway_csv, sep=',', header=0, index_col=[0])
    pathway_feature = None
    print("NOTE: Pathway data loading is commented out to increase dataset size by excluding pathway-based filtering.")

    # ----- Align cell line indices across all omics by intersection
    index_sets = [
        set(genomics_feature.index),
        set(epigenomics_feature.index),
        set(transcriptomics_feature.index),
        set(proteomics_feature.index),
        set(metabolomics_feature.index),
        # PATHWAY EXCLUDED FROM INTERSECTION TO INCREASE DATASET SIZE
        # set(pathway_feature.index),
    ]
    common_cells = sorted(list(set.intersection(*index_sets)))
    print(f"Found {len(common_cells)} common cell lines across 5 omics (pathway excluded from intersection).")
    if len(common_cells) == 0:
        raise ValueError('No common cell line IDs across all omics CSVs.')

    genomics_feature = genomics_feature.loc[common_cells]
    epigenomics_feature = epigenomics_feature.loc[common_cells]
    transcriptomics_feature = transcriptomics_feature.loc[common_cells]
    proteomics_feature = proteomics_feature.loc[common_cells]
    metabolomics_feature = metabolomics_feature.loc[common_cells]
    # PATHWAY FILTERING COMMENTED OUT
    # pathway_feature = pathway_feature.loc[common_cells]

    # ----- Load response data (cell_line_id, pubchem_id, label)
    resp = pd.read_csv(response_file, sep=',', header=0)
    print(f"Loaded response file with {len(resp)} rows")
    if not {'cell_line_id', 'pubchem_id', 'label'}.issubset(set(resp.columns)):
        raise ValueError('response_file must contain columns: cell_line_id, pubchem_id, label')

    # Filter to pairs present in both drug features and omics cell lines
    print(f"Filtering by drug features ({len(drug_pubchem_id_set)} drugs available)")
    resp_before_drug = len(resp)
    resp = resp[resp['pubchem_id'].astype(str).isin(drug_pubchem_id_set)]
    print(f"After drug filter: {len(resp)} rows (removed {resp_before_drug - len(resp)})")
    
    print(f"Filtering by cell lines ({len(common_cells)} cells available)")
    resp_before_cell = len(resp)
    resp = resp[resp['cell_line_id'].astype(str).isin(common_cells)]
    print(f"After cell line filter: {len(resp)} rows (removed {resp_before_cell - len(resp)})")

    # Eliminate ambiguity (keep one label per (cell, drug))
    data_idx = [(str(r.cell_line_id), str(r.pubchem_id), int(r.label)) for _, r in resp.iterrows()]
    data_sort = sorted(data_idx, key=(lambda x: [x[0], x[1], x[2]]), reverse=True)
    data_tmp = []
    data_new = []
    data_idx1 = [[i[0], i[1]] for i in data_sort]
    for i, k in zip(data_idx1, data_sort):
        if i not in data_tmp:
            data_tmp.append(i)
            data_new.append(k)

    nb_celllines = len(set([item[0] for item in data_new]))
    nb_drugs = len(set([item[1] for item in data_new]))
    print('All %d pairs across %d cell lines and %d drugs.' % (len(data_new), nb_celllines, nb_drugs))

    return (
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
    )