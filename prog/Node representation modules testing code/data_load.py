"""
Optimized data loader for baseline (GraphCDR) and modified experiments.
Assumes data in final_dataset is already pre-aligned and clean.
"""

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
    use_genomics=True,
    use_epigenomics=True,
    use_transcriptomics=True,
    use_proteomics=True,
    use_metabolomics=True,
    use_pathway=True
):
    """
    OPTIMIZED LOADER:
    - Only loads requested modalities.
    - Skips redundant intersection and majority voting (assumes final_dataset is clean).
    """

    # ----- 1. Load drug features (HKL)
    drug_feature = {}
    print(f"Loading drug features from: {drug_feature_dir}")
    hkl_files = [f for f in os.listdir(drug_feature_dir) if f.endswith('.hkl')]
    for each in hkl_files:
        pubchem_id = each.split('.')[0]
        pubchem_id_clean = str(int(float(pubchem_id)))
        feat_mat, adj_list, degree_list = hkl.load(os.path.join(drug_feature_dir, each))
        drug_feature[pubchem_id_clean] = [feat_mat, adj_list, degree_list]
    print(f"Loaded {len(drug_feature)} drug features.")

    # ----- 2. Load requested Cell-centric CSVs
    def load_csv(path, name, enabled):
        if enabled and path and os.path.exists(path):
            print(f"Loading {name} -> {os.path.basename(path)}")
            return pd.read_csv(path, sep=',', header=0, index_col=[0])
        return pd.DataFrame()

    genomics_feature = load_csv(genomics_csv, "Genomics", use_genomics)
    epigenomics_feature = load_csv(epigenomics_csv, "Epigenomics", use_epigenomics)
    transcriptomics_feature = load_csv(transcriptomics_csv, "Transcriptomics", use_transcriptomics)
    proteomics_feature = load_csv(proteomics_csv, "Proteomics", use_proteomics)
    metabolomics_feature = load_csv(metabolomics_csv, "Metabolomics", use_metabolomics)
    pathway_feature = load_csv(pathway_csv, "Pathway", use_pathway)

    # ----- 3. Load Response Data
    print(f"Loading response pairs: {os.path.basename(response_file)}")
    resp = pd.read_csv(response_file, sep=',', header=0)
    
    # final_dataset response_pairs.csv has: cell_line_id, pubchem_id, label
    data_new = []
    for _, r in resp.iterrows():
        data_new.append((str(r.cell_line_id), str(int(float(r.pubchem_id))), int(r.label)))

    nb_celllines = len(set([item[0] for item in data_new]))
    nb_drugs = len(set([item[1] for item in data_new]))
    print(f'DATASET SIZE: {len(data_new)} pairs | {nb_celllines} cells | {nb_drugs} drugs.\n')

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
        nb_drugs
    )