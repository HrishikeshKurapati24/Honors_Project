
import sys
import os
import numpy as np
import pandas as pd
import torch

# Absolute paths to avoid ambiguity
BASE_PATH = "/Volumes/Work/Semester - 6/Honors/CDRP models testing"
DRUG_CELL_PATH = os.path.join(BASE_PATH, "new_prog/Drug-cell pair processing module testing code")
NODE_REP_PATH = os.path.join(BASE_PATH, "new_prog/Node representation modules testing code")
NEW_DATA_ROOT = os.path.join(BASE_PATH, "new_data")

COMMON_CONFIG = {
    'drug_feature_dir': os.path.join(NEW_DATA_ROOT, 'GDSC/Drug/drug_graph_feat'),
    'response_file': os.path.join(NEW_DATA_ROOT, 'GDSC/Processed data/response_labels_1.csv'),
    'genomics_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/mutation_data.csv'),
    'epigenomics_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv'),
    'transcriptomics_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/gene_expression_data.csv'),
    'proteomics_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/reverse_phase_protein_array_data.csv'),
    'metabolomics_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/metabolomics_data.csv'),
    'pathway_csv': os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/pathway_data.csv'),
    'use_genomics': True,
    'use_epigenomics': True,
    'use_transcriptomics': True,
    'use_proteomics': False,
    'use_metabolomics': False,
    'use_pathway': False
}

def verify_drug_cell_module():
    sys.path.insert(0, DRUG_CELL_PATH)
    from data_load import dataload
    from data_process import process
    
    print("\n--- [1] Drug-Cell Pair Module Loading ---")
    res = dataload(
        **COMMON_CONFIG,
        similarity_csv=os.path.join(NEW_DATA_ROOT, 'CCLE/Processed data/cell_line_similarity_feature.csv'),
        physiochemical_csv=os.path.join(NEW_DATA_ROOT, 'GDSC/Processed data/pubchem_physiochemical_properties_1.csv')
    )
    # drug_feature, genomics_feature, epigenomics_feature, transcriptomics_feature, proteomics_feature, metabolomics_feature, pathway_feature, similarity_feature, data_new, nb_celllines, nb_drugs, cellid, drugid
    nb_cells = res[9]
    nb_drugs = res[10]
    nb_pairs = len(res[8])
    
    print(f"Cells: {nb_cells}, Drugs: {nb_drugs}, Total Pairs: {nb_pairs}")
    
    sys.path.remove(DRUG_CELL_PATH)
    del sys.modules['data_load']
    del sys.modules['data_process']
    return nb_cells, nb_drugs, nb_pairs

def verify_node_rep_module():
    sys.path.insert(0, NODE_REP_PATH)
    from data_load import dataload
    # node rep lacks similarity_csv and physiochemical_csv args
    
    print("\n--- [2] Node Representation Module Loading ---")
    res = dataload(**COMMON_CONFIG)
    # drug_feature, genomics_feature, epigenomics_feature, transcriptomics_feature, proteomics_feature, metabolomics_feature, pathway_feature, data_new, nb_celllines, nb_drugs
    nb_cells = res[8]
    nb_drugs = res[9]
    nb_pairs = len(res[7])
    
    print(f"Cells: {nb_cells}, Drugs: {nb_drugs}, Total Pairs: {nb_pairs}")
    
    sys.path.remove(NODE_REP_PATH)
    del sys.modules['data_load']
    return nb_cells, nb_drugs, nb_pairs

if __name__ == "__main__":
    c1, d1, p1 = verify_drug_cell_module()
    c2, d2, p2 = verify_node_rep_module()
    
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Drug-Cell':<12} | {'Node-Rep':<12}")
    print("-" * 50)
    print(f"{'Cell Lines':<20} | {c1:<12} | {c2:<12}")
    print(f"{'Drugs':<20} | {d1:<12} | {d2:<12}")
    print(f"{'Pairs':<20} | {p1:<12} | {p2:<12}")
    print("="*50)
    
    if (c1, d1, p1) == (c2, d2, p2):
        print("\nRESULT: SUCCESS! Counts match exactly across both modules.")
    else:
        print("\nRESULT: FAILURE! Counts differ.")
