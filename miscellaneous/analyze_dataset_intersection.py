"""
Dataset Intersection Analysis Script (v5)

Two Pipelines:
- Pipeline A (Mutation Path): Mutation → Vista Methylation → Gene Expression → SMILES → Physio → HKL
- Pipeline B (Chromatin Path): Chromatin Profiling → Vista Methylation → Gene Expression → SMILES → Physio → HKL

Shows cell lines and drugs at each stage, then final intersection with response data.
Now INCLUDES HKL Filtering.
"""

import os
import pandas as pd

BASE_DIR = "/Volumes/Work/Semester - 6/Honors/GraphCDR/new_data"
CCLE_INITIAL = os.path.join(BASE_DIR, "CCLE", "Initial data")
GDSC_DIR = os.path.join(BASE_DIR, "GDSC")
GDSC_PROCESSED = os.path.join(GDSC_DIR, "Processed data")
GDSC_DRUG_FEAT = os.path.join(GDSC_DIR, "Drug", "drug_graph_feat")

# File paths
RESPONSE_LABELS_FILE = os.path.join(GDSC_PROCESSED, "response_labels_1.csv")
CELL_ANNOTATIONS_FILE = os.path.join(CCLE_INITIAL, "Cell_lines_annotations.csv")
MUTATION_FILE = os.path.join(CCLE_INITIAL, "mutation_data.csv")
CHROMATIN_PROFILING_FILE = os.path.join(CCLE_INITIAL, "chromatin_profiling_data.csv")
VISTA_ENHANCERS_FILE = os.path.join(CCLE_INITIAL, "DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv")
GENE_EXPRESSION_FILE = os.path.join(CCLE_INITIAL, "gene_expression_data.csv")
SMILES_FILE = os.path.join(GDSC_PROCESSED, "pubchem_smiles_1.txt")
PHYSIOCHEMICAL_FILE = os.path.join(GDSC_PROCESSED, "pubchem_physiochemical_properties_1.csv")


def normalize_pubchem_id(pid):
    try:
        return str(int(float(pid)))
    except:
        return str(pid).replace('.0', '')


def get_cell_annotations():
    df = pd.read_csv(CELL_ANNOTATIONS_FILE, sep=',', header=0)
    name_to_ach = {}
    if 'CCLE_ID' in df.columns and 'depMapID' in df.columns:
        for _, row in df.iterrows():
            name_to_ach[str(row['CCLE_ID']).upper()] = str(row['depMapID'])
    return name_to_ach


def get_cell_ids_from_index(csv_path):
    df = pd.read_csv(csv_path, sep=',', header=0, index_col=0)
    return set(df.index.astype(str))


def get_cell_ids_from_columns(csv_path, name_to_ach):
    df = pd.read_csv(csv_path, sep=',', header=0, nrows=1)
    skip_cols = {'cluster_id', 'cpg_sites_hg19', 'avg_coverage', 'gene_id', 'transcript_ids'}
    ids = set()
    for col in df.columns:
        if col.lower() not in skip_cols and not col.lower().startswith('unnamed'):
            name = str(col).upper()
            if name in name_to_ach:
                ids.add(name_to_ach[name])
    return ids


def get_chromatin_ach_ids(csv_path):
    df = pd.read_csv(csv_path, sep=',', header=0)
    for col in df.columns:
        if 'broadid' in col.lower():
            return set(df[col].astype(str))
    return set()


def get_drug_ids_from_smiles(smiles_path):
    ids = set()
    with open(smiles_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                ids.add(normalize_pubchem_id(parts[0]))
    return ids


def get_drug_ids_from_physiochemical(csv_path):
    df = pd.read_csv(csv_path, sep=',', header=0)
    for col in df.columns:
        if 'pubchem' in col.lower() or 'cid' in col.lower():
            return set(df[col].apply(normalize_pubchem_id))
    return set(df.iloc[:, 0].apply(normalize_pubchem_id))


def get_hkl_drug_ids(hkl_dir):
    ids = set()
    for fname in os.listdir(hkl_dir):
        if fname.endswith('.hkl'):
            pid = fname.replace('.hkl', '')
            ids.add(normalize_pubchem_id(pid))
    return ids


def analyze():
    print("=" * 70)
    print("DATASET INTERSECTION ANALYSIS (With HKL Filtering)")
    print("=" * 70)
    
    name_to_ach = get_cell_annotations()
    print(f"Cell line mappings loaded: {len(name_to_ach)}")
    
    # =========================================================================
    # CELL LINE DATA INTERSECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("CELL LINE DATA INTERSECTION")
    print("=" * 70)
    
    # Load each cell line dataset
    mutation_cells = get_cell_ids_from_index(MUTATION_FILE)
    chromatin_cells = get_chromatin_ach_ids(CHROMATIN_PROFILING_FILE)
    vista_cells = get_cell_ids_from_columns(VISTA_ENHANCERS_FILE, name_to_ach)
    gexpr_cells = get_cell_ids_from_columns(GENE_EXPRESSION_FILE, name_to_ach)
    
    # Pipeline A: Mutation → Vista → Gene Expression
    print("\n--- PIPELINE A: Mutation Path ---")
    pA_cells = mutation_cells.copy()
    print(f"  Start (Mutation):                   {len(pA_cells)}")
    pA_cells = pA_cells.intersection(vista_cells)
    print(f"  + Vista Methylation:                {len(pA_cells)}")
    pA_cells = pA_cells.intersection(gexpr_cells)
    print(f"  + Gene Expression:                  {len(pA_cells)}")
    final_cells_A = pA_cells
    
    # Pipeline B: Chromatin → Vista → Gene Expression
    print("\n--- PIPELINE B: Chromatin Path ---")
    pB_cells = chromatin_cells.copy()
    print(f"  Start (Chromatin Profiling):        {len(pB_cells)}")
    pB_cells = pB_cells.intersection(vista_cells)
    print(f"  + Vista Methylation:                {len(pB_cells)}")
    pB_cells = pB_cells.intersection(gexpr_cells)
    print(f"  + Gene Expression:                  {len(pB_cells)}")
    final_cells_B = pB_cells
    
    # =========================================================================
    # DRUG DATA INTERSECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("DRUG DATA INTERSECTION")
    print("=" * 70)
    
    smiles_drugs = get_drug_ids_from_smiles(SMILES_FILE)
    physio_drugs = get_drug_ids_from_physiochemical(PHYSIOCHEMICAL_FILE)
    hkl_drugs = get_hkl_drug_ids(GDSC_DRUG_FEAT)
    
    print(f"\n1. SMILES drugs:                      {len(smiles_drugs)}")
    print(f"2. Physiochemical drugs:              {len(physio_drugs)}")
    print(f"3. HKL drugs (Drug Features):         {len(hkl_drugs)}")
    
    final_drugs = smiles_drugs.intersection(physio_drugs)
    print(f"   After SMILES + Physio:             {len(final_drugs)}")
    
    final_drugs = final_drugs.intersection(hkl_drugs)
    print(f"   After + HKL Features:              {len(final_drugs)}")
    
    # =========================================================================
    # RESPONSE DATA INTERSECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESPONSE DATA INTERSECTION")
    print("=" * 70)
    
    resp = pd.read_csv(RESPONSE_LABELS_FILE, sep=',', header=0)
    cell_col = [c for c in resp.columns if 'depmap' in c.lower()][0]
    drug_col = [c for c in resp.columns if 'pubchem' in c.lower() or 'cid' in c.lower()][0]
    resp['drug_norm'] = resp[drug_col].apply(normalize_pubchem_id)
    
    original_cells = set(resp[cell_col].astype(str))
    original_drugs = set(resp['drug_norm'])
    
    print(f"\nOriginal Response Data:")
    print(f"  Cell Lines: {len(original_cells)}")
    print(f"  Drugs: {len(original_drugs)}")
    print(f"  Response Pairs: {len(resp)}")
    
    # Pipeline A final
    cells_A_final = final_cells_A.intersection(original_cells)
    drugs_A_final = final_drugs.intersection(original_drugs)
    resp_A = resp[
        (resp[cell_col].astype(str).isin(cells_A_final)) &
        (resp['drug_norm'].isin(drugs_A_final))
    ]
    
    # Pipeline B final
    cells_B_final = final_cells_B.intersection(original_cells)
    drugs_B_final = final_drugs.intersection(original_drugs)
    resp_B = resp[
        (resp[cell_col].astype(str).isin(cells_B_final)) &
        (resp['drug_norm'].isin(drugs_B_final))
    ]
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print("\n--- PIPELINE A: Mutation → Vista → GeneExpr → SMILES → Physio → HKL ---")
    print(f"  Final Cell Lines:    {len(cells_A_final)}")
    print(f"  Final Drugs:         {len(drugs_A_final)}")
    print(f"  Final Response Pairs (Samples): {len(resp_A)}")
    
    print("\n--- PIPELINE B: Chromatin → Vista → GeneExpr → SMILES → Physio → HKL ---")
    print(f"  Final Cell Lines:    {len(cells_B_final)}")
    print(f"  Final Drugs:         {len(drugs_B_final)}")
    print(f"  Final Response Pairs (Samples): {len(resp_B)}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Pipeline A (Mutation)':<25} {'Pipeline B (Chromatin)':<25}")
    print("-" * 80)
    print(f"{'Cell Lines':<30} {len(cells_A_final):<25} {len(cells_B_final):<25}")
    print(f"{'Drugs':<30} {len(drugs_A_final):<25} {len(drugs_B_final):<25}")
    print(f"{'Response Pairs (Samples)':<30} {len(resp_A):<25} {len(resp_B):<25}")
    
    # Save results
    results = {
        'Pipeline': ['A (Mutation)', 'B (Chromatin)'],
        'Final Cell Lines': [len(cells_A_final), len(cells_B_final)],
        'Final Drugs': [len(drugs_A_final), len(drugs_B_final)],
        'Final Response Pairs': [len(resp_A), len(resp_B)]
    }
    df_results = pd.DataFrame(results)
    output_path = os.path.join(BASE_DIR, "intersection_analysis_v5.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    analyze()
