import os
import csv
import shutil
import pandas as pd
from collections import defaultdict, Counter

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'data')

DRUG_FEAT_DIR = os.path.join(DATA_ROOT, 'GDSC/Drug/drug_graph_feat')
RESPONSE_FILE = os.path.join(DATA_ROOT, 'GDSC/Processed data/response_labels_1.csv')

# Cell-line omics
MUTATION_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/mutation_data.csv')
CHROMATIN_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/chromatin_profiling.csv')
EPIGENOMICS_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv')
TRANSCRIPTOMICS_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/gene_expression_data.csv')
PROTEOMICS_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/reverse_phase_protein_array_data.csv')
METABOLOMICS_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/metabolomics_data.csv')
PATHWAY_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/cell_pathway_scores_from_gene_expression.csv')
SIMILARITY_CSV = os.path.join(DATA_ROOT, 'CCLE/Processed data/cellline_pathway_activity_PROGENy_Pathway.csv')

PHYSIOCHEMICAL_CSV = os.path.join(DATA_ROOT, 'GDSC/Processed data/pubchem_physiochemical_properties_1.csv')

# Output
OUT_DIR = os.path.join(SCRIPT_DIR, 'final_dataset')
DRUG_FEAT_OUT = os.path.join(OUT_DIR, 'drug_graph_feat')


# =============================================================================
# Helper Functions
# =============================================================================

def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} CSV not found at {path}")

    print(f"  Loading {name} -> {os.path.basename(path)}")
    df = pd.read_csv(path)

    # Ensure index is first column
    first_col = df.columns[0]
    df = df.set_index(first_col)

    return df


def normalize_pubchem(x):
    """Normalize PubChem IDs to string"""
    return str(int(float(x)))


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DRUG_FEAT_OUT, exist_ok=True)

    print(f"\nOutput directory: {OUT_DIR}\n")

    # -------------------------------------------------------------------------
    # 1. Scan drug HKL files
    # -------------------------------------------------------------------------

    print("=== 1. Scanning drug graph features ===")

    drug_pubchem_id_set = set()

    for fname in os.listdir(DRUG_FEAT_DIR):

        if fname.endswith(".hkl"):

            pubchem_id = normalize_pubchem(os.path.splitext(fname)[0])
            drug_pubchem_id_set.add(pubchem_id)

    print(f"  Found {len(drug_pubchem_id_set)} drug HKL features.")

    # -------------------------------------------------------------------------
    # 2. Load omics datasets
    # -------------------------------------------------------------------------

    print("\n=== 2. Loading cell-line CSVs ===")

    mutation_df = load_csv(MUTATION_CSV, "Genomics/Mutation")
    chromatin_df = load_csv(CHROMATIN_CSV, "Genomics/Chromatin")
    epigenomics_df = load_csv(EPIGENOMICS_CSV, "Epigenomics")
    transcriptomics_df = load_csv(TRANSCRIPTOMICS_CSV, "Transcriptomics")
    proteomics_df = load_csv(PROTEOMICS_CSV, "Proteomics")
    metabolomics_df = load_csv(METABOLOMICS_CSV, "Metabolomics")
    pathway_df = load_csv(PATHWAY_CSV, "Pathway")
    similarity_df = load_csv(SIMILARITY_CSV, "Similarity")

    # -------------------------------------------------------------------------
    # 3. Compute strict omics intersection
    # -------------------------------------------------------------------------

    print("\n=== 3. Computing strict cell-line intersection ===")

    index_sets = [
        set(mutation_df.index),
        set(chromatin_df.index),
        set(epigenomics_df.index),
        set(transcriptomics_df.index),
        set(proteomics_df.index),
        set(metabolomics_df.index),
        set(pathway_df.index),
        set(similarity_df.index),
    ]

    common_cells = sorted(set.intersection(*index_sets))

    print(f"  Common cell lines across omics: {len(common_cells)}")

    if len(common_cells) == 0:
        raise ValueError("No common cell lines across omics.")

    # -------------------------------------------------------------------------
    # 4. Load physicochemical features
    # -------------------------------------------------------------------------

    print("\n=== 4. Loading physicochemical features ===")

    phys_df = pd.read_csv(PHYSIOCHEMICAL_CSV)

    phys_df['PUBCHEM_CID'] = phys_df['PUBCHEM_CID'].apply(normalize_pubchem)
    phys_df = phys_df.set_index('PUBCHEM_CID')

    phys_keys = set(phys_df.index)

    common_drugs = sorted(drug_pubchem_id_set.intersection(phys_keys))

    print(f"  Drugs with graph + physicochemical features: {len(common_drugs)}")

    # -------------------------------------------------------------------------
    # 5. Process response dataset
    # -------------------------------------------------------------------------

    print("\n=== 5. Processing response pairs ===")

    resp = pd.read_csv(RESPONSE_FILE)

    print(f"  Raw response rows: {len(resp)}")

    resp['pubchem_id'] = resp['pubchem_id'].apply(normalize_pubchem)
    resp['cell_line_id'] = resp['cell_line_id'].astype(str)

    resp = resp[
        resp['pubchem_id'].isin(common_drugs) &
        resp['cell_line_id'].isin(common_cells)
    ]

    print(f"  After intersection filter: {len(resp)} rows")

    pair_labels = defaultdict(list)

    for _, r in resp.iterrows():

        pair_labels[(r.cell_line_id, r.pubchem_id)].append(int(r.label))

    data_new = []

    for (cell, drug), labels in pair_labels.items():

        counts = Counter(labels)

        majority_label = 1 if counts[1] >= counts[-1] else -1

        data_new.append((cell, drug, majority_label))

    final_cells = sorted(set(x[0] for x in data_new))
    final_drugs = sorted(set(x[1] for x in data_new))

    print(
        f"  Final dataset: {len(data_new)} pairs | "
        f"{len(final_cells)} cells | {len(final_drugs)} drugs"
    )

    # -------------------------------------------------------------------------
    # 6. Save response pairs
    # -------------------------------------------------------------------------

    pairs_out = os.path.join(OUT_DIR, 'response_pairs.csv')

    with open(pairs_out, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow(['cell_id', 'drug_id', 'label'])
        writer.writerows(data_new)

    print(f"  Saved response_pairs.csv ({len(data_new)} rows)")

    # -------------------------------------------------------------------------
    # 7. Save filtered omics
    # -------------------------------------------------------------------------

    print("\n=== 6. Saving filtered omics datasets ===")

    def save(df, filename):

        filtered = df.loc[final_cells]

        path = os.path.join(OUT_DIR, filename)

        filtered.to_csv(path)

        print(f"  Saved {filename} ({len(filtered)} rows)")

    save(mutation_df, 'genomics_mutation.csv')
    save(chromatin_df, 'genomics_chromatin.csv')
    save(epigenomics_df, 'epigenomics.csv')
    save(transcriptomics_df, 'transcriptomics.csv')
    save(proteomics_df, 'proteomics.csv')
    save(metabolomics_df, 'metabolomics.csv')
    save(pathway_df, 'pathway.csv')
    save(similarity_df, 'similarity.csv')

    # -------------------------------------------------------------------------
    # 8. Save physicochemical features
    # -------------------------------------------------------------------------

    phys_filtered = phys_df.loc[final_drugs]

    phys_filtered.index.name = "PUBCHEM_CID"

    phys_filtered.to_csv(os.path.join(OUT_DIR, 'physicochemical.csv'))

    print(f"  Saved physicochemical.csv ({len(final_drugs)} rows)")

    # -------------------------------------------------------------------------
    # 9. Save drug list
    # -------------------------------------------------------------------------

    with open(os.path.join(OUT_DIR, 'drugs.txt'), 'w') as f:

        f.write('\n'.join(final_drugs))

    print(f"  Saved drugs.txt ({len(final_drugs)} drugs)")

    # -------------------------------------------------------------------------
    # 10. Copy filtered HKL files
    # -------------------------------------------------------------------------

    print("\n=== 7. Copying filtered drug HKL files ===")

    copied = 0

    for fname in os.listdir(DRUG_FEAT_DIR):

        if not fname.endswith(".hkl"):
            continue

        pubchem_id = normalize_pubchem(os.path.splitext(fname)[0])

        if pubchem_id in final_drugs:

            shutil.copy2(
                os.path.join(DRUG_FEAT_DIR, fname),
                os.path.join(DRUG_FEAT_OUT, fname)
            )

            copied += 1

    print(f"  Copied {copied} HKL files")

    print("\n=== Export Complete ===")
    print(f"All files saved to: {OUT_DIR}")


# =============================================================================

if __name__ == "__main__":
    main()