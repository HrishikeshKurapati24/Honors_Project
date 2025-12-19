import pandas as pd
import numpy as np

# --- Load files ---
drug_response_file = "drug_response_data.csv"
cell_line_mapper_file = "Cell_lines_annotations.csv"
drug_mapper_file = "drug_to_cid.csv"

drug_df = pd.read_csv(drug_response_file)
cell_mapper_df = pd.read_csv(cell_line_mapper_file)
drug_mapper_df = pd.read_csv(drug_mapper_file)

# --- Map cell line names to depMap IDs ---
drug_df = drug_df.merge(cell_mapper_df, left_on='CELL_LINE_NAME', right_on='Name', how='left')

# --- Map drug names to PubChem IDs ---
drug_df = drug_df.merge(drug_mapper_df, on='DRUG_NAME', how='left')

# --- Compute label ---
drug_df['label'] = np.where(drug_df['LN_IC50'] < np.log(drug_df['MAX_CONC']), 1, -1)

# --- Select desired columns ---
output_df = drug_df[['depMapID', 'PUBCHEM_CID', 'label']]

# --- Save to CSV ---
output_df.to_csv("response_labels.csv", index=False)

print("Output saved to response_labels.csv")