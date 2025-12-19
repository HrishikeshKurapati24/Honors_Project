import os
import pandas as pd

# --- Configuration ---
omics_files = [
    "metabolomics_data.csv",
    "miRNA_expression_data.csv",
    "mutation_data.csv",
    "reverse_phase_protein_array_data.csv",
    "DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv",
    "DNA_methylation_promoter_CpG_clusters.csv",
    "gene_expression_data.csv",
    "cell_pathway_scores_from_gene_expression.csv"
]

input_dir = "./column_filtered_omics"       # directory containing the CSV files
output_dir = "./row_filtered_omics"  # directory to save cleaned files
threshold = 0.1        # 10% missing value threshold (adjustable)

# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Row filtering function ---
def clean_file(file_path, output_path, threshold):
    df = pd.read_csv(file_path)
    min_non_missing = int((1 - threshold) * len(df.columns))
    cleaned_df = df.dropna(thresh=min_non_missing)
    cleaned_df.to_csv(output_path, index=False)
    print(f"[{os.path.basename(file_path)}]  Rows before: {len(df)} | after: {len(cleaned_df)}")
    return cleaned_df

# --- Process all omics files ---
for filename in omics_files:
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(input_path):
        clean_file(input_path, output_path, threshold)
    else:
        print(f"⚠️  File not found: {filename}")

print("\n✅ All available omics files processed.")
print(f"Row-filtered files saved in: {os.path.abspath(output_dir)}")