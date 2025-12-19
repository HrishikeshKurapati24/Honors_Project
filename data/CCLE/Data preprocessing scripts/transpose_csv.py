import pandas as pd

# --- Configuration ---
input_file = "row_filtered_omics/DNA_methylation_promoter_CpG_clusters.csv"          # path to your input CSV
output_file = "transposed/DNA_methylation_promoter_CpG_clusters.csv"    # path for the output file

# --- Load the data ---
# The first column is treated as row labels (index)
df = pd.read_csv(input_file, index_col=0)

# --- Transpose ---
df_transposed = df.T

# --- Save the transposed data ---
df_transposed.to_csv(output_file)

print(f"✅ Transposed '{input_file}' → '{output_file}'")
print(f"Original shape: {df.shape}, Transposed shape: {df_transposed.shape}")