import pandas as pd

# --- Configuration ---
cell_line_file = "transposed/cell_pathway_scores_from_gene_expression.csv"   # CSV with cell-line data
mapping_file = "transposed/Cell_lines_annotations.csv"               # CSV with columns: Name, depMapID
output_file = "Final csvs/cell_pathway_scores_from_gene_expression.csv"      # Output file
name_col = "Name"                                                    # Column in both files that contains cell-line name

# --- Load data ---
cell_df = pd.read_csv(cell_line_file)
mapping_df = pd.read_csv(mapping_file)

# --- Normalize names (remove '-' and ' ', convert to lowercase) ---
mapping_df[name_col] = (
    mapping_df[name_col]
    .astype(str)
    .str.replace("-", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.lower()
)

cell_df.columns.values[0] = name_col
cell_df[name_col] = (
    cell_df[name_col]
    .astype(str)
    .str.replace("-", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.lower()
)

# --- Create mapping dictionary ---
mapping_dict = dict(zip(mapping_df[name_col], mapping_df["depMapID"]))

# --- Filter and replace ---
filtered_df = cell_df[cell_df[name_col].isin(mapping_dict.keys())].copy()
filtered_df[name_col] = filtered_df[name_col].map(mapping_dict)

# --- Save the result ---
filtered_df.to_csv(output_file, index=False)

print(f"✅ Saved mapped cell-line data to '{output_file}' ({len(filtered_df)} records).")
print(f"⚠️ Skipped {len(cell_df) - len(filtered_df)} records without mapping.")