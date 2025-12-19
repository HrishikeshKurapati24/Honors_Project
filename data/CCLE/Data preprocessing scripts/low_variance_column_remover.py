import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np

# --- Configuration ---
input_file = "transposed/gene_expression_data.csv"           
output_file = "transposed/gene_expression_data.csv"  
variance_threshold = 0.03  # Keep features with >3% variance

# --- Load the data ---
df = pd.read_csv(input_file)

# Keep first column separately (e.g., Gene names)
first_col = df.iloc[:, 0]
X = df.iloc[:, 1:]

# --- Convert values to numeric, treat non-numeric as NaN ---
X_numeric = X.apply(lambda col: pd.to_numeric(col.astype(str).str.strip(), errors='coerce'))

# --- Fill NaN with 0 only for variance calculation ---
X_for_variance = X_numeric.fillna(0)

# --- Apply VarianceThreshold ---
selector = VarianceThreshold(threshold=variance_threshold)
X_reduced = selector.fit_transform(X_for_variance)

# --- Get columns kept ---
cols_kept = X.columns[selector.get_support()]

# --- Reconstruct dataframe using original data (keep NAs) ---
df_reduced = df.loc[:, [df.columns[0]] + list(cols_kept)]

# --- Save cleaned CSV ---
df_reduced.to_csv(output_file, index=False)

print(f"Original columns: {len(df.columns)}")
print(f"Columns after low-variance removal: {len(df_reduced.columns)}")
print(f"Saved cleaned data to '{output_file}'")