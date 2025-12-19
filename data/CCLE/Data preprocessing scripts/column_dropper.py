import pandas as pd

# --- Configuration ---
input_file = "transposed/mutation_data.csv"            # Path to your input CSV
output_file = "Final csvs/mutation_data.csv"         # Path for the output CSV
columns_to_drop = ["CellLineName"]  # Columns you want to remove

# --- Load the data ---
df = pd.read_csv(input_file)

# --- Drop the specified columns ---
df_dropped = df.drop(columns=columns_to_drop, errors='ignore')

# --- Save the cleaned data ---
df_dropped.to_csv(output_file, index=False)

print(f"✅ Dropped columns: {columns_to_drop}")
print(f"Original columns: {len(df.columns)} | After drop: {len(df_dropped.columns)}")
print(f"Saved cleaned data to '{output_file}'")