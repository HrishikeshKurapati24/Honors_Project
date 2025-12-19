import os
import pandas as pd
from sklearn.impute import KNNImputer

# --- Configuration ---
input_folder = "Final csvs"  # Folder containing your CSVs (use '.' if script is in the same folder)
output_folder = "Final imputed csvs"  # Folder to save imputed CSVs
n_neighbors = 5  # K value for KNN Imputer

# --- Create output folder if not exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Function to perform KNN imputation on one file ---
def impute_file(file_path, output_path):
    print(f"🔹 Processing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)

    # Handle unnamed first column (DepMapID)
    df.columns = ['DepMapID'] + list(df.columns[1:])
    depmap_ids = df['DepMapID']
    features = df.drop(columns=['DepMapID'])

    # Convert to numeric (ignore non-numeric if any)
    features = features.apply(pd.to_numeric, errors='coerce')

    # Perform KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_features = imputer.fit_transform(features)

    # Recreate DataFrame
    imputed_df = pd.DataFrame(imputed_features, columns=features.columns)
    imputed_df.insert(0, 'DepMapID', depmap_ids)

    # Save output
    imputed_df.to_csv(output_path, index=False)
    print(f"✅ Saved imputed file to: {output_path}\n")


# --- Main loop over all CSVs in folder ---
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") and "Cell_lines_annotations" not in filename:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        impute_file(input_path, output_path)

print("🎉 All imputations completed successfully!")