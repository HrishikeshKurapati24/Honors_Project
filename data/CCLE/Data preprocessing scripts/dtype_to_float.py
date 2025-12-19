import pandas as pd
import os

# --- Configuration ---
input_folder = "Final imputed csvs"   # Folder containing your CSV files
output_folder = "Final imputed csvs"    # Folder to save processed CSVs

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all CSV files in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        
        # Read CSV with low_memory=False to suppress dtype warnings
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert all columns except first to float
        feature_cols = df.columns[1:]  # All except first
        df[feature_cols] = df[feature_cols].astype(float)
        
        # Save the processed CSV
        output_path = os.path.join(output_folder, file_name)
        df.to_csv(output_path, index=False)
        
        print(f"Processed and saved: {file_name}")