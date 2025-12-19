import pandas as pd

# --- Configuration ---
input_csv = "PubChem_SMILES_data.csv"  # Your input CSV file
output_txt = "pubchem_smiles.txt"      # Output text file

# Read the CSV file
df = pd.read_csv(input_csv)

# Extract PubChem CID and SMILES columns
# Adjust column names based on your CSV header
pubchem_cids = df.iloc[:, 0].astype(str)  # First column: PubChem CID
smiles = df.iloc[:, 1].astype(str)       # Second column: SMILES

# Write to tab-separated text file
with open(output_txt, 'w') as f:
    for cid, smi in zip(pubchem_cids, smiles):
        # Remove quotes from SMILES if present
        smi_clean = smi.strip('"""').strip("'")
        f.write(f"{cid}\t{smi_clean}\n")

print(f"✅ Converted {len(df)} rows from '{input_csv}' to '{output_txt}'")
print(f"Output format: PubChemID[TAB]SMILES (no header)")