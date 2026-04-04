import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os

# =============== CONFIG ===============
INPUT_FILE = "/tmp/all_smiles_to_calculate.csv"
OUTPUT_FILE = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/data/GDSC/Processed data/pubchem_physiochemical_properties_338.csv"
EXISTING_FILE = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/data/GDSC/Processed data/pubchem_physiochemical_properties_1.csv"

# Load existing columns (specifically the features)
df_existing = pd.read_csv(EXISTING_FILE)
target_features = [col for col in df_existing.columns if col != "PUBCHEM_CID"]
print(f"Targeting {len(target_features)} features.")

# Load input SMILES
df = pd.read_csv(INPUT_FILE)

# Initialize Mordred calculator (2D only)
calc = Calculator(descriptors, ignore_3D=True)

records = []
print(f"🔹 Computing descriptors for {len(df)} molecules...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    cid = row["PUBCHEM_CID"]
    smiles = row["SMILES"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES for CID {cid}")
        continue

    try:
        desc_values = calc(mol)
        record = desc_values.asdict()
        filtered_record = {"PUBCHEM_CID": cid}
        for feat in target_features:
            # Handle potential non-numeric values
            val = record.get(feat, 0.0)
            try:
                filtered_record[feat] = float(val)
            except:
                filtered_record[feat] = 0.0
        records.append(filtered_record)
    except Exception as e:
        print(f"Error calculating CID {cid}: {e}")
        continue

# Convert to DataFrame
desc_df = pd.DataFrame(records)

# Normalize features
scaler = StandardScaler()
# Fill NaNs with 0 just in case
desc_df = desc_df.fillna(0.0)

scaled = scaler.fit_transform(desc_df[target_features])
scaled_df = pd.DataFrame(scaled, columns=target_features)

# Insert CID column first
scaled_df.insert(0, "PUBCHEM_CID", desc_df["PUBCHEM_CID"].values)

# Save normalized features
scaled_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Done! Saved {len(scaled_df)} molecules × {len(target_features)} normalized features to '{OUTPUT_FILE}'")
