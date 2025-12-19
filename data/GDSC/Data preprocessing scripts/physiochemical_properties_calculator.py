import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =============== CONFIG ===============
INPUT_FILE = "../Processed data/pubchem_smiles.txt"   # tab-separated, no headers
OUTPUT_FILE = "../Processed data/pubchem_physiochemical_properties.csv"
NUM_FEATURES = 64  # number of top descriptors to retain
# =====================================

# Load input file (tab-separated, no headers)
df = pd.read_csv(INPUT_FILE, sep="\t", header=None, names=["PUBCHEM_CID", "SMILES"])

# Initialize Mordred calculator (2D only)
calc = Calculator(descriptors, ignore_3D=True)

records = []
print(f"🔹 Computing descriptors for {len(df)} molecules...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    cid = row["PUBCHEM_CID"]
    smiles = row["SMILES"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue

    try:
        desc_values = calc(mol)
        record = desc_values.asdict()
        record["PUBCHEM_CID"] = cid
        records.append(record)
    except Exception:
        continue

# Convert to DataFrame
desc_df = pd.DataFrame(records)

# Remove non-numeric and constant columns
desc_df = desc_df.select_dtypes(include=["float64", "int64"])
desc_df = desc_df.loc[:, desc_df.nunique() > 1]

# Select top 64 most variant descriptors
variances = desc_df.var().sort_values(ascending=False)
top_features = variances.index[:NUM_FEATURES]

desc_df = desc_df[top_features]
desc_df["PUBCHEM_CID"] = [r["PUBCHEM_CID"] for r in records]

# Normalize features
scaler = StandardScaler()
scaled = scaler.fit_transform(desc_df.drop(columns=["PUBCHEM_CID"]))
scaled_df = pd.DataFrame(scaled, columns=desc_df.columns.drop("PUBCHEM_CID"))

# Insert CID column first
scaled_df.insert(0, "PUBCHEM_CID", desc_df["PUBCHEM_CID"].values)

# Save normalized features
scaled_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Done! Saved {len(scaled_df)} molecules × {NUM_FEATURES} normalized features to '{OUTPUT_FILE}'")