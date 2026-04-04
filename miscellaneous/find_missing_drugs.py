import os
import pandas as pd

base_path = "/Volumes/Work/Semester - 6/Honors/CDRP models testing"
drug_feat_dir = os.path.join(base_path, "benchmark models/GraphCDR/data/Drug/drug_graph_feat")
phys_csv = os.path.join(base_path, "data/GDSC/Processed data/pubchem_physiochemical_properties_1.csv")
smiles_txt = os.path.join(base_path, "benchmark models/GraphCDR/data/Drug/222drugs_pubchem_smiles.txt")

# 1. CIDs from drug_graph_feat (these are the ones we want to have)
drug_files = [f for f in os.listdir(drug_feat_dir) if f.endswith('.hkl')]
cids_feat = set(os.path.splitext(f)[0] for f in drug_files)
print(f"Drugs in graph feat dir: {len(cids_feat)}")

# 2. CIDs already in the physiochemical properties CSV
df_phys = pd.read_csv(phys_csv)
cids_phys = set(df_phys['PUBCHEM_CID'].astype(str))
print(f"Drugs already in properties CSV: {len(cids_phys)}")

# 3. Missing CIDs (Need to calculate for these)
missing_cids = cids_feat - cids_phys
print(f"Number of missing drugs to calculate: {len(missing_cids)}")

# 4. Find SMILES for these missing CIDs from pubchem_smiles_1.txt
# Using sep=None and engine='python' to auto-detect tab or space
df_smiles = pd.read_csv(smiles_txt, sep=None, engine='python', header=None, names=["PUBCHEM_CID", "SMILES"])
df_smiles['PUBCHEM_CID'] = df_smiles['PUBCHEM_CID'].astype(str)

missing_with_smiles = df_smiles[df_smiles['PUBCHEM_CID'].isin(missing_cids)]
print(f"Found SMILES for {len(missing_with_smiles)} out of {len(missing_cids)} missing drugs.")

if len(missing_with_smiles) < len(missing_cids):
    still_missing = missing_cids - set(missing_with_smiles['PUBCHEM_CID'])
    print(f"Still missing SMILES for: {list(still_missing)}")

# Save these to /tmp/missing_smiles.csv
missing_with_smiles.to_csv("/tmp/missing_smiles.csv", index=False)
print("Saved missing SMILES to /tmp/missing_smiles.csv")
