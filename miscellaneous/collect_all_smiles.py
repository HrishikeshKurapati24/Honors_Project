import os
import pandas as pd

base_path = "/Volumes/Work/Semester - 6/Honors/CDRP models testing"
phys_csv = os.path.join(base_path, "data/GDSC/Processed data/pubchem_physiochemical_properties_1.csv")
smiles_txt_222 = os.path.join(base_path, "benchmark models/GraphCDR/data/Drug/222drugs_pubchem_smiles.txt")
smiles_txt_main = os.path.join(base_path, "data/GDSC/Processed data/pubchem_smiles_1.txt")

# 1. CIDs in existing properties file (298)
df_phys = pd.read_csv(phys_csv)
cids_existing = set(df_phys['PUBCHEM_CID'].astype(str))
print(f"Existing CIDs: {len(cids_existing)}")

# 2. CIDs in 222 list
df_222 = pd.read_csv(smiles_txt_222, sep=None, engine='python', header=None, names=["PUBCHEM_CID", "SMILES"])
df_222['PUBCHEM_CID'] = df_222['PUBCHEM_CID'].astype(str)
cids_222 = set(df_222['PUBCHEM_CID'])
print(f"CIDs in 222 list: {len(cids_222)}")

# 3. New ones (40)
missing_cids = cids_222 - cids_existing
print(f"Missing CIDs: {len(missing_cids)}")

# 4. Total CIDs needed (298 + 40 = 338)
total_cids = cids_existing.union(cids_222)
print(f"Total unique CIDs: {len(total_cids)}")

# 5. Collect SMILES for all 338
# Load main smiles file
df_main = pd.read_csv(smiles_txt_main, sep=None, engine='python', header=None, names=["PUBCHEM_CID", "SMILES"])
df_main['PUBCHEM_CID'] = df_main['PUBCHEM_CID'].astype(str)

combined_smiles = pd.concat([df_222, df_main]).drop_duplicates('PUBCHEM_CID')
final_list = combined_smiles[combined_smiles['PUBCHEM_CID'].isin(total_cids)]

print(f"Final list with SMILES: {len(final_list)}")

if len(final_list) < len(total_cids):
    still_missing = total_cids - set(final_list['PUBCHEM_CID'])
    print(f"Still missing SMILES for {len(still_missing)} drugs: {list(still_missing)}")
else:
    final_list.to_csv("/tmp/all_smiles_to_calculate.csv", index=False)
    print("Saved all SMILES to /tmp/all_smiles_to_calculate.csv")
