import pandas as pd
import zipfile
import os

# Define paths
gdsc_ic50_path = "3OmicsBenchmarking/dataset-1/Celline/GDSC_IC50.csv"
drug_zip_path = "3OmicsBenchmarking/dataset-1/Drug/drug_graph_feat.zip"
drug_list_path = "3OmicsBenchmarking/dataset-1/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv"

# 1. Read GDSC IC50 and get drug names
print(f"Reading {gdsc_ic50_path}...")
df_gdsc = pd.read_csv(gdsc_ic50_path, index_col=0)
gdsc_drug_names = set(df_gdsc.columns.astype(str))
print(f"Found {len(gdsc_drug_names)} drug names in GDSC_IC50.csv")

# 2. Read Drug List to create mapping from drug_id to Name
print(f"Reading {drug_list_path}...")
df_drugs = pd.read_csv(drug_list_path)
# Create a dictionary mapping drug_id (as string) to Name
id_to_name = dict(zip(df_drugs['drug_id'].astype(str), df_drugs['Name'].astype(str)))
print(f"Found {len(id_to_name)} drug mappings.")

# 3. Read zip file, getting hkl drug IDs, then map them to Names
print(f"Reading contents of {drug_zip_path}...")
hkl_drug_ids = set()
with zipfile.ZipFile(drug_zip_path, 'r') as z:
    for filename in z.namelist():
        if filename.endswith('.hkl'):
            basename = os.path.basename(filename)
            drug_id = basename.replace('.hkl', '')
            hkl_drug_ids.add(drug_id)
print(f"Found {len(hkl_drug_ids)} drug IDs in {drug_zip_path}")

# Map hkl_drug_ids to Names
hkl_drug_names = set()
unmapped_ids = 0
for drug_id in hkl_drug_ids:
    if drug_id in id_to_name:
        hkl_drug_names.add(id_to_name[drug_id])
    else:
        unmapped_ids += 1

print(f"Mapped {len(hkl_drug_names)} drug IDs to Names. {unmapped_ids} IDs were unmapped.")

# 4. Find Intersection of names
intersection = gdsc_drug_names.intersection(hkl_drug_names)
print(f"Found {len(intersection)} drugs in common.")

# 5. Save intersection Names and IDs
output_file = "intersected_drugs.csv"
print(f"Saving intersection to {output_file}...")

# Let's save both Name and ID
name_to_id = {v: k for k, v in id_to_name.items()}
with open(output_file, 'w') as f:
    f.write("Drug_Name,Drug_ID\n")
    for name in sorted(list(intersection)):
        f.write(f"{name},{name_to_id.get(name, '')}\n")

print("Done.")
