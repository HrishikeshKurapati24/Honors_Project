import pubchempy as pcp
import pandas as pd

# Read drug names from the txt file
txt_file = 'unique_drug_names.txt'  # replace with your file path
with open(txt_file, 'r') as f:
    drug_names = [line.strip() for line in f if line.strip()]  # remove empty lines

# Store results
drug_to_cid = {}

for drug in drug_names:
    try:
        compounds = pcp.get_compounds(drug, 'name')  # search by name
        if compounds:
            drug_to_cid[drug] = compounds[0].cid
            print(compounds[0].cid)
        else:
            drug_to_cid[drug] = None
    except Exception as e:
        drug_to_cid[drug] = None

# Convert to DataFrame and save
df = pd.DataFrame(list(drug_to_cid.items()), columns=['DRUG_NAME', 'PUBCHEM_CID'])
df.to_csv('drug_to_cid.csv', index=False)

print(df)