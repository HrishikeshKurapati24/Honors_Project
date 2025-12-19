import pandas as pd
import requests
import time

# Load your list of PubChem IDs
df = pd.read_csv("new_data/GDSC/drug_to_cid.csv")  # must contain column 'PUBCHEM_CID'
smiles_data = []

for cid in df["PUBCHEM_CID"].dropna().astype(int):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/CSV"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            smiles_line = response.text.strip().split('\n')[-1]  # second line contains CID, SMILES
            cid_value, smiles = smiles_line.split(',')
            smiles_data.append((cid_value, smiles))
            print(smiles)
        else:
            smiles_data.append((cid, None))
    except Exception as e:
        smiles_data.append((cid, None))
    time.sleep(0.2)  # small delay to avoid server throttling

smiles_df = pd.DataFrame(smiles_data, columns=["PUBCHEM_CID", "SMILES"])
smiles_df.to_csv("PubChem_SMILES_data.csv", index=False)
print("✅ SMILES downloaded and saved as PubChem_SMILES_data.csv")