import pandas as pd

# Read the CSV file
df = pd.read_csv("new_data/GDSC/drug_to_cid.csv")  # change filename if needed

# Filter out rows where PUBCHEM_CID is missing or empty
df_filtered = df[df["PUBCHEM_CID"].notna() & (df["PUBCHEM_CID"].astype(str).str.strip() != "")]

df_filtered["PUBCHEM_CID"] = df_filtered["PUBCHEM_CID"].astype(float).astype(int)

# Extract the PUBCHEM_CID column (unique values if you want)
pubchem_ids = df_filtered["PUBCHEM_CID"].drop_duplicates()

# Save to a text file (one ID per line)
pubchem_ids.to_csv("pubchem_cids.txt", index=False, header=False)

print(f"Saved {len(pubchem_ids)} valid PubChem CIDs to pubchem_cids.txt")