import pandas as pd

# Load the CSV generated earlier
df = pd.read_csv("response_labels.csv")

# Drop rows where depMapID or PUBCHEM_CID is missing
filtered_df = df.dropna(subset=['depMapID', 'PUBCHEM_CID'])

# Save the filtered dataframe
filtered_df.to_csv("response_labels_filtered.csv", index=False)

print(f"Filtered data saved to drug_labels_filtered.csv ({len(filtered_df)} rows)")