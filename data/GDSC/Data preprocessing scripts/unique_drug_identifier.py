import pandas as pd

# Load the CSV file
csv_file = 'new_data/GDSC/drug_response_data.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Get unique values of the DRUG_NAME column
unique_drugs = df['DRUG_NAME'].dropna().unique()  # dropna() removes any missing values

# Save the unique values to a text file
txt_file = 'unique_drug_names.txt'
with open(txt_file, 'w') as f:
    for drug in unique_drugs:
        f.write(str(drug) + '\n')

print(f"{len(unique_drugs)} unique drug names saved to '{txt_file}'")