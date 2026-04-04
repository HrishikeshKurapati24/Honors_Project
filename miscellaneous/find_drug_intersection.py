import pandas as pd
import zipfile
import os

csv_path = "3OmicsBenchmarking/dataset-1/Celline/GDSC_IC50.csv"
zip_path = "3OmicsBenchmarking/dataset-1/Drug/drug_graph_feat.zip"
output_path = "intersecting_drugs.txt"

def main():
    print(f"Reading CSV header from {csv_path}...")
    # Read just the header row
    df = pd.read_csv(csv_path, nrows=0)
    # the first column is likely "Cell Line" or "id"
    # drug IDs are the rest of columns
    csv_drug_ids = list(df.columns)[1:]
    # Strip any extra whitespace
    csv_drug_ids = [str(d).strip() for d in csv_drug_ids]
    print(f"Found {len(csv_drug_ids)} drug IDs in CSV header. (Examples: {csv_drug_ids[:5]})")

    print(f"Reading zip archive listing from {zip_path}...")
    hkl_drug_ids = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.hkl'):
                # Extract filename without extension
                base = os.path.basename(name)
                drug_id = os.path.splitext(base)[0]
                hkl_drug_ids.append(str(drug_id).strip())

    print(f"Found {len(hkl_drug_ids)} drug IDs in zip file. (Examples: {hkl_drug_ids[:5]})")

    # intersection
    intersection = set(csv_drug_ids).intersection(set(hkl_drug_ids))
    print(f"\nFound {len(intersection)} intersecting drug IDs.")

    # Save to file
    with open(output_path, "w") as f:
        for drug in sorted(list(intersection)):
            f.write(f"{drug}\n")

    print(f"List of intersecting drug IDs saved to {output_path}")

if __name__ == "__main__":
    main()
