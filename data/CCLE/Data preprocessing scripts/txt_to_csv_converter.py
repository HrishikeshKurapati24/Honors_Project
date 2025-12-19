import pandas as pd

# Read the tab-separated text file
df = pd.read_csv("non-csv data files/Cell_lines_annotations.txt", sep="\t")

# Save it as CSV
df.to_csv("transposed/Cell_lines_annotations.csv", index=False)