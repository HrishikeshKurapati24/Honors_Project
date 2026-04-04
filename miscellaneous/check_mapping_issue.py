import pandas as pd
import os

B_PATH = '/Volumes/Work/Semester - 6/Honors/CDRP models testing'
MAPPING_FILE = f'{B_PATH}/benchmark models/GPDRP-main/data/gpdrp_cell_mapping.csv'
EXPR_FILE = f'{B_PATH}/data/CCLE/Processed data/gene_expression_data.csv'

# 1. Load Mapped IDs
df_map = pd.read_csv(MAPPING_FILE)
depmap_ids = set(df_map['depMapID'])
print(f"Total Mapped DepMap IDs: {len(depmap_ids)}")

# 2. Load Expression IDs
print("Loading CCLE Expression ID column...")
df_expr_ids = pd.read_csv(EXPR_FILE, usecols=[0])
expr_cell_ids = set(df_expr_ids.iloc[:, 0])
print(f"Total Cell IDs in CCLE Expr: {len(expr_cell_ids)}")

# 3. Analyze Intersection
found = depmap_ids.intersection(expr_cell_ids)
print(f"Intersection size (Found): {len(found)}")

missing = depmap_ids - expr_cell_ids
print(f"Missing from Expr: {len(missing)}")

if missing:
    print("First 10 missing from Expr:", sorted(list(missing))[:10])

# 4. Check IDs format
if depmap_ids:
    print("Sample mapped ID:", list(depmap_ids)[0])
if expr_cell_ids:
    print("Sample expr ID:", list(expr_cell_ids)[0])
