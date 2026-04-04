import pandas as pd
import os

# Define paths
base_dir = "/Volumes/Work/Semester - 6/Honors/GraphCDR"
ccle_proc_dir = os.path.join(base_dir, "new_data/CCLE/Processed data")
gdsc_proc_dir = os.path.join(base_dir, "new_data/GDSC/Processed data")

def count_rows(directory, label):
    print(f"\n--- {label} Row Counts ---")
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    files = [f for f in os.listdir(directory) if not f.startswith('.')] # Ignore hidden files
    for f in sorted(files):
        path = os.path.join(directory, f)
        if os.path.isdir(path): continue
        try:
            # For non-csv or general files, generally counting lines is safer if pd.read_csv fails
            if f.endswith('.csv'):
                df = pd.read_csv(path, usecols=[0])
                print(f"{f}: {len(df)}")
            else:
                 with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                    count = sum(1 for _ in fp)
                    # Adjust for header if it looks like a table? 
                    # Usually "rows" implies data entries. text files might not have headers or might.
                    # I'll just print raw line count for non-csv and let user interpret, or assume 1 header?
                    # Safer to print raw lines for txt.
                    print(f"{f}: {count} (lines)")
        except Exception as e:
            print(f"Error reading {f}: {e}")

# 1. Count rows
count_rows(ccle_proc_dir, "CCLE Processed Data")
count_rows(gdsc_proc_dir, "GDSC Processed Data")

print("\n--- Intersections ---")

# Define file paths for intersections
f_chromatin = os.path.join(ccle_proc_dir, "chromatin_profiling.csv")
f_meth_vista = os.path.join(ccle_proc_dir, "DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv")
f_expr = os.path.join(ccle_proc_dir, "gene_expression_data.csv")
f_mut = os.path.join(ccle_proc_dir, "mutation_data.csv")
f_meth_promoter = os.path.join(ccle_proc_dir, "DNA_methylation_promoter_CpG_clusters.csv")
f_response = os.path.join(gdsc_proc_dir, "response_labels.csv")

def get_indices(path, col_idx=0):
    try:
        # Assumes index/ID is in the first column based on inspection
        df = pd.read_csv(path, usecols=[col_idx])
        return set(df.iloc[:, 0].astype(str))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return set()

# Intersection 1: Chromatin + Meth(Vista) + Expr
s_chromatin = get_indices(f_chromatin)
s_meth_vista = get_indices(f_meth_vista)
s_expr = get_indices(f_expr)

# Intersection 2: Mutation + Meth(Vista) + Expr  <-- CHANGED: Use Vista instead of Promoter
s_mut = get_indices(f_mut)
# s_meth_promoter was used before, now using s_meth_vista
# s_meth_promoter = get_indices(f_meth_promoter) 

# Calculate Intersections
intersect1 = s_chromatin & s_meth_vista & s_expr
intersect2 = s_mut & s_meth_vista & s_expr  # Changed to use Vista

print(f"Intersection 1 (Chromatin + Meth(Vista) + Expr): {len(intersect1)}")
print(f"Intersection 2 (Mutation + Meth(Vista) + Expr): {len(intersect2)}")

# Drugs Available
try:
    df_resp = pd.read_csv(f_response)
    # Ensure columns are treated as strings for intersection
    df_resp['cell_line_id'] = df_resp['cell_line_id'].astype(str)
    df_resp['pubchem_id'] = df_resp['pubchem_id'].astype(str)
    
    drugs_available = set(df_resp['pubchem_id'].unique())
    print(f"No of drugs available (from response_labels.csv): {len(drugs_available)}")
    
    # 1. Cell-line intersection 1 with @response_labels
    resp_cell_lines = set(df_resp['cell_line_id'])
    intersect1_with_resp = intersect1 & resp_cell_lines
    print(f"Intersection 1 with Response Labels (Cells): {len(intersect1_with_resp)}")
    
    # Function to print filtered stats
    def print_filtered_stats(valid_cells_set, name):
        df_filtered = df_resp[df_resp['cell_line_id'].isin(valid_cells_set)]
        
        final_cells = df_filtered['cell_line_id'].nunique()
        final_drugs = df_filtered['pubchem_id'].nunique()
        final_responses = len(df_filtered)
        
        print(f"\n--- Complete Intersection Stats ({name} & Response Labels) ---")
        print(f"Cell-lines: {final_cells}")
        print(f"Drugs: {final_drugs}")
        print(f"Response Values (Rows in filtered response_labels): {final_responses}")

    # Calculate and print for both
    print_filtered_stats(intersect1, "Intersection 1")
    print_filtered_stats(intersect2, "Intersection 2")

except Exception as e:
    print(f"Error processing response_labels.csv: {e}")
