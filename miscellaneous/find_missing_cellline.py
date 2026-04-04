import pandas as pd
import os

base_path = "/Volumes/Work/Semester - 6/Honors/CDRP models testing"
cellline_path = os.path.join(base_path, "benchmark models/GraphCDR/data/Celline")

# Files
progeny_file = os.path.join(base_path, "data/CCLE/Processed data/cellline_pathway_activity_PROGENy.csv")
mut_file = os.path.join(cellline_path, "genomic_mutation_34673_demap_features.csv")
meth_file = os.path.join(cellline_path, "genomic_methylation_561celllines_808genes_demap_features.csv")
trans_file = os.path.join(cellline_path, "genomic_expression_561celllines_697genes_demap_features.csv")

def get_cells(path):
    df = pd.read_csv(path, usecols=[0], index_col=0)
    return set(df.index.astype(str))

cells_pro = get_cells(progeny_file)
cells_mut = get_cells(mut_file)
cells_meth = get_cells(meth_file)
cells_trans = get_cells(trans_file)

# Intersection of GraphCDR 3 Omics
graphcdr_intersection = cells_mut.intersection(cells_meth).intersection(cells_trans)

print(f"GraphCDR 3-Omics Intersection size: {len(graphcdr_intersection)}")

# Find the one in GraphCDR but not in Progeny
missing_from_progeny = graphcdr_intersection - cells_pro

print(f"Cell-lines in GraphCDR 3-Omics but MISSING in Progeny: {missing_from_progeny}")
