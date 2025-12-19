import pandas as pd

# Load your raw gene expression
gene_expr = pd.read_csv("new_data/CCLE/gene_expression_data.csv", sep="\t")

# Load GPDRP's pathway-level expression(GPDRP and GraphCDR have this same file, but GraphCDR doesn't use it. GPDRP uses it as input for cell-line data.)
gpdrp_expr = pd.read_csv("data/Celline/cell_ge.csv")

# Get cell line columns (excluding the first identifier columns)
gene_expr_cells = set(gene_expr.columns[2:])  # Skip gene_id, transcript_ids
gpdrp_cells = set(gpdrp_expr.columns[2:])     # Skip GENE_SYMBOLS, GENE_title

# Find which are missing
missing_in_gpdrp = gene_expr_cells - gpdrp_cells
print("Cell lines missing in GPDRP:", missing_in_gpdrp)