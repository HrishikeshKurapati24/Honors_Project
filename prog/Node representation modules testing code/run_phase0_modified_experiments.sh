#!/bin/bash

# ==========================================================
# GraphCDR Modified Architecture - Phase 0 Smoke Tests
# ==========================================================

set -euo pipefail

cd "$(dirname "$0")"

echo "================================================"
echo " PHASE 0 : Modified Architecture Smoke Testing"
echo "================================================"
echo "Python executable: ${PYTHON_BIN:-$(which python3)}"
echo ""

# ----------------------------------------------------------
# Environment setup
# ----------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

LOG_DIR="logs/phase0_modified_${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/phase0_summary.csv"

mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------
# Runtime configuration
# ----------------------------------------------------------

EPOCHS="${EPOCHS:-1}"
KFOLD="${KFOLD:-1}"

BASE_CMD="${PYTHON_BIN} main.py \
--epoch ${EPOCHS} \
--execution_architecture modified \
--k_fold ${KFOLD}"

# ----------------------------------------------------------
# Dataset paths
# ----------------------------------------------------------

CHROMATIN_DATA="--genomics_csv '../../final_dataset/genomics_chromatin.csv'"
MUTATION_DATA="--genomics_csv '../../final_dataset/genomics_mutation.csv'"

# ----------------------------------------------------------
# Omics combinations
# ----------------------------------------------------------

OMICS_3="--use_genomics --use_epigenomics --use_transcriptomics"
OMICS_6="${OMICS_3} --use_proteomics --use_metabolomics --use_pathway"

# ----------------------------------------------------------
# Initialize summary file
# ----------------------------------------------------------

echo "run_name,status,log_file,notes" > "${SUMMARY_FILE}"

# ==========================================================
# Experiment runner
# ==========================================================

run_case() {

    local run_name="$1"
    local run_args="$2"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo "------------------------------------------------"
    echo "Running experiment: ${run_name}"
    echo "------------------------------------------------"

    echo "${BASE_CMD} ${run_args}"
    echo ""

    set +e
    eval "${BASE_CMD} ${run_args}" > "${log_file}" 2>&1
    local exit_code=$?
    set -e

    if [ ${exit_code} -eq 0 ]; then
        echo "${run_name},PASS,${log_file},completed" >> "${SUMMARY_FILE}"
        echo "Status: SUCCESS"
    else
        echo "${run_name},FAIL,${log_file},exit_code=${exit_code}" >> "${SUMMARY_FILE}"
        echo "Status: FAILED (exit ${exit_code})"
    fi

    echo ""
}

# ==========================================================
# Stage 1 : Omics / Modality Combinations
# (Includes previous Stage 6 + Extra Experiments)
# ==========================================================

run_case "S1_6Omics_Chromatin_GIN" "${CHROMATIN_DATA} --gnn_type GIN ${OMICS_6}"
run_case "S1_6Omics_Mutation_GIN"  "${MUTATION_DATA} --gnn_type GIN ${OMICS_6}"

run_case "S1_3Omics_Chromatin_GIN" "${CHROMATIN_DATA} --gnn_type GIN ${OMICS_3}"
run_case "S1_3Omics_Mutation_GIN"  "${MUTATION_DATA} --gnn_type GIN ${OMICS_3}"

run_case "S1_Trans_Epi_GIN"        "${CHROMATIN_DATA} --gnn_type GIN --use_transcriptomics --use_epigenomics"
run_case "S1_Genomics_Trans_GIN"   "${CHROMATIN_DATA} --gnn_type GIN --use_genomics --use_transcriptomics"

# Additional modalities (old Stage 6)

run_case "S1_Trans_Proteomics"   "${CHROMATIN_DATA} --use_transcriptomics --use_proteomics"
run_case "S1_Trans_Metabolomics" "${CHROMATIN_DATA} --use_transcriptomics --use_metabolomics"

run_case "S1_Trans_miRNA" "--use_transcriptomics --use_proteomics --proteomics_csv '../../new_data/CCLE/Processed data/miRNA_expression_data.csv'"

# Extra combinations

run_case "S1_Mutation_Expression" "${MUTATION_DATA} --use_genomics --use_transcriptomics"
run_case "S1_Pathway_Only"        "${MUTATION_DATA} --use_pathway"
run_case "S1_4Omics_Mut_Meth_Expr_Pathway" "${MUTATION_DATA} --use_genomics --use_epigenomics --use_transcriptomics --use_pathway"


# ==========================================================
# Stage 2 : GNN Variants
# (Merged old Stage 2 and Stage 3)
# ==========================================================

# --- 3 Omics ---

run_case "S2_3Omics_GIN"       "${CHROMATIN_DATA} --gnn_type GIN ${OMICS_3}"
run_case "S2_3Omics_GCN"       "${CHROMATIN_DATA} --gnn_type GCN ${OMICS_3}"
run_case "S2_3Omics_GraphSAGE" "${CHROMATIN_DATA} --gnn_type GraphSAGE ${OMICS_3}"
run_case "S2_3Omics_GAT"       "${CHROMATIN_DATA} --gnn_type GAT ${OMICS_3}"

# --- 6 Omics ---

run_case "S2_6Omics_GIN"       "${CHROMATIN_DATA} --gnn_type GIN ${OMICS_6}"
run_case "S2_6Omics_GCN"       "${CHROMATIN_DATA} --gnn_type GCN ${OMICS_6}"
run_case "S2_6Omics_GraphSAGE" "${CHROMATIN_DATA} --gnn_type GraphSAGE ${OMICS_6}"
run_case "S2_6Omics_GAT"       "${CHROMATIN_DATA} --gnn_type GAT ${OMICS_6}"


# ==========================================================
# Stage 3 : Cell-Line Representation Variants
# ==========================================================

run_case "S3_CellModule_FC" "${CHROMATIN_DATA} ${OMICS_3} --cell_line_module_variation FC"
run_case "S3_CellModule_AE" "${CHROMATIN_DATA} ${OMICS_3} --cell_line_module_variation AE"


# ==========================================================
# Stage 4 : Drug Representation Variants
# ==========================================================

run_case "S4_Transformer_Drug" "${CHROMATIN_DATA} --gnn_type GIN --use_transcriptomics --use_epigenomics --use_transformer_drug"

run_case "S4_Enhanced_Drug_Active" "${CHROMATIN_DATA} ${OMICS_3} --active"


# ==========================================================
# Completion Message
# ==========================================================

echo "================================================"
echo " PHASE 0 COMPLETE"
echo "================================================"
echo "Summary file : ${SUMMARY_FILE}"
echo "Log directory: ${LOG_DIR}"
echo "================================================"