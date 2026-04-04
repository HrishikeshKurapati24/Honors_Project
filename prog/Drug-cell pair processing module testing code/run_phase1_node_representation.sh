#!/bin/bash

# ==========================================================
# Drug-Cell Pair Processing - Phase 1 Node Representation
# Optimized for Synchronized Architecture
# ==========================================

set -euo pipefail
cd "$(dirname "$0")"

echo "================================================"
echo " PHASE 1 : Node Representation Sync Experiments"
echo "================================================"
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="logs/phase1_sync_${TIMESTAMP}"
SUMMARY_CSV="${LOG_DIR}/phase1_summary.csv"

mkdir -p "${LOG_DIR}"

# ==========================================================
# Runtime configuration
# ==========================================================

KFOLD="${KFOLD:-2}"
SOULCDR_CONFIG="--graph_GNN_type heterogenous --global_gnn_type SAGE --use_graph_transformer"
BASE_CMD="${PYTHON_BIN} main.py --k_fold ${KFOLD} ${SOULCDR_CONFIG}"

# ==========================================================
# Dataset base filenames (passed to --genomics_file)
# ==========================================================

CHROMATIN="--genomics_file genomics_chromatin.csv"
MUTATION="--genomics_file genomics_mutation.csv"

# ==========================================================
# Omics configurations
# ==========================================================

OMICS_3="--use_genomics --use_epigenomics --use_transcriptomics"
OMICS_6="${OMICS_3} --use_proteomics --use_metabolomics --use_pathway"

echo "case_name,status,auc,aupr,f1,acc,time(s),log_file" > "${SUMMARY_CSV}"

# ==========================================================
# Experiment runner
# ==========================================================

run_case() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    echo "------------------------------------------"
    echo "Running ${name} | k_fold ${KFOLD}"
    echo "------------------------------------------"

    CMD="${BASE_CMD} ${args}"

    start_time=$SECONDS
    set +e
    eval "${CMD}" | tee "${log_file}"
    rc=$?
    set -e
    elapsed=$((SECONDS - start_time))

    if [ ${rc} -ne 0 ]; then
        echo "${name},FAIL,,,,,${elapsed},${log_file}" >> "${SUMMARY_CSV}"
        echo "FAILED: ${name}"
        return
    fi

    # Parse metrics from log (Drug-cell main.py outputs: Final_AUC: 0.8... or Average Final_AUC: 0.8...)
    AUC=$(grep -E "Final_AUC:" "${log_file}" | tail -1 | sed -n 's/.*Final_AUC: \([0-9.]*\).*/\1/p')
    AUPR=$(grep -E "Final_AUPR:" "${log_file}" | tail -1 | sed -n 's/.*Final_AUPR: \([0-9.]*\).*/\1/p')
    F1=$(grep -E "Final_F1:" "${log_file}" | tail -1 | sed -n 's/.*Final_F1: \([0-9.]*\).*/\1/p')
    ACC=$(grep -E "Final_ACC:" "${log_file}" | tail -1 | sed -n 's/.*Final_ACC: \([0-9.]*\).*/\1/p')

    # If K-fold avg output is different
    if [ -z "$AUC" ]; then
        AUC=$(grep -E "Average Final_AUC:" "${log_file}" | tail -1 | sed -n 's/.*Average Final_AUC: \([0-9.]*\).*/\1/p')
        AUPR=$(grep -E "Average Final_AUPR:" "${log_file}" | tail -1 | sed -n 's/.*Average Final_AUPR: \([0-9.]*\).*/\1/p')
        F1=$(grep -E "Average Final_F1:" "${log_file}" | tail -1 | sed -n 's/.*Average Final_F1: \([0-9.]*\).*/\1/p')
        ACC=$(grep -E "Average Final_ACC:" "${log_file}" | tail -1 | sed -n 's/.*Average Final_ACC: \([0-9.]*\).*/\1/p')
    fi

    AUC=${AUC:-NA}
    AUPR=${AUPR:-NA}
    F1=${F1:-NA}
    ACC=${ACC:-NA}

    echo "${name},PASS,${AUC},${AUPR},${F1},${ACC},${elapsed},${log_file}" >> "${SUMMARY_CSV}"
}

# ==========================================================
# Stage 1 : Modality / Omics Comparisons
# ==========================================================

run_case "S1_6Omics_Chromatin_GIN" "${CHROMATIN} --drug_gnn_type GIN ${OMICS_6}"
run_case "S1_6Omics_Mutation_GIN" "${MUTATION} --drug_gnn_type GIN ${OMICS_6}"

run_case "S1_3Omics_Chromatin_GIN" "${CHROMATIN} --drug_gnn_type GIN ${OMICS_3}"
run_case "S1_3Omics_Mutation_GIN" "${MUTATION} --drug_gnn_type GIN ${OMICS_3}"

run_case "S1_Trans_Epi_GIN" "${CHROMATIN} --drug_gnn_type GIN --use_transcriptomics --use_epigenomics"
run_case "S1_Genomics_Trans_GIN" "${CHROMATIN} --drug_gnn_type GIN --use_genomics --use_transcriptomics"

run_case "S1_Trans_Proteomics"   "${CHROMATIN} --use_transcriptomics --use_proteomics"
run_case "S1_Trans_Metabolomics" "${CHROMATIN} --use_transcriptomics --use_metabolomics"

# ==========================================================
# Stage 2 : Drug GNN Architecture Comparison
# ==========================================================

run_case "S2_3Omics_GIN" "${CHROMATIN} --drug_gnn_type GIN ${OMICS_3}"
run_case "S2_3Omics_GCN" "${CHROMATIN} --drug_gnn_type GCN ${OMICS_3}"
run_case "S2_3Omics_GraphSAGE" "${CHROMATIN} --drug_gnn_type GraphSAGE ${OMICS_3}"
run_case "S2_3Omics_GAT" "${CHROMATIN} --drug_gnn_type GAT ${OMICS_3}"

run_case "S2_6Omics_GIN" "${CHROMATIN} --drug_gnn_type GIN ${OMICS_6}"
run_case "S2_6Omics_GCN" "${CHROMATIN} --drug_gnn_type GCN ${OMICS_6}"
run_case "S2_6Omics_GraphSAGE" "${CHROMATIN} --drug_gnn_type GraphSAGE ${OMICS_6}"
run_case "S2_6Omics_GAT" "${CHROMATIN} --drug_gnn_type GAT ${OMICS_6}"

# ==========================================================
# Stage 3 : Cell-Line Encoder Variants
# ==========================================================

run_case "S3_CellModule_FC" "${CHROMATIN} ${OMICS_3} --cell_line_module_variation FC"
run_case "S3_CellModule_AE" "${CHROMATIN} ${OMICS_3} --cell_line_module_variation AE"

# ==========================================================
# Stage 4 : Drug Representation Variants
# ==========================================================

run_case "S4_Transformer_Drug" "${CHROMATIN} ${OMICS_3} --use_transformer_drug"
run_case "S4_Enhanced_Drug_Active" "${CHROMATIN} ${OMICS_3} --active"

# ==========================================================
# Extra scenarios
# ==========================================================

run_case "S5_Mutation_Expression" "${MUTATION} --use_genomics --use_transcriptomics"
run_case "S5_Pathway_Only" "${MUTATION} --use_pathway"
run_case "S5_4Omics_Mut_Meth_Expr_Pathway" "${MUTATION} --use_genomics --use_epigenomics --use_transcriptomics --use_pathway"

echo ""
echo "================================================"
echo " PHASE 1 COMPLETE"
echo " Results: ${SUMMARY_CSV}"
echo "================================================"