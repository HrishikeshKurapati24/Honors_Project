#!/bin/bash

# ==========================================================
# GraphCDR Modified Architecture - Phase 1 Experiments
#
# Purpose
#   Run full experiments to shortlist promising architectures.
#
# Output
#   - Logs for each run
#   - CSV summary with evaluation metrics
# ==========================================================

set -euo pipefail
cd "$(dirname "$0")"

echo "================================================"
echo " PHASE 1 : Shortlisting Experiments"
echo "================================================"
echo ""

PYTHON_BIN="${PYTHON_BIN:-python3}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="logs/phase1_shortlist_${TIMESTAMP}"
SUMMARY_CSV="${LOG_DIR}/phase1_summary.csv"

mkdir -p "${LOG_DIR}"

# ==========================================================
# Runtime configuration
# ==========================================================

KFOLD="${KFOLD:-1}"

BASE_CMD="${PYTHON_BIN} main.py --execution_architecture modified --k_fold ${KFOLD}"

# ==========================================================
# Dataset paths
# ==========================================================

CHROMATIN="--genomics_csv '../../final_dataset/genomics_chromatin.csv'"
MUTATION="--genomics_csv '../../final_dataset/genomics_mutation.csv'"

# ==========================================================
# Omics configurations
# ==========================================================

OMICS_3="--use_genomics --use_epigenomics --use_transcriptomics"
OMICS_6="${OMICS_3} --use_proteomics --use_metabolomics --use_pathway"

echo "run_name,status,auc,aupr,f1,acc,time(s),log_file" > "${SUMMARY_CSV}"

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

    # Parse metrics from log
    FINAL_LINE=$(grep -E "Final Results .AUC=" "${log_file}" | tail -1 || true)

    if [ -n "${FINAL_LINE}" ]; then
        AUC=$(echo "${FINAL_LINE}" | sed -n 's/.AUC=([0-9.])./\1/p')
        AUPR=$(echo "${FINAL_LINE}" | sed -n 's/.AUPR=([0-9.])./\1/p')
        F1=$(echo "${FINAL_LINE}" | sed -n 's/.F1=([0-9.])./\1/p')
        ACC=$(echo "${FINAL_LINE}" | sed -n 's/.ACC=([0-9.]).*/\1/p')
    else
        LAST_TEST_LINE=$(grep -E "test auc:" "${log_file}" | tail -1 || true)

        AUC=$(echo "${LAST_TEST_LINE}" | awk '{for(i=1;i<=NF;i++) if($i=="auc:"){print $(i+1); exit}}')
        AUPR=$(echo "${LAST_TEST_LINE}" | awk '{for(i=1;i<=NF;i++) if($i=="aupr:"){print $(i+1); exit}}')
        F1=$(echo "${LAST_TEST_LINE}" | awk '{for(i=1;i<=NF;i++) if($i=="f1:"){print $(i+1); exit}}')
        ACC=$(echo "${LAST_TEST_LINE}" | awk '{for(i=1;i<=NF;i++) if($i=="acc:"){print $(i+1); exit}}')
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

run_case "S1_6Omics_Chromatin_GIN" "${CHROMATIN} --gnn_type GIN ${OMICS_6}"
run_case "S1_6Omics_Mutation_GIN" "${MUTATION} --gnn_type GIN ${OMICS_6}"

run_case "S1_3Omics_Chromatin_GIN" "${CHROMATIN} --gnn_type GIN ${OMICS_3}"
run_case "S1_3Omics_Mutation_GIN" "${MUTATION} --gnn_type GIN ${OMICS_3}"

run_case "S1_Trans_Epi_GIN" "${CHROMATIN} --gnn_type GIN --use_transcriptomics --use_epigenomics"

run_case "S1_Genomics_Trans_GIN" "${CHROMATIN} --gnn_type GIN --use_genomics --use_transcriptomics"

# Additional modalities (old Stage 6)

run_case "S1_Trans_Proteomics"   "${CHROMATIN} --use_transcriptomics --use_proteomics"
run_case "S1_Trans_Metabolomics" "${CHROMATIN} --use_transcriptomics --use_metabolomics"

run_case "S1_Trans_miRNA" "--use_transcriptomics --use_proteomics --proteomics_csv '../../new_data/CCLE/Processed data/miRNA_expression_data.csv'"

# Extra modality experiments

run_case "S1_Mutation_Expression" "${MUTATION} --use_genomics --use_transcriptomics"
run_case "S1_Pathway_Only" "${MUTATION} --use_pathway"
run_case "S1_4Omics_Mut_Meth_Expr_Pathway" "${MUTATION} --use_genomics --use_epigenomics --use_transcriptomics --use_pathway"


# ==========================================================
# Stage 2 : GNN Architecture Comparison
# ==========================================================

# --- 3 Omics ---

run_case "S2_3Omics_GIN" "${CHROMATIN} --gnn_type GIN ${OMICS_3}"
run_case "S2_3Omics_GCN" "${CHROMATIN} --gnn_type GCN ${OMICS_3}"
run_case "S2_3Omics_GraphSAGE" "${CHROMATIN} --gnn_type GraphSAGE ${OMICS_3}"
run_case "S2_3Omics_GAT" "${CHROMATIN} --gnn_type GAT ${OMICS_3}"

# --- 6 Omics ---

run_case "S2_6Omics_GIN" "${CHROMATIN} --gnn_type GIN ${OMICS_6}"
run_case "S2_6Omics_GCN" "${CHROMATIN} --gnn_type GCN ${OMICS_6}"
run_case "S2_6Omics_GraphSAGE" "${CHROMATIN} --gnn_type GraphSAGE ${OMICS_6}"
run_case "S2_6Omics_GAT" "${CHROMATIN} --gnn_type GAT ${OMICS_6}"


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


echo ""
echo "================================================"
echo " PHASE 1 COMPLETE"
echo " Results: ${SUMMARY_CSV}"
echo "================================================"