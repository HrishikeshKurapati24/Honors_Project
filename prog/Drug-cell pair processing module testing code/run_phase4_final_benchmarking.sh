#!/bin/bash
# --------------------------------------------------------------------------------
# PHASE 4 — 5-Fold Cross Validation
# --------------------------------------------------------------------------------

# 6-Omics Flags (Full Stack)
OMICS_FLAGS_6="--use_genomics --use_epigenomics --use_transcriptomics --use_proteomics --use_metabolomics --use_pathway"
# 3-Omics Flags (Baseline Comparison)
OMICS_FLAGS_3="--use_genomics --use_epigenomics --use_transcriptomics"

DRUG_CONFIG="--drug_gnn_type GIN"
KFOLD_FIVE="--k_fold 5"
KFOLD_ONE="--k_fold 1"
CONTRASTIVE_ARGS="--use_contrastive"

# ============================================================================
echo "===================================================================="
echo "P4.2: Top 3 Model Robustness (Stage 3.2, 5-Fold)"
echo "Goal: Run select models with 5-fold CV to establish robustness"
echo "===================================================================="

echo "1. Model A: 3 Omics + Hetero + SAGE + GT=ON (CL)"
python3 main.py $KFOLD_FIVE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_3

echo "2. Model B: 3 Omics + Hetero + SAGE + GT=ON (CL)"
python3 main.py $KFOLD_FIVE \
    --graph_GNN_type heterogenous \
    --global_gnn_type GAT \
    --use_graph_transformer \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_3

echo "3. Model C: Pathway only + Hetero + SAGE + GT=ON (CL)"
python3 main.py $KFOLD_FIVE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    --use_pathway

# ============================================================================
echo "===================================================================="
echo "P4.3: Training Curves & ROC Data Generation (Stage 3.3, 1-Fold)"
echo "Goal: Save training metrics and predictions for the top models"
echo "===================================================================="

echo "1. Model A: 3 Omics + Hetero + SAGE + GT=ON (CL)"
python3 main.py $KFOLD_ONE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    --save_predictions \
    --save_training_curves \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_3

echo "2. Model B: 3 Omics + Hetero + GAT + GT=ON (CL)"
python3 main.py $KFOLD_ONE \
    --graph_GNN_type heterogenous \
    --global_gnn_type GAT \
    --use_graph_transformer \
    --save_predictions \
    --save_training_curves \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_3

echo "3. Model C: Pathway only + Hetero + SAGE + GT=ON (CL)"
python3 main.py $KFOLD_ONE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    --save_predictions \
    --save_training_curves \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    --use_pathway

# ============================================================================
echo "===================================================================="
echo "P4.4: Contrastive Learning Ablation (Bar Chart Data)"
echo "Goal: Compare top model with vs without contrastive learning"
echo "===================================================================="

ABLATION_CL_LOG="logs/phase4_cl_ablation.csv"
echo "model,use_cl,auc,aupr,f1,acc" > "$ABLATION_CL_LOG"

run_ablation() {
    local label="$1"
    local use_cl="$2"
    local cl_args="$3"
    local tmp="logs/tmp_ablation.txt"

    python3 main.py $KFOLD_FIVE \
        --graph_GNN_type heterogenous \
        --global_gnn_type SAGE \
        --use_graph_transformer \
        $cl_args \
        $DRUG_CONFIG \
        $OMICS_FLAGS_3 | tee "$tmp"

    AUC=$(grep -E "Final_AUC:|Average Final_AUC:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    AUPR=$(grep -E "Final_AUPR:|Average Final_AUPR:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    F1=$(grep -E "Final_F1:|Average Final_F1:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    ACC=$(grep -E "Final_ACC:|Average Final_ACC:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    echo "$label,$use_cl,${AUC:-NA},${AUPR:-NA},${F1:-NA},${ACC:-NA}" >> "$ABLATION_CL_LOG"
    rm "$tmp"
}

run_ablation "Hetero+SAGE+GT" "yes" "$CONTRASTIVE_ARGS"
run_ablation "Hetero+SAGE+GT" "no"  ""

echo "CL ablation results saved to $ABLATION_CL_LOG"

# ============================================================================
echo "===================================================================="
echo "P4.5: Omics Contribution Ablation (Bar Chart Data)"
echo "Goal: Drop one omics modality at a time from the 6-omics full stack"
echo "===================================================================="

ABLATION_OMICS_LOG="logs/phase4_omics_ablation.csv"
echo "dropped_modality,auc,aupr,f1,acc" > "$ABLATION_OMICS_LOG"

run_omics_ablation() {
    local label="$1"
    local omics_args="$2"
    local tmp="logs/tmp_omics_ablation.txt"

    python3 main.py $KFOLD_FIVE \
        --graph_GNN_type heterogenous \
        --global_gnn_type SAGE \
        --use_graph_transformer \
        $CONTRASTIVE_ARGS \
        $DRUG_CONFIG \
        $omics_args | tee "$tmp"

    AUC=$(grep -E "Final_AUC:|Average Final_AUC:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    AUPR=$(grep -E "Final_AUPR:|Average Final_AUPR:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    F1=$(grep -E "Final_F1:|Average Final_F1:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    ACC=$(grep -E "Final_ACC:|Average Final_ACC:" "$tmp" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    echo "$label,${AUC:-NA},${AUPR:-NA},${F1:-NA},${ACC:-NA}" >> "$ABLATION_OMICS_LOG"
    rm "$tmp"
}

# Full 6-omics baseline
ALL6="--use_genomics --use_epigenomics --use_transcriptomics --use_proteomics --use_metabolomics --use_pathway"
run_omics_ablation "All_6_Omics" "$ALL6"

# Drop one at a time
run_omics_ablation "Drop_Genomics"       "--use_epigenomics --use_transcriptomics --use_proteomics --use_metabolomics --use_pathway"
run_omics_ablation "Drop_Epigenomics"    "--use_genomics --use_transcriptomics --use_proteomics --use_metabolomics --use_pathway"
run_omics_ablation "Drop_Transcriptomics" "--use_genomics --use_epigenomics --use_proteomics --use_metabolomics --use_pathway"
run_omics_ablation "Drop_Proteomics"     "--use_genomics --use_epigenomics --use_transcriptomics --use_metabolomics --use_pathway"
run_omics_ablation "Drop_Metabolomics"   "--use_genomics --use_epigenomics --use_transcriptomics --use_proteomics --use_pathway"
run_omics_ablation "Drop_Pathway"        "--use_genomics --use_epigenomics --use_transcriptomics --use_proteomics --use_metabolomics"

echo "Omics ablation results saved to $ABLATION_OMICS_LOG"

# ============================================================================
echo "===================================================================="
echo "P4.6: 3-Omics vs 6-Omics Comparison (Best Config, 5-Fold)"
echo "Goal: Compare the impact of adding 3 extra omics modalities"
echo "===================================================================="

echo "1. 3-Omics Baseline"
python3 main.py $KFOLD_FIVE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_3

echo "2. 6-Omics Full Stack"
python3 main.py $KFOLD_FIVE \
    --graph_GNN_type heterogenous \
    --global_gnn_type SAGE \
    --use_graph_transformer \
    $CONTRASTIVE_ARGS \
    $DRUG_CONFIG \
    $OMICS_FLAGS_6

echo "Phase 4 Complete."