#!/bin/bash
# --------------------------------------------------------------------------------
# PHASE 2 — Graph/Global GNN & Transformer Variations
# 2-Fold Cross Validation
# --------------------------------------------------------------------------------

KFOLD="--k_fold 2"

OMICS_FLAGS="--use_genomics --use_epigenomics --use_transcriptomics"

DRUG_CONFIG="--drug_gnn_type GIN"

echo "===================================================================="
echo "P2: Graph/Global GNN & Transformer Variations"
echo "===================================================================="

# Homogeneous Graphs
for global in "GCN" "SAGE"; do
    echo "Homogenous + $global + GraphTrans=ON"
    python3 main.py $KFOLD \
        --graph_GNN_type homogenous \
        --global_gnn_type $global \
        $DRUG_CONFIG \
        --use_graph_transformer \
        $OMICS_FLAGS
done


# Heterogeneous Graphs
for global in "GAT" "SAGE"; do
    for g_trans in "" "--use_graph_transformer"; do
        echo "Heterogenous + $global + GraphTrans=${g_trans:-OFF}"
        python3 main.py $KFOLD \
            --graph_GNN_type heterogenous \
            --global_gnn_type $global \
            $DRUG_CONFIG \
            $g_trans \
            $OMICS_FLAGS
    done
done

echo "P4 Complete."