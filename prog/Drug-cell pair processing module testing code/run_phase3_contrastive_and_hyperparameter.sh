#!/bin/bash
# --------------------------------------------------------------------------------
# PHASE 3 — Contrastive Learning Experiments
# 2-Fold Cross Validation
# --------------------------------------------------------------------------------

KFOLD="--k_fold 2"

OMICS_FLAGS="--use_genomics --use_epigenomics --use_transcriptomics"

DRUG_CONFIG="--drug_gnn_type GIN"

# echo "===================================================================="
# echo "P3.1: Contrastive Learning Grid Search"
# echo "===================================================================="

# WEIGHTS="0.001 0.005 0.01 0.05"
# TEMPS="0.005 0.01 0.05 0.1"

BEST_AUC="0"
BEST_W="0.001"
BEST_T="0.01"

# for w in $WEIGHTS; do
#     for t in $TEMPS; do

#         echo "Grid Search: Weight=$w Temp=$t"

#         OUTPUT_FILE="temp_grid_output.txt"

#         python3 main.py $KFOLD \
#             --graph_GNN_type heterogenous \
#             --global_gnn_type SAGE \
#             --use_graph_transformer \
#             --use_contrastive \
#             --contrastive_weight $w \
#             --temperature $t \
#             --warmup_epochs 10 \
#             --hyperparameter_tuning \
#             $DRUG_CONFIG \
#             $OMICS_FLAGS | tee "$OUTPUT_FILE"

#         CURRENT_AUC=$(grep "val auc:" "$OUTPUT_FILE" | awk '{print $3}' | sort -nr | head -n1)

#         if [ -n "$CURRENT_AUC" ]; then
#             IS_BETTER=$(awk -v curr="$CURRENT_AUC" -v max="$BEST_AUC" 'BEGIN {print (curr > max ? 1 : 0)}')
#             if [ "$IS_BETTER" -eq 1 ]; then
#                 BEST_AUC=$CURRENT_AUC
#                 BEST_W=$w
#                 BEST_T=$t
#                 echo ">>> NEW BEST: AUC=$BEST_AUC | W=$BEST_W | T=$BEST_T"
#             fi
#         fi

#         rm "$OUTPUT_FILE"

#     done
# done

# echo "Best Contrastive Params: W=$BEST_W T=$BEST_T"

echo "===================================================================="
echo "P3.2: Hyperparameter Optimization (LR, Hidden, Output, Fusion)"
echo "Using best CL params: W=$BEST_W T=$BEST_T"
echo "Baseline: Hetero + SAGE + GT=ON + CL"
echo "===================================================================="

# Grid values
LRS="0.001 0.0005 0.0001"
HIDDEN="128 256 512"
OUTPUT="64 256"
FUSION="128 256 512"

BEST_AUC_HP="0"
BEST_LR=""
BEST_H=""
BEST_O=""
BEST_F=""

# Continue from Tuning: LR=0.0005 | Hidden=128 | Output=64 | Fusion=256
# ------------------------------------------------------------------
# RESUME CONTROL
# ------------------------------------------------------------------
START_LR="0.0005"
START_H="128"
START_O="64"
START_F="256"

START_FOUND=false
# ------------------------------------------------------------------

# Log file for summary
SUMMARY_LOG="logs/phase3_hp_summary.txt"
mkdir -p logs
echo "LR,Hidden,Output,Fusion,AUC" > "$SUMMARY_LOG"

for lr in $LRS; do
    for h in $HIDDEN; do
        for o in $OUTPUT; do
            for f in $FUSION; do

                # Skip until we reach the resume point
                if [ "$START_FOUND" = false ]; then
                    if [ "$lr" = "$START_LR" ] && \
                       [ "$h" = "$START_H" ] && \
                       [ "$o" = "$START_O" ] && \
                       [ "$f" = "$START_F" ]; then
                        START_FOUND=true
                    else
                        continue
                    fi
                fi
                
                echo "--------------------------------------------------------"
                echo "Tuning: LR=$lr | Hidden=$h | Output=$o | Fusion=$f"
                echo "--------------------------------------------------------"

                TEMP_OUT="logs/temp_phase3_hp_output.txt"

                python3 main.py $KFOLD \
                    --graph_GNN_type heterogenous \
                    --global_gnn_type SAGE \
                    --use_graph_transformer \
                    --lr $lr \
                    --hidden_channels $h \
                    --output_channels $o \
                    --fusion_dim $f \
                    --hyperparameter_tuning \
                    --use_contrastive --contrastive_weight $BEST_W --temperature $BEST_T --warmup_epochs 10 \
                    $DRUG_CONFIG \
                    $OMICS_FLAGS | tee "$TEMP_OUT"

                CURRENT_AUC=$(grep "val auc:" "$TEMP_OUT" | awk '{print $3}' | sort -nr | head -n1)

                if [ -n "$CURRENT_AUC" ]; then
                    echo "$lr,$h,$o,$f,$CURRENT_AUC" >> "$SUMMARY_LOG"
                    IS_BETTER=$(awk -v curr="$CURRENT_AUC" -v max="$BEST_AUC_HP" 'BEGIN {print (curr > max ? 1 : 0)}')
                    if [ "$IS_BETTER" -eq 1 ]; then
                        BEST_AUC_HP=$CURRENT_AUC
                        BEST_LR=$lr
                        BEST_H=$h
                        BEST_O=$o
                        BEST_F=$f
                        echo ">>> NEW BEST: AUC=$BEST_AUC_HP | LR=$BEST_LR | H=$BEST_H | O=$BEST_O | F=$BEST_F"
                    fi
                fi

                rm "$TEMP_OUT"
            done
        done
    done
done

echo "===================================================================="
echo "P3.1 + P3.2 Complete."
echo "Best Contrastive Params: W=$BEST_W T=$BEST_T"
echo "Best Hyperparameters:    LR=$BEST_LR | Hidden=$BEST_H | Output=$BEST_O | Fusion=$BEST_F"
echo "Top HP AUC: $BEST_AUC_HP"
echo "HP Results logged to $SUMMARY_LOG"
echo "===================================================================="