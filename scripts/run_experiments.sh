#!/bin/bash

# ==========================================
# GraphCDR Experiments Runner (Colab Version)
# ==========================================

# Change to script directory
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

echo "=========================================="
echo " RUNNING IN GOOGLE COLAB (No Conda) "
echo "=========================================="
echo "Using Python: $(which python3)"
echo ""

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ----------------------------
# Function to run an experiment
# ----------------------------
run_experiment() {
    local exp_num=$1
    local exp_name=$2
    local cmd=$3
    local log_file="logs/experiment_${exp_num}_${exp_name}_${TIMESTAMP}.log"
    
    echo "=========================================="
    echo "Experiment $exp_num: $exp_name"
    echo "=========================================="
    echo "Command: $cmd"
    echo "Log file: $log_file"
    echo "Starting at: $(date)"
    echo ""
    
    # Execute and save output to log
    eval $cmd 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    echo ""
    echo "Experiment $exp_num completed at: $(date)"
    echo "Exit code: $exit_code"
    echo "=========================================="
    echo ""
    
    if [ $exit_code -ne 0 ]; then
        echo "WARNING: Experiment $exp_num FAILED with exit code $exit_code"
        echo "Check log file: $log_file"
    fi
    
    sleep 2
}

# ----------------------------
# Default Parameters
# ----------------------------
EPOCHS=200
HIDDEN_CHANNELS=256
OUTPUT_CHANNELS=100
SINGLE_SEED=666
SEEDS="42 123 456 789 999"

# Base command
BASE_CMD="python3 ../src/graphCDR_node_representation_modified.py --epoch $EPOCHS --hidden_channels $HIDDEN_CHANNELS --output_channels $OUTPUT_CHANNELS"

# ----------------------------
# Run all experiments
# ----------------------------

echo "=========================================="
echo " STARTING ALL EXPERIMENTS "
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Total Experiments: 5"
echo ""

run_experiment 1 "GCLM_CDR_single_seed" \
    "$BASE_CMD --execution_architecture GCLM_CDR --single_seed $SINGLE_SEED"

run_experiment 2 "GCLM_CDR_multi_seed" \
    "$BASE_CMD --execution_architecture GCLM_CDR --multi_seed --seeds $SEEDS"

run_experiment 3 "GCLM_CDR_modified_chromatin_single_seed" \
    "$BASE_CMD --execution_architecture GCLM_CDR_modified --genomics_csv '../data/CCLE/Processed data/chromatin_profiling.csv' --single_seed $SINGLE_SEED"

run_experiment 4 "GCLM_CDR_modified_chromatin_multi_seed" \
    "$BASE_CMD --execution_architecture GCLM_CDR_modified --genomics_csv '../data/CCLE/Processed data/chromatin_profiling.csv' --multi_seed --seeds $SEEDS"

# ----------------------------
# Completion Summary
# ----------------------------
echo "=========================================="
echo " ALL EXPERIMENTS COMPLETED "
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "End Time: $(date)"
echo "Logs saved in: logs/"
echo "=========================================="

echo ""
echo "Generated log files:"
ls -lh logs/experiment_*_${TIMESTAMP}.log 2>/dev/null || echo "No log files found"