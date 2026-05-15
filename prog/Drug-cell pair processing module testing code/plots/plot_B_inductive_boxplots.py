import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os

# Configuration
RESULTS_FILE = "results_section.md"
OUTPUT_DIR = "prog/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define models and their colors
COLORS = {"GraphCDR": "#1f77b4", "RedCDR": "#ff7f0e", "FUSE-CDR": "#2ca02c"}
MODEL_MAP = {"FUSECDR": "FUSE-CDR", "RedCDR": "RedCDR", "GraphCDR": "GraphCDR"}

def parse_inductive_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    # Define sections to parse: B (Cell), C (Drug), D (Both)
    splits = {
        "Unseen Cells": r"### B\.Cell-line split(.*?)(?=### [C|D]|\Z)",
        "Unseen Drugs": r"### C\.Drug split(.*?)(?=### [D|E]|\Z)",
        "Unseen Both": r"### D\.Unseen both split(.*?)(?=## Model development|\Z)"
    }
    
    records = []
    
    for split_name, pattern in splits.items():
        section = re.search(pattern, content, re.DOTALL)
        if not section:
            print(f"Warning: Section {split_name} not found.")
            continue
        
        section_text = section.group(1)
        
        # Parse models within section
        # Format: ModelLabel:
        # dataset-X
        # fold,metrics...
        model_blocks = re.split(r"([a-zA-Z0-9]+):", section_text)
        
        for i in range(1, len(model_blocks), 2):
            raw_model_name = model_blocks[i].strip()
            model_data = model_blocks[i+1]
            model_name = MODEL_MAP.get(raw_model_name, raw_model_name)
            
            # Find datasets (dataset-1, dataset-2)
            dataset_blocks = re.split(r"(dataset-[12])", model_data)
            for j in range(1, len(dataset_blocks), 2):
                dataset_name = dataset_blocks[j].strip().capitalize().replace("Dataset-1", "Dataset-1").replace("Dataset-2", "Dataset-2")
                csv_data = dataset_blocks[j+1].strip()
                
                # Parse CSV rows
                for line in csv_data.split("\n"):
                    if line.startswith("fold") or not line.strip():
                        continue
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                        
                    # fold, val_auc, auc, aupr, f1, acc
                    records.append({
                        "Split": split_name,
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "Fold": int(parts[0]),
                        "AUC": float(parts[2]),
                        "AUPR": float(parts[3])
                    })
                    
    return pd.DataFrame(records)

def plot_inductive(df, dataset_name):
    subset = df[df["Dataset"] == dataset_name]
    if subset.empty:
        print(f"No data for {dataset_name}")
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    metrics = ["AUC", "AUPR"]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Boxplot
        sns.boxplot(
            data=subset, 
            x="Split", 
            y=metric, 
            hue="Model", 
            ax=ax, 
            palette=COLORS,
            showmeans=False,
            fliersize=0,
            linewidth=1.2,
            width=0.7
        )
        
        # Jittered Points
        sns.stripplot(
            data=subset, 
            x="Split", 
            y=metric, 
            hue="Model", 
            dodge=True, 
            ax=ax, 
            palette=COLORS,
            alpha=0.4, 
            size=4,
            jitter=0.2,
            edgecolor="gray",
            linewidth=0.5
        )
        
        # Axis labels and styling
        ax.set_title(f"{metric} Comparison ({dataset_name})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Testing Protocol (Inductive)", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Manage legend
        if i == 0:
            ax.legend(title="Model", loc="upper left", bbox_to_anchor=(0, 1))
        else:
            ax.get_legend().remove()
            
    plt.tight_layout()
    output_filename = f"{OUTPUT_DIR}/fig_B_inductive_{dataset_name.lower().replace('-', '')}.pdf"
    plt.savefig(output_filename)
    print(f"Saved: {output_filename}")
    plt.close()

if __name__ == "__main__":
    try:
        data = parse_inductive_data(RESULTS_FILE)
        if data.empty:
            print("Error: Parsed dataframe is empty. Check regex patterns.")
        else:
            plot_inductive(data, "Dataset-1")
            plot_inductive(data, "Dataset-2")
    except Exception as e:
        print(f"An error occurred: {e}")
