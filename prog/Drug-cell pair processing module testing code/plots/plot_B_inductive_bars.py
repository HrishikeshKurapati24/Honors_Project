import matplotlib.pyplot as plt
import matplotlib.container
import seaborn as sns
import pandas as pd
import re
import os

# Configuration
RESULTS_FILE = "results_section.md"
OUTPUT_DIR = "prog/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets renaming map
DATASET_MAP = {
    "dataset-1": "Benchmark Dataset",
    "dataset-2": "Custom Dataset"
}

# Colors for Models
MODEL_COLORS = {"GraphCDR": "#1f77b4", "RedCDR": "#ff7f0e", "FUSECDR": "#2ca02c"}

def parse_inductive_table(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    # Sections to parse
    sections = {
        "Unseen Cells": r"### B\.Cell-line split(.*?)(?=### C|\Z)",
        "Unseen Drugs": r"### C\.Drug split(.*?)(?=### D|\Z)",
        "Unseen Both": r"### D\.Unseen both split(.*?)(?=### E|\Z)"
    }
    
    records = []
    
    for split_name, pattern in sections.items():
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        
        section_text = match.group(1)
        # Find lines like: GraphCDR,dataset-1,0.6768 +/- 0.0245,0.2423 +/- 0.0202...
        # We need Model, Dataset, AUC_mean, AUC_std, AUPR_mean, AUPR_std
        lines = section_text.strip().split("\n")
        for line in lines:
            if "dataset" not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
                
            model = parts[0]
            raw_dataset = parts[1]
            dataset = DATASET_MAP.get(raw_dataset, raw_dataset)
            
            auc_part = parts[2].split("+/-")
            aupr_part = parts[3].split("+/-")
            
            records.append({
                "Split": split_name,
                "Model": model,
                "Dataset": dataset,
                "Metric": "AUC",
                "Mean": float(auc_part[0]),
                "Std": float(auc_part[1])
            })
            records.append({
                "Split": split_name,
                "Model": model,
                "Dataset": dataset,
                "Metric": "AUPR",
                "Mean": float(aupr_part[0]),
                "Std": float(aupr_part[1])
            })
            
    return pd.DataFrame(records)

def plot_split(df_split, split_name):
    # Unique datasets in this split
    datasets = df_split["Dataset"].unique()
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 6 * len(datasets)), sharex=False)
    if len(datasets) == 1:
        axes = [axes]
        
    for i, ds_name in enumerate(datasets):
        ax = axes[i]
        ds_data = df_split[df_split["Dataset"] == ds_name]
        
        # Grouped bar plot
        models = ds_data["Model"].unique()
        metrics = ds_data["Metric"].unique()
        
        n_groups = len(models)
        n_metrics = len(metrics)
        bar_width = 0.35
        index = range(n_groups)
        
        for j, metric in enumerate(metrics):
            m_data = ds_data[ds_data["Metric"] == metric]
            m_data = m_data.set_index("Model").loc[models].reset_index()
            
            colors = [MODEL_COLORS.get(m, "#ADB5BD") for m in m_data["Model"]]
            
            bars = ax.bar([p + j * bar_width for p in index], m_data["Mean"], bar_width, 
                   yerr=m_data["Std"], label=metric, color=colors,
                   capsize=5, alpha=0.9, edgecolor="black", linewidth=0.5)
            
            # Use hatches for AUPR to distinguish from AUC
            if metric == "AUPR":
                for bar in bars:
                    bar.set_hatch("///")
            
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(f"{split_name}: {ds_name}", fontsize=14, fontweight="bold")
        ax.set_xticks([p + bar_width/2 for p in index])
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(title="Metric")
        
        # Add labels on top of bars
        for container in ax.containers:
            # Skip error bars
            if isinstance(container, matplotlib.container.BarContainer):
                ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    sanitized_split = split_name.lower().replace(" ", "_")
    output_path = f"{OUTPUT_DIR}/fig_B_{sanitized_split}.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    df = parse_inductive_table(RESULTS_FILE)
    for split in df["Split"].unique():
        plot_split(df[df["Split"] == split], split)
