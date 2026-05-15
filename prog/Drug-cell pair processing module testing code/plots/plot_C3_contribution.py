import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuration
OUTPUT_DIR = "prog/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data from implementation plan (Verified against results_section.md Phase 4.5)
data = [
    {"Config": "All 6 Omics (Baseline)", "AUC": 0.9266, "Type": "Baseline"},
    {"Config": "Drop Genomics", "AUC": 0.9266, "Type": "Baseline"},
    {"Config": "Drop Epigenomics", "AUC": 0.9268, "Type": "Better"},
    {"Config": "Drop Transcriptomics", "AUC": 0.9271, "Type": "Better"},
    {"Config": "Drop Proteomics", "AUC": 0.9269, "Type": "Better"},
    {"Config": "Drop Metabolomics", "AUC": 0.9273, "Type": "Better"},
    {"Config": "Drop Pathway", "AUC": 0.9271, "Type": "Better"},
]

df = pd.DataFrame(data)

def plot_omics_contribution(df):
    plt.figure(figsize=(10, 6))
    
    # Define colors - Following FUSECDR Green schema
    palette = {"Baseline": "#ADB5BD", "Better": "#2ca02c"}
    
    # Horizontal Bar Plot
    ax = sns.barplot(
        data=df, 
        y="Config", 
        x="AUC", 
        hue="Type", 
        palette=palette,
        dodge=False
    )
    
    # Vertical line at baseline (0.9266)
    baseline_val = 0.9266
    plt.axvline(baseline_val, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"6-Omics Baseline: {baseline_val}")
    
    # X-axis zoom to emphasize differences
    plt.xlim(0.9250, 0.9280)
    
    # Title and labels
    plt.title("Omics Contribution Analysis (Drop-one-out Ablation)", fontsize=14, fontweight="bold")
    plt.xlabel("AUC Performance (Higher is Better)", fontsize=12)
    plt.ylabel("")
    
    # Add annotations on bars
    for i, p in enumerate(ax.patches):
        if p.get_width() > 0:
            ax.annotate(f"{p.get_width():.4f}", 
                        (p.get_width(), p.get_y() + p.get_height() / 2.), 
                        ha='left', va='center', 
                        xytext=(5, 0), 
                        textcoords='offset points',
                        fontsize=10, 
                        fontweight='bold' if p.get_width() > baseline_val else 'normal')

    plt.legend(title="Impact", loc="upper right")
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/fig_C3_omics_contribution.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_omics_contribution(df)
