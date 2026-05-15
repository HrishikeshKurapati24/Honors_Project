import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuration
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
data = [
    {"GNN": "GIN", "Omics": "3-Omics", "AUC": 0.9223, "AUPR": 0.8940},
    {"GNN": "GCN", "Omics": "3-Omics", "AUC": 0.9228, "AUPR": 0.8950},
    {"GNN": "GraphSAGE", "Omics": "3-Omics", "AUC": 0.9220, "AUPR": 0.8946},
    {"GNN": "GAT", "Omics": "3-Omics", "AUC": 0.9192, "AUPR": 0.8890},
    {"GNN": "GIN", "Omics": "7-Modalities", "AUC": 0.9230, "AUPR": 0.8948},
    {"GNN": "GCN", "Omics": "7-Modalities", "AUC": 0.9227, "AUPR": 0.8945},
    {"GNN": "GraphSAGE", "Omics": "7-Modalities", "AUC": 0.9222, "AUPR": 0.8938},
    {"GNN": "GAT", "Omics": "7-Modalities", "AUC": 0.9209, "AUPR": 0.8913},
]

df = pd.DataFrame(data)

def plot_drug_encoder(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    metric = "AUC"
    
    # ✅ Filter only 7-Modalities
    df_7 = df[df["Omics"] == "7-Modalities"]
    
    # Single color (since no grouping now)
    color = "#2ca02c"
    
    # Bar plot (no hue)
    sns.barplot(
        data=df_7, 
        x="GNN", 
        y=metric, 
        ax=ax, 
        color=color
    )
    
    # Title and Labels
    ax.set_title(f"Drug Encoder Comparison ({metric})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Local Drug GNN Architecture", fontsize=12)
    ax.set_ylabel(metric)
    
    # Range adjustment
    ax.set_ylim(0.915, 0.926)
        
    # Add labels on bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.4f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 7), 
                        textcoords='offset points',
                        fontsize=9)
            
    # Highlight best performer
    ax.axhline(0.9230, color="red", linestyle="--", alpha=0.4)
    
    # ❌ Removed legend completely

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/fig_D1_drug_encoder_comparison.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_drug_encoder(df)