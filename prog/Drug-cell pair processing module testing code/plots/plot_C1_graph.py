import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuration
OUTPUT_DIR = "prog/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data from implementation plan (Verified against results_section.md)
data = [
    {"Config": "Homo+GCN+HGT", "Graph": "Homo", "AUC": 0.9157, "AUPR": 0.8831},
    {"Config": "Homo+SAGE+HGT", "Graph": "Homo", "AUC": 0.9176, "AUPR": 0.8856},
    {"Config": "Hetero+GAT+NoGT", "Graph": "Hetero+NoHGT", "AUC": 0.8528, "AUPR": 0.7958},
    {"Config": "Hetero+GAT+HGT", "Graph": "Hetero+HGT", "AUC": 0.9226, "AUPR": 0.8949},
    {"Config": "Hetero+SAGE+NoHGT", "Graph": "Hetero+NoHGT", "AUC": 0.9206, "AUPR": 0.8905},
    {"Config": "Hetero+SAGE+HGT", "Graph": "Hetero+HGT", "AUC": 0.9229, "AUPR": 0.8949},
]

df = pd.DataFrame(data)

def plot_graph_ablation(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    metric = "AUC"
    
    # Color mapping - Following FUSECDR Green schema
    palette = {
        "Homo": "#ADB5BD", 
        "Hetero+NoHGT": "#91cf60", 
        "Hetero+HGT": "#2ca02c"
    }
    
    # Bar plot
    sns.barplot(
        data=df, 
        x="Config", 
        y=metric, 
        hue="Graph", 
        ax=ax, 
        palette=palette,
        dodge=False
    )
    
    # Highlight best bar (Hetero+SAGE+GT)
    best_idx = df[df["Config"] == "Hetero+SAGE+HGT"].index[0]
    ax.patches[best_idx].set_edgecolor("gold")
    ax.patches[best_idx].set_linewidth(2)
    
    # Reference line for best value
    best_val = df[metric].max()
    ax.axhline(best_val, color="red", linestyle="--", alpha=0.6, label=f"Max: {best_val:.4f}")
    
    # Labels and annotations
    ax.set_title(f"Impact of Graph Design on {metric}", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Config"], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0.75, 0.95)
    
    # Add labels on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.4f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=10)
    
    ax.legend(title="Graph Type", loc="lower right")

    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/fig_C1_graph_construction.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_graph_ablation(df)
