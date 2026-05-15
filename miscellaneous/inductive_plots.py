import matplotlib.pyplot as plt
import numpy as np

# Data from results_section.md L183-213
datasets = ["dataset-1", "dataset-2"]
metrics = ["AUC", "AUPR"]
protocols = ["Cell-line split", "Drug split", "Unseen both split"]
models = ["GraphCDR", "RedCDR", "FUSECDR"]

# Data Structure: [Dataset][Metric][Protocol][Model]
data = {
    "dataset-1": {
        "AUC": {
            "Cell-line split": [0.5848, 0.5412, 0.6768],
            "Drug split": [0.5961, 0.7024, 0.7165],
            "Unseen both split": [0.5730, 0.4900, 0.6444]
        },
        "AUPR": {
            "Cell-line split": [0.1388, 0.1304, 0.2423],
            "Drug split": [0.2274, 0.2475, 0.2599],
            "Unseen both split": [0.1349, 0.1142, 0.1847]
        },
        "AUC_std": {
            "Cell-line split": [0.0192, 0.0217, 0.0245],
            "Drug split": [0.0284, 0.0270, 0.0165],
            "Unseen both split": [0.0526, 0.0291, 0.0324]
        },
        "AUPR_std": {
            "Cell-line split": [0.0078, 0.0177, 0.0202],
            "Drug split": [0.0491, 0.0119, 0.0211],
            "Unseen both split": [0.0085, 0.0285, 0.0298]
        }
    },
    "dataset-2": {
        "AUC": {
            "Cell-line split": [0.7157, 0.8559, 0.8965],
            "Drug split": [0.5223, 0.5252, 0.6168],
            "Unseen both split": [0.5265, 0.5245, 0.5811]
        },
        "AUPR": {
            "Cell-line split": [0.5847, 0.7779, 0.8499],
            "Drug split": [0.4582, 0.4329, 0.5238],
            "Unseen both split": [0.4437, 0.4174, 0.5035]
        },
        "AUC_std": {
            "Cell-line split": [0.0200, 0.0114, 0.0041],
            "Drug split": [0.0565, 0.0456, 0.0183],
            "Unseen both split": [0.0479, 0.0493, 0.0173]
        },
        "AUPR_std": {
            "Cell-line split": [0.0324, 0.0276, 0.0094],
            "Drug split": [0.0745, 0.0757, 0.0492],
            "Unseen both split": [0.0381, 0.0612, 0.0597]
        }
    }
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
x = np.arange(len(protocols))
width = 0.25

for metric in metrics:
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model in enumerate(models):
            means = [data[ds][metric][p][i] for p in protocols]
            stds = [data[ds][metric + "_std"][p][i] for p in protocols]
            ax.bar(x + i*width, means, width, label=model, color=colors[i])

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} across Inductive Protocols ({ds})')
        ax.set_xticks(x + width)
        ax.set_xticklabels(protocols)
        ax.legend()
        # Updated y-axis scaling
        if ds == "dataset-1" and metric == "AUPR":
            ax.set_ylim(0, 0.4) # Reduced max value for dataset-1 AUPR visibility
        else:
            ax.set_ylim(0, 1.0)
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        filename = f"inductive_{ds}_{metric.lower()}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()