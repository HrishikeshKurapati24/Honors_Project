import os
import pandas as pd

base_path = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsBenchmarking/results"
protocols = ["random", "unseen_cells", "unseen_drugs", "unseen_both"]
datasets = ["dataset-1", "dataset-2"]
models = ["GraphCDR", "RedCDR", "SOULCDR"]

results = []

for protocol in protocols:
    for dataset in datasets:
        for model in models:
            file_path = os.path.join(base_path, protocol, dataset, model, "fold_metrics.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    results.append({
                        "Protocol": protocol,
                        "Dataset": dataset,
                        "Model": model,
                        "Fold": int(row["fold"]),
                        "AUC": row["auc"],
                        "AUPR": row["aupr"]
                    })

output_df = pd.DataFrame(results)
print(output_df.to_string(index=False))