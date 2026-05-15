import os
import pandas as pd

models = ['GraphCDR', 'RedCDR', 'DeepTTC', 'GADRP', 'GraTransDRP', 'FUSECDR']
base_dir = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsStrictBenchmarking/results'
markdown_file = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/benchmark_results.md'

with open(markdown_file, "a") as f:
    f.write("\n## Unseen Both\n\n")
    
    for dataset in ['dataset-1', 'dataset-2']:
        f.write(f"### {dataset.title()}\n\n")
        f.write("| Model | AUC | AUPR |\n")
        f.write("|---|---|---|\n")
        
        for model in models:
            path = os.path.join(base_dir, 'unseen_both', dataset, model, 'fold_metrics.csv')
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    auc_mean = df['auc'].mean()
                    auc_std = df['auc'].std()
                    aupr_mean = df['aupr'].mean()
                    aupr_std = df['aupr'].std()
                    f.write(f"| {model} | {auc_mean:.4f} ± {auc_std:.4f} | {aupr_mean:.4f} ± {aupr_std:.4f} |\n")
                except Exception as e:
                    f.write(f"| {model} | N/A | N/A |\n")
            else:
                f.write(f"| {model} | N/A | N/A |\n")
        f.write("\n")

print("Appended Unseen Both results to benchmark_results.md")
