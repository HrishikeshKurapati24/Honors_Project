import json, os, glob
import pandas as pd

models = ['GraphCDR', 'RedCDR', 'DeepTTC', 'GADRP', 'GraphDRP', 'FUSECDR']
base_dir = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsStrictBenchmarking/results'

data = []
for split in ['unseen_cells', 'unseen_drugs', 'random', 'unseen_both']:
    for dataset in ['dataset-1', 'dataset-2']:
        for model in models:
            path = os.path.join(base_dir, split, dataset, model, 'fold_metrics.csv')
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    auc_mean = df['auc'].mean()
                    auc_std = df['auc'].std()
                    aupr_mean = df['aupr'].mean()
                    aupr_std = df['aupr'].std()
                    data.append({
                        'Split': split.replace('_', ' ').title(),
                        'Dataset': dataset.title(),
                        'Model': model,
                        'AUC_Mean': auc_mean,
                        'AUC': f'{auc_mean:.4f} ± {auc_std:.4f}',
                        'AUPR': f'{aupr_mean:.4f} ± {aupr_std:.4f}'
                    })
                except Exception as e:
                    pass

df = pd.DataFrame(data)

with open("/Volumes/Work/Semester - 6/Honors/CDRP models testing/benchmark_results.md", "w") as f:
    f.write("# 3OmicsStrictBenchmarking Results\n\n")
    
    # Define an order for the splits
    split_order = ['Random', 'Unseen Cells', 'Unseen Drugs', 'Unseen Both']
    
    for split in split_order:
        if split not in df['Split'].unique(): continue
        f.write(f"## {split}\n\n")
        
        split_df = df[df['Split'] == split]
        
        for dataset in ['Dataset-1', 'Dataset-2']:
            if dataset not in split_df['Dataset'].unique(): continue
            f.write(f"### {dataset}\n\n")
            
            table_df = split_df[split_df['Dataset'] == dataset].sort_values(by='AUC_Mean', ascending=False)
            
            # Markdown table formatting
            f.write("| Model | AUC | AUPR |\n")
            f.write("|---|---|---|\n")
            for _, row in table_df.iterrows():
                f.write(f"| {row['Model']} | {row['AUC']} | {row['AUPR']} |\n")
            f.write("\n")

print("Generated benchmark_results.md sorted by AUC")
