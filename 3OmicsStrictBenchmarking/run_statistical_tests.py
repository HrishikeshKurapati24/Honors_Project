import os
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

models = ['GraphCDR', 'RedCDR', 'DeepTTC', 'GADRP', 'GraphDRP', 'FUSECDR']
base_dir = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsStrictBenchmarking/results'

# 1. Rank models per split and dataset
data = []
for split in ['unseen_cells', 'unseen_drugs', 'random', 'unseen_both']:
    for dataset in ['dataset-1', 'dataset-2']:
        model_means = []
        for model in models:
            path = os.path.join(base_dir, split, dataset, model, 'fold_metrics.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                auc_mean = df['auc'].mean()
                model_means.append((model, auc_mean, df))
        
        if len(model_means) >= 2:
            model_means.sort(key=lambda x: x[1], reverse=True)
            top1_model, _, df1 = model_means[0]
            top2_model, _, df2 = model_means[1]
            
            # Align by fold
            df1 = df1.sort_values('fold')
            df2 = df2.sort_values('fold')
            
            # AUC Tests
            auc1 = df1['auc'].values
            auc2 = df2['auc'].values
            try:
                auc_wilcoxon_p = wilcoxon(auc1, auc2).pvalue
            except Exception as e:
                auc_wilcoxon_p = float('nan')
            try:
                auc_ttest_p = ttest_rel(auc1, auc2).pvalue
            except Exception as e:
                auc_ttest_p = float('nan')

            # AUPR Tests
            aupr1 = df1['aupr'].values
            aupr2 = df2['aupr'].values
            try:
                aupr_wilcoxon_p = wilcoxon(aupr1, aupr2).pvalue
            except Exception as e:
                aupr_wilcoxon_p = float('nan')
            try:
                aupr_ttest_p = ttest_rel(aupr1, aupr2).pvalue
            except Exception as e:
                aupr_ttest_p = float('nan')
                
            data.append({
                'Split': split.replace('_', ' ').title(),
                'Dataset': dataset.title(),
                'Comparison': f"{top1_model} vs {top2_model}",
                'AUC Wilcoxon p-value': f"{auc_wilcoxon_p:.4g}" if not pd.isna(auc_wilcoxon_p) else "N/A",
                'AUC Paired t-test p-value': f"{auc_ttest_p:.4g}" if not pd.isna(auc_ttest_p) else "N/A",
                'AUPR Wilcoxon p-value': f"{aupr_wilcoxon_p:.4g}" if not pd.isna(aupr_wilcoxon_p) else "N/A",
                'AUPR Paired t-test p-value': f"{aupr_ttest_p:.4g}" if not pd.isna(aupr_ttest_p) else "N/A",
            })

results_df = pd.DataFrame(data)

markdown_file = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/benchmark_results.md'

with open(markdown_file, "a") as f:
    f.write("\n## Statistical Significance Tests\n\n")
    f.write("Tests performed on 5-fold cross-validation results comparing the 1st and 2nd ranked models (by mean AUC) per experiment type.\n\n")
    f.write(results_df.to_markdown(index=False))
    f.write("\n")

print("Statistical tests performed and appended to benchmark_results.md")
