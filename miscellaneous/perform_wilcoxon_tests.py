import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import io

def perform_analysis():
    # 1. Load the data
    with open("per_fold_results.md", "r") as f:
        content = f.read()
    
    # Clean the content for pandas reading (fixed width)
    df = pd.read_csv(io.StringIO(content), sep='\s+')

    protocols = df['Protocol'].unique()
    datasets = df['Dataset'].unique()
    models = ["FUSECDR", "RedCDR", "GraphCDR"]
    metrics = ["AUC", "AUPR"]
    
    results = []

    for protocol in protocols:
        for dataset in datasets:
            subset = df[(df['Protocol'] == protocol) & (df['Dataset'] == dataset)]
            
            # Pairwise comparisons
            comparisons = [
                ("FUSECDR", "GraphCDR"),
                ("FUSECDR", "RedCDR"),
                ("RedCDR", "GraphCDR")
            ]
            
            for m1, m2 in comparisons:
                res_entry = {
                    "Protocol": protocol,
                    "Dataset": dataset,
                    "Comparison": f"{m1} vs {m2}"
                }
                
                for metric in metrics:
                    vals1 = subset[subset['Model'] == m1].sort_values('Fold')[metric].values
                    vals2 = subset[subset['Model'] == m2].sort_values('Fold')[metric].values
                    
                    if len(vals1) == 5 and len(vals2) == 5:
                        # Wilcoxon signed-rank test
                        # Using zero_method='pratt' or 'wilcox' - default is fine for 5 samples
                        try:
                            stat, p = wilcoxon(vals1, vals2)
                            res_entry[f"{metric}_p"] = p
                            res_entry[f"{metric}_sig"] = "*" if p < 0.05 else ""
                        except Exception as e:
                            res_entry[f"{metric}_p"] = np.nan
                            res_entry[f"{metric}_sig"] = "Error"
                    else:
                        res_entry[f"{metric}_p"] = np.nan
                        res_entry[f"{metric}_sig"] = "N/A"
                
                results.append(res_entry)

    results_df = pd.DataFrame(results)
    
    # Format output
    output_str = "# Wilcoxon Signed-Rank Test Results\n\n"
    output_str += "Tests performed on 5-fold cross-validation results. Results show p-values.\n"
    output_str += "(*) indicates significance at p < 0.05.\n\n"
    
    # Reorder columns for readability
    cols = ["Protocol", "Dataset", "Comparison", "AUC_p", "AUC_sig", "AUPR_p", "AUPR_sig"]
    output_str += results_df[cols].to_markdown(index=False)
    
    with open("wilcoxon_test_results.md", "w") as f:
        f.write(output_str)
    
    print("Wilcoxon tests completed. Results saved to wilcoxon_test_results.md")

if __name__ == "__main__":
    perform_analysis()
