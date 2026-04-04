import os
import pandas as pd

def analyze_prepared_datasets(base_dirs):
    results = []
    for base_dir in base_dirs:
        prepared_dir = os.path.join(base_dir, "prepared")
        if not os.path.exists(prepared_dir):
            continue
        
        for dataset_folder in sorted(os.listdir(prepared_dir)):
            dataset_path = os.path.join(prepared_dir, dataset_folder)
            if not os.path.isdir(dataset_path) or not dataset_folder.startswith("dataset"):
                continue
            
            # Analyze response pairs
            pairs_count = 0
            unique_cells = 0
            unique_drugs = 0
            pairs_file = os.path.join(dataset_path, "response_pairs.csv")
            if os.path.exists(pairs_file):
                df_pairs = pd.read_csv(pairs_file)
                pairs_count = len(df_pairs)
                if 'cell_id' in df_pairs.columns and 'drug_id' in df_pairs.columns:
                    unique_cells = df_pairs['cell_id'].nunique()
                    unique_drugs = df_pairs['drug_id'].nunique()
                    
            # Analyze feature files (omics and aux)
            feature_shapes = {}
            for fname in os.listdir(dataset_path):
                if fname.endswith(".csv") and fname != "response_pairs.csv" and fname != "similarity.csv":
                    try:
                        df = pd.read_csv(os.path.join(dataset_path, fname), index_col=0)
                        feature_shapes[fname.replace('.csv', '')] = df.shape
                    except Exception:
                        pass
            
            # Aux files (drug fingerprints or physicochemical)
            aux_dir = os.path.join(dataset_path, "aux")
            if os.path.exists(aux_dir):
                for fname in os.listdir(aux_dir):
                    if fname.endswith(".csv"):
                        try:
                            df = pd.read_csv(os.path.join(aux_dir, fname), index_col=0)
                            feature_shapes[f"aux/{fname.replace('.csv', '')}"] = df.shape
                        except Exception:
                            pass
                            
            results.append({
                "Benchmark": base_dir,
                "Dataset": dataset_folder,
                "Response Pairs": pairs_count,
                "Unique Cells": unique_cells,
                "Unique Drugs": unique_drugs,
                "Features": feature_shapes
            })
            
    return results

if __name__ == "__main__":
    benchmarks = ["3OmicsBenchmarking", "ExpressionBenchmarking", "PathwayBenchmarking"]
    
    print(f"{'Benchmark':<25} | {'Dataset':<10} | {'Pairs':<8} | {'Cells':<6} | {'Drugs':<6} | {'Features (Rows x Cols)'}")
    print("-" * 140)
    
    results = analyze_prepared_datasets(benchmarks)
    for res in results:
        features_str = ", ".join([f"{k}: {v[0]}x{v[1]}" for k, v in res['Features'].items()])
        print(f"{res['Benchmark']:<25} | {res['Dataset']:<10} | {res['Response Pairs']:<8} | {res['Unique Cells']:<6} | {res['Unique Drugs']:<6} | {features_str}")
