import pandas as pd
import os

markdown_file = '/Volumes/Work/Semester - 6/Honors/CDRP models testing/benchmark_results.md'
output = []
for d in ['dataset-1', 'dataset-2']:
    path = f'/Volumes/Work/Semester - 6/Honors/CDRP models testing/3OmicsStrictBenchmarking/prepared/{d}/response_pairs.csv'
    df = pd.read_csv(path)
    vc = df['label'].value_counts()
    pos = vc.get(1, 0)
    neg = vc.get(0, 0)
    total = pos + neg
    ratio = pos / total if total > 0 else 0
    output.append(f"- **{d.title()} Skewness:** {pos} positive pairs ({ratio*100:.2f}%), {neg} negative pairs ({(1-ratio)*100:.2f}%)")

with open(markdown_file, 'r') as f:
    content = f.read()

with open(markdown_file, 'w') as f:
    f.write("# 3OmicsStrictBenchmarking Results\n\n")
    f.write("## Dataset Class Skewness (Response Pairs)\n\n")
    for o in output:
        f.write(o + "\n")
    f.write("\n")
    
    # Write the rest of the file without the original header
    content = content.replace("# 3OmicsStrictBenchmarking Results\n\n", "", 1)
    f.write(content)

print("Prepended skewness to markdown.")
