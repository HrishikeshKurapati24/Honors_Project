# Wilcoxon Signed-Rank Test Results

Tests performed on 5-fold cross-validation results. Results show p-values.
(*) indicates significance at p < 0.05.

| Protocol     | Dataset   | Comparison          |   AUC_p | AUC_sig   |   AUPR_p | AUPR_sig   |
|:-------------|:----------|:--------------------|--------:|:----------|---------:|:-----------|
| random       | dataset-1 | FUSECDR vs GraphCDR |  1      |           |   0.625  |            |
| random       | dataset-1 | FUSECDR vs RedCDR   |  0.0625 |           |   0.0625 |            |
| random       | dataset-1 | RedCDR vs GraphCDR  |  0.3125 |           |   0.0625 |            |
| random       | dataset-2 | FUSECDR vs GraphCDR |  0.0625 |           |   0.0625 |            |
| random       | dataset-2 | FUSECDR vs RedCDR   |  0.125  |           |   0.125  |            |
| random       | dataset-2 | RedCDR vs GraphCDR  |  0.0625 |           |   0.0625 |            |
| unseen_cells | dataset-1 | FUSECDR vs GraphCDR |  0.0625 |           |   0.0625 |            |
| unseen_cells | dataset-1 | FUSECDR vs RedCDR   |  0.0625 |           |   0.0625 |            |
| unseen_cells | dataset-1 | RedCDR vs GraphCDR  |  0.0625 |           |   0.3125 |            |
| unseen_cells | dataset-2 | FUSECDR vs GraphCDR |  0.0625 |           |   0.0625 |            |
| unseen_cells | dataset-2 | FUSECDR vs RedCDR   |  0.0625 |           |   0.0625 |            |
| unseen_cells | dataset-2 | RedCDR vs GraphCDR  |  0.0625 |           |   0.0625 |            |
| unseen_drugs | dataset-1 | FUSECDR vs GraphCDR |  0.0625 |           |   0.3125 |            |
| unseen_drugs | dataset-1 | FUSECDR vs RedCDR   |  0.4375 |           |   0.625  |            |
| unseen_drugs | dataset-1 | RedCDR vs GraphCDR  |  0.0625 |           |   0.8125 |            |
| unseen_drugs | dataset-2 | FUSECDR vs GraphCDR |  0.0625 |           |   0.0625 |            |
| unseen_drugs | dataset-2 | FUSECDR vs RedCDR   |  0.0625 |           |   0.125  |            |
| unseen_drugs | dataset-2 | RedCDR vs GraphCDR  |  0.625  |           |   0.4375 |            |
| unseen_both  | dataset-1 | FUSECDR vs GraphCDR |  0.125  |           |   0.0625 |            |
| unseen_both  | dataset-1 | FUSECDR vs RedCDR   |  0.0625 |           |   0.0625 |            |
| unseen_both  | dataset-1 | RedCDR vs GraphCDR  |  0.0625 |           |   0.1875 |            |
| unseen_both  | dataset-2 | FUSECDR vs GraphCDR |  0.0625 |           |   0.125  |            |
| unseen_both  | dataset-2 | FUSECDR vs RedCDR   |  0.0625 |           |   0.0625 |            |
| unseen_both  | dataset-2 | RedCDR vs GraphCDR  |  0.8125 |           |   0.3125 |            |