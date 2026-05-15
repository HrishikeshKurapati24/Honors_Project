# Section V – Experimental Setup (Filled Details)

---

## A. Datasets

### 1) Custom Dataset (7-Omics — `final_dataset` used during 'Model Development and ablation analysis')

**Source:** Cancer Cell Line Encyclopedia (CCLE) for omics data; GDSC for drug response labels and molecular drug features.

**Description:** A custom multi-omics drug–cell interaction dataset constructed in this work for ablation and flexibility analysis. Drug response labels (binary: sensitive = 1, resistant = −1) are derived from GDSC IC50 measurements and transformed using a drug-specific Z-score threshold. Drug molecular graph features were extracted from SMILES strings using DeepChem's `ConvMolFeaturizer`.

**Modalities (cell-line features — all 7):**

| Modality | Category | Feature Dimension |
|---|---|---|
| Somatic mutation | Genomics | 34,673 |
| Chromatin accessibility | Epigenomics | 40 |
| DNA methylation (CpG) | Epigenomics | 1,355 |
| Gene expression (RNA-seq) | Transcriptomics | 32,361 |
| Reverse-phase protein array (RPPA) | Proteomics | 214 |
| Metabolomics profiling | Metabolomics | 225 |
| PROGENy pathway activity scores | Pathway | 50 |

Cell–cell similarity is constructed from the PROGENy pathway activity vectors (50 pathways). Drug–drug similarity is constructed from physicochemical property vectors (63 features, sourced from PubChem).

**Preprocessing:**
Several preprocessing steps were applied to ensure data quality and consistency across modalities.

First, missing values were handled using a combination of filtering and imputation strategies. Columns and rows containing more than 10\% missing values were removed from the dataset. Remaining missing values were imputed using K-Nearest Neighbor (KNN) imputation with $k = 5$.

Second, low-variance features were removed to eliminate uninformative variables. A VarianceThreshold filter with threshold 0.03 was applied to the omics matrices.

The PROGENy pathway activity scores used for cell–cell similarity construction were standardized during the pathway activity computation step.

After preprocessing, a strict intersection across all seven omics modalities, the drug graph feature library, and the physicochemical property table was computed. Only cell lines and drugs present in all modalities were retained.

**Final Statistics (7-omics):**
- **Drugs:** 298
- **Cell lines:** 425
- **Drug–cell response pairs:** 109,446
  - Sensitive (label = 1): 44,582
  - Resistant (label = −1): 64,864

**Usage:** Ablation studies, flexibility and modality experiments, partial benchmarking.

---

### 2) Custom Dataset (3-Omics — `dataset-2` used during '3OmicsBenchmarking evaluation')

**Source:** CCLE / GDSC (same pipeline as above, restricted to 3 omics modalities).

**Modalities:**

| Modality | Category | Feature Dimension |
|---|---|---|
| Somatic mutation | Genomics | 34,673 |
| Gene expression (RNA-seq) | Transcriptomics | 32,361 |
| DNA methylation (CpG) | Epigenomics | 1,355 |

**Final Statistics (3-omics custom):**
- **Drugs:** 298
- **Cell lines:** 456
- **Drug–cell response pairs:** 117,330
  - Sensitive (label = 1): 47,994
  - Resistant (label = −1): 69,336

**Usage:** Fair 3-omics comparison with baseline models (GraphCDR, DeepCDR, RedCDR).

---

### 3) Benchmark Dataset — GraphCDR Dataset (`dataset-1` used during '3OmicsBenchmarking evaluation')

**Source:** GraphCDR official repository. Cell-line omics from GDSC/CCLE; drug response from GDSC IC50 values; drug molecular features from PubChem SMILES strings.

**Description:** The standard benchmark dataset used in prior work (GraphCDR, DeepCDR, RedCDR) for fair comparison.

**Modalities:**

| Modality | Category | Feature Dimension |
|---|---|---|
| Somatic mutation | Genomics | 34,673 |
| Gene expression | Transcriptomics | 697 |
| DNA methylation | Epigenomics | 808 |
| PROGENy pathway activity (similarity) | — | 14 pathways |

**Preprocessing (and why pair count is reduced):**
The GraphCDR benchmark dataset undergoes several filtering steps that reduce the total number of response pairs from the raw GDSC IC50 matrix:

1. **Drug intersection with graph features:** Only drugs for which molecular graph features (atom/adjacency graph) could be successfully extracted from SMILES strings are retained. This limits the drug universe from 266 GDSC drug entries to **222 drugs**.
2. **IC50 thresholding to binary labels:** Each drug has a pre-computed IC50 threshold (from `drug_threshold.csv`). Raw IC50 values are binarized: $\text{label} = \mathbb{1}[\text{ln-IC50} < \text{threshold}]$. NaN IC50 entries are excluded entirely.
3. **Multi-omics cell-line intersection:** Only cell lines present in all three omics matrices (mutation: 961 cells, expression: 561 cells, methylation: 561 cells) are retained. The binding constraint is expression and methylation with **561 cell lines**.
4. **Physicochemical feature intersection:** Only drugs present in the physicochemical property table are retained.

After all filtering steps, the final prepared benchmark contains:

**Final Statistics (GraphCDR benchmark):**
- **Drugs:** 222
- **Cell lines:** 561
- **Drug–cell response pairs:** 99,147
  - Sensitive (label = 1): 10,888 (~11.0%)
  - Resistant (label = 0): 88,259 (~89.0%)

> **Note:** The strong class imbalance (≈11% sensitive) makes AUPR a particularly important metric for this benchmark.

> **Note on pair-count discrepancy vs. GraphCDR paper:** The GraphCDR paper reports 100,572 response pairs (11,591 sensitive, 88,981 resistant), whereas our prepared benchmark yields 99,147 pairs (10,888 sensitive, 88,259 resistant) — a difference of ~1.4%. This arises from two sources: (i) our pipeline does not apply GraphCDR's implicit cell-line cancer-type annotation filter (`Cell_lines_annotations.txt`), which silently drops cell lines missing a TCGA label; and (ii) we resolve ambiguous (cell, drug) response pairs using majority voting, whereas GraphCDR's deduplication strategy sorts descending by label and keeps the first occurrence, giving precedence to the sensitive class in ties. These differences are negligible and do not affect the validity of benchmark comparisons.

**Usage:** Benchmark comparisons with GraphCDR, RedCDR.

---

## B. Data Split Settings(During benchmark evaluation)(NOTE: For internal(Model Development and ablation analysis) evaluation, only random data split was used, by default, it ran 2-fold cross-validation, except the final phase of evaluation, which used 5-fold cross-validation)

All response pairs are pre-processed via `canonicalize_response_pairs` (deduplication by majority label, sorted by cell/drug ID) before any split is applied. All protocols use 5-fold cross-validation with seed 0, a global validation ratio of 10% of the full dataset, and are persisted to disk as `train.csv`, `val.csv`, `test.csv` per fold alongside an `entities.json` recording the exact cell/drug ID sets per split. Disjointness of entity sets is hard-validated at generation time — any entity-set overlap raises a `ValueError` before training begins.

Hyperparameters (learning rate, dropout, etc.) are tuned exclusively on the **random split** using a held-out validation fold. For all inductive protocols, those tuned hyperparameters are **reused directly** without re-tuning, ensuring no information leakage from the inductive test partition.

---

### 1. Random Split

All response pairs are partitioned at the **pair level** using `sklearn.KFold` with shuffling. For each of the 5 folds:

- **Test set:** $\sim$20% of all pairs (one KFold partition).
- **Validation set:** 10% of all pairs drawn from the remaining 80%, sampled with a fixed random permutation.
- **Training set:** the remaining $\sim$70%.

Cell lines and drugs can appear in all three splits. This is the standard transductive setting used for primary benchmarking.

---

### 2. Unseen Cell Lines (Inductive — Cells)

Splits are performed at the **cell-line entity level** using `sklearn.KFold` over the sorted unique set of cell IDs:

- Test cell lines: $\sim$20% of all cell lines (one KFold partition).
- Validation cell lines: 10% of all cell lines, sampled from the remaining 80%.
- Training cell lines: the remaining $\sim$70%.

All drugs are available in all three splits. Response pairs are then derived by filtering to the respective cell-line scopes.

**Strict inductive testing:** For the inductive protocols, three separate model scopes are constructed at runtime (in `fusecdr_runner.py`):
- **Training scope:** omics features, graph features, and physicochemical features loaded exclusively for `{train_cells} × {train_drugs}`. The graph edges (cell–cell similarity, drug–drug similarity, drug→cell response) are built only over training entities.
- **Validation scope:** loaded for `{train_cells ∪ val_cells} × {train_drugs ∪ val_drugs}` with the combined pair table `train_pairs ∪ val_pairs`. Validation cell lines have their omics features included but their response pairs are only evaluated — not used as graph edges.
- **Test scope:** loaded for `{train_cells ∪ test_cells} × {train_drugs ∪ test_drugs}` with the combined pair table `train_pairs ∪ test_pairs`. Test cell lines are included in the graph solely as isolated nodes with their omics-derived initialisation — they have **zero training response edges** connecting them.

This ensures that during test-time inference, the model must generalise drug response predictions to cell lines (or drugs) that were entirely absent from the training graph topology.

---

### 3. Unseen Drugs (Inductive — Drugs)

Splits are performed at the **drug entity level** using the same `_entity_split` procedure over the sorted unique set of drug IDs. All cell lines are available in all three splits; response pairs are filtered to the respective drug scopes.

The same three-scope inductive construction applies: training, validation, and test drug sets are strictly disjoint. Test drugs appear as isolated nodes in the graph with only their GIN-encoded molecular graph features; no response edges connect them to cell lines during training.

---

### 4. Unseen Both (Inductive — Cells and Drugs)

Independent `_entity_split` calls are made over both cell IDs and drug IDs separately with the same seed, then zipped fold-by-fold. This yields:

- Test set: response pairs where **both** the cell line and the drug are entirely unseen during training.
- Validation set: pairs where both the validation cell subset and validation drug subset co-occur.
- Training set: pairs from the train cell × train drug Cartesian product only.

This is the most challenging and clinically realistic setting, requiring the model to generalise across both biological (cell-line) and chemical (drug) axes simultaneously.

> **Disjointness guarantee:** After each split is created, `_validate_entities` explicitly asserts zero overlap between `train_cells`/`val_cells`/`test_cells` (for cell-inductive protocols) and `train_drugs`/`val_drugs`/`test_drugs` (for drug-inductive protocols), raising a `ValueError` immediately if any leakage is detected.

---

## C. Baseline Models

- **GraphCDR:** Heterogeneous graph-based model integrating multi-omics cell-line features and molecular drug graphs. Uses GNN-based drug encoding and graph convolutional layers over a drug–cell interaction graph.
- **RedCDR:** Dual-branch architecture that separates attribute-level learning from interaction-level modeling using representation disentanglement.

All baselines are evaluated on the same prepared dataset with consistent preprocessing wherever possible.

---

## D. Implementation Details

All models were implemented in PyTorch with PyTorch Geometric and trained on a Google Colab environment with an NVIDIA Tesla T4 GPU (12 GB RAM), Intel Xeon CPU (~2.20 GHz), and CUDA-enabled acceleration.

### Hyperparameter selection details

#### Internal(Model Development and ablation analysis) Evaluation (FUSE-CDR on Custom 3-Omics Dataset — `dataset-2`)

Internal hyperparameter tuning for FUSE-CDR was conducted in two sequential phases using **2-fold cross-validation** (except the final Phase 4 robustness runs, which used 5-fold CV). Selection was based on **validation AUC** throughout.

**Phase 3.1 — Contrastive Learning Grid Search:**

A 4×4 grid search was run over contrastive weight and temperature:
- `contrastive_weight` ∈ {0.001, 0.005, 0.010, 0.050}
- `temperature` ∈ {0.005, 0.010, 0.050, 0.100}

All 16 combinations were evaluated using the best graph configuration identified in Phase 2 (heterogeneous graph + SAGE global GNN + graph transformer ON). The best result from the full grid:

| contrastive\_weight | temperature | Final AUC |
|---|---|---|
| **0.001** | **0.010** | **0.9239** ← selected |
| 0.010 | 0.005 | 0.9237 |
| 0.001 | 0.005 | 0.9234 |
| 0.050 | 0.010 | 0.9236 |

Selected internal CL configuration: `contrastive_weight = 0.001`, `temperature = 0.01`.

> **Note:** The benchmarking configuration uses `contrastive_weight = 0.005` and `temperature = 0.05` following the final benchmarking tuning on dataset-1. The internal Phase 3.1 values (0.001, 0.01) were identified during model development on the custom dataset before the external tuning was done.

**Phase 3.2 — Hyperparameter Tuning (LR × Hidden × Output × Fusion):**

A full grid search was conducted with the selected Phase 3.1 CL settings:
- `lr` ∈ {0.001, 0.0005, 0.0001}
- `hidden_channels` ∈ {128, 256, 512}
- `output_channels` ∈ {64, 256}
- `fusion_dim` ∈ {128, 256, 512}

25 combinations were recorded in `logs/phase3_hp_summary.txt`. Top 10 settings by validation AUC:

| lr | hidden | output | fusion | Best Val AUC |
|---|---|---|---|---|
| **0.001** | **256** | **64** | **512** | **0.9258** ← selected |
| 0.001 | 256 | 256 | 512 | 0.9255 |
| 0.001 | 256 | 256 | 128 | 0.9252 |
| 0.001 | 256 | 64 | 256 | 0.9250 |
| 0.001 | 128 | 256 | 128 | 0.9249 |
| 0.001 | 256 | 256 | 256 | 0.9247 |
| 0.001 | 512 | 64 | 128 | 0.9247 |
| 0.001 | 128 | 256 | 256 | 0.9244 |
| 0.001 | 256 | 64 | 128 | 0.9240 |
| 0.001 | 128 | 256 | 512 | 0.9230 |

**Selected internal configuration** (adopted as model defaults in `main.py`):

| Parameter | Value |
|---|---|
| contrastive\_weight | 0.001 |
| temperature | 0.01 |
| lr | 0.001 |
| hidden\_channels | 256 |
| output\_channels | 64 |
| fusion\_dim | 512 |

Corresponding 2-fold metrics for this configuration (from raw log `20260316_020223.log`):
AUC = 0.9235, AUPR = 0.8961, F1 = 0.8151, ACC = 0.8419.

During evaluation, the classification threshold was not fixed at 0.5. Instead, the threshold was dynamically selected by maximizing the F1 score across candidate threshold values.

#### Benchmarking Evaluation (FUSE-CDR, GraphCDR, RedCDR on dataset-1 and dataset-2)

Hyperparameter selection follows a **two-stage successive halving** strategy on the random split only (seed 0, 5-fold CV). In Stage 1, all candidates are evaluated on fold 1 with a small epoch budget; the top 50% of candidates by validation AUC are promoted. In Stage 2, survivors are evaluated on folds 1–2 with a larger epoch budget; the single best candidate is selected. For inductive protocols, the configuration tuned on the random split is **reused directly** without further tuning.

**Early stopping:** In all models, the best model checkpoint is selected as the epoch achieving the highest validation AUC, and test evaluation uses that checkpoint.

---

### FUSE-CDR

**Fixed architecture parameters (all runs):**

| Parameter | Value |
|---|---|
| Dropout | 0.2 |
| Graph conv layers (local + global) | 2 |
| Attention heads (HGT) | 4 |
| GIN layers (drug branch) | 3 |
| Drug batch size | Full batch |
| Top-$k$ graph neighbours | 10 |
| Optimizer | Adam, weight decay $10^{-5}$ |
| Temperature $\tau$ (SCL) | 0.05 |
| Contrastive weight $\lambda$ | 0.005 |
| Warmup epochs (SCL) | 10 |
| Max contrastive pairs | 2,048 |
| Epochs (benchmarking) | 400 |
| $k$-fold CV | 5, seed 0 |

**Tuning search space** (8 candidates, tuned only on dataset-1):

| lr | hidden\_channels | output\_channels | fusion\_channels |
|---|---|---|---|
| 1e-3 | 256 | 64 | 512 |
| 1e-3 | 256 | 256 | 512 |
| 1e-3 | 256 | 256 | 128 |
| 1e-3 | 256 | 64 | 256 |
| 1e-3 | 128 | 256 | 128 |
| 1e-3 | 512 | 64 | 128 |
| 1e-3 | 256 | 256 | 256 |
| 1e-3 | 128 | 256 | 256 |

Tuning budget: Stage 1 — 40 epochs (fold 1); Stage 2 — 80 epochs (folds 1–2). Selection metric: validation AUC.

**Best configuration — dataset-1 (GraphCDR benchmark):**

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| Hidden dimension ($d_h$) | 256 |
| Output dimension (predictor) | 128 |
| Fusion dimension (cell module) | 256 |
| Val AUC at selection | 0.8029 |

**Best configuration — dataset-2 (Custom 3-omics):** Hyperparameter tuning was not re-run for dataset-2; the **default configuration** (`hidden_channels=256`, `output_channels=64`, `fusion_channels=512`, `lr=0.001`) was used directly, as this is the native dataset and it was fine-tuned on this dataset during internal evaluation.

**Contrastive Learning Settings:**

| Parameter | Value |
|---|---|
| Temperature $\tau$ | 0.05 |
| Contrastive weight $\lambda$ | 0.005 |
| Warmup epochs before CL activation | 10 |
| Max contrastive pairs per step | 2048 |

During evaluation, the classification threshold was not fixed at 0.5. Instead, the threshold was dynamically selected by maximizing the F1 score across candidate threshold values.

---

### GraphCDR

GraphCDR uses a Deep Graph Infomax (DGI) auxiliary loss in addition to the binary cross-entropy prediction loss. The total loss is:
$$\mathcal{L} = (1 - \alpha - \beta)\,\mathcal{L}_{\text{BCE}} + \alpha\,\mathcal{L}_{\text{DGI}}^{+} + \beta\,\mathcal{L}_{\text{DGI}}^{-}$$
where $\alpha$ and $\beta$ control the weight of positive and negative DGI terms respectively.

**Fixed parameters (all runs):**

| Parameter | Value |
|---|---|
| Hidden dimension | 256 |
| Output/embedding dimension | 100 |
| Learning rate | 0.001 |
| Weight decay | 0.0 |
| Optimizer | Adam |
| Epochs | 350 |
| $k$-fold CV | 5, seed 0 |

**Tuning search space** (6 candidates, $\alpha$ and $\beta$ only):

| $\alpha$ | $\beta$ |
|---|---|
| 0.2 | 0.2 |
| 0.2 | 0.3 |
| 0.3 | 0.2 |
| 0.3 | 0.3 |
| 0.3 | 0.4 |
| 0.4 | 0.3 |

Tuning budget: Stage 1 — 60 epochs (fold 1); Stage 2 — 120 epochs (folds 1–2). Selection metric: validation AUC.

**Best configuration — dataset-1 (GraphCDR benchmark):**

| Parameter | Value |
|---|---|
| $\alpha$ | 0.2 |
| $\beta$ | 0.2 |
| Val AUC at selection | 0.8115 |

**Best configuration — dataset-2 (Custom 3-omics):**

| Parameter | Value |
|---|---|
| $\alpha$ | 0.2 |
| $\beta$ | 0.3 |
| Val AUC at selection | 0.8935 |

---

### RedCDR

RedCDR combines BCE prediction loss with a representation disentanglement (RD) loss and a pair-level distribution alignment (PD) loss:
$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + r_d \cdot \mathcal{L}_{\text{RD}} + w_{\text{pd}} \cdot \mathcal{L}_{\text{PD}}$$

For inductive protocols, RedCDR transfers shared encoder weights to the new-entity scoped model and initialises unseen entity embeddings from training-set entities via `_copy_train_state_to_eval`.

**Fixed parameters (all runs):**

| Parameter | Value |
|---|---|
| Embedding dim (`dim_feat`) | 100 |
| GNN layers (drug) | 3 × [256, 256, 256] |
| Graph conv layers | 2 |
| Dropout | 0.4 |
| Disentanglement weight ($\alpha$) | 8.0 |
| Weight decay | $10^{-5}$ |
| Optimizer | Adam |
| Epochs | 400 |
| $k$-fold CV | 5, seed 0 |

**Tuning search space** (6 candidates, lr / numk / rd / pd\_weight):

| lr | numk | rd | pd\_weight |
|---|---|---|---|
| 5e-4 | 3 | 0.25 | 1.0 |
| 5e-4 | 5 | 0.25 | 1.0 |
| 1e-3 | 5 | 0.25 | 1.0 |
| 1e-3 | 5 | 0.5 | 1.5 |
| 1e-3 | 7 | 0.5 | 1.5 |
| 5e-4 | 7 | 0.5 | 2.0 |

Tuning budget: Stage 1 — 60 epochs (fold 1); Stage 2 — 120 epochs (folds 1–2). Selection metric: validation AUC.

**Best configuration — dataset-1 (GraphCDR benchmark):**

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| numk | 5 |
| rd | 0.25 |
| pd\_weight | 1.0 |
| Val AUC at selection | 0.7864 |

**Best configuration — dataset-2 (Custom 3-omics):**

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| numk | 5 |
| rd | 0.25 |
| pd\_weight | 1.0 |
| Val AUC at selection | 0.9150 |

**Multi-seed Evaluation:** All experiments use seed 0 for reproducibility.

---

## E. Evaluation Metrics

- **AUC** (Area Under the ROC Curve): Measures overall discriminative ability across all thresholds.
- **AUPR** (Area Under the Precision-Recall Curve): Particularly important due to class imbalance in drug response data (especially in the GraphCDR benchmark where sensitive pairs are ~11% of all pairs).
- **F1 Score:** Harmonic mean of precision and recall at the optimal threshold.
- **Accuracy:** Binary classification accuracy at the optimal F1 threshold.

AUC and AUPR were used as the primary metrics for evaluating model performance.