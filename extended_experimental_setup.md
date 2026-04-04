EXTENDED EXPERIMENTAL SETUP

------------------------------------------------------------
S1. DATA SOURCES AND CONSTRUCTION
------------------------------------------------------------

We construct and evaluate on three datasets with varying modality coverage and benchmarking roles.

Custom 7-Modalities Dataset (Model Development):
This dataset is built by integrating CCLE (DepMap 2019) and GDSC1. Cell-line identifiers are unified to DepMap IDs by mapping GDSC cell names via Cell_lines_annotations.csv and normalizing CCLE identifiers (lowercasing, removing hyphens/spaces). Unmapped entries are removed. Drugs are filtered to ensure presence across response tables, molecular graphs, and descriptor tables.

Custom 3-Omics Dataset (Benchmarking):
This dataset is independently constructed from the same upstream pipeline using mutation, gene expression, and methylation modalities. Compared to the 7-modal dataset, relaxed modality constraints retain more cell lines and drug–cell pairs.

GraphCDR Benchmark Dataset:
The publicly available GraphCDR dataset is reprocessed into a unified format. Labels are rebuilt, features are aligned, and splits are regenerated using a shared benchmarking pipeline to ensure fair comparison across models.

------------------------------------------------------------
Dataset Summary
------------------------------------------------------------

| Dataset                    | Modalities                         | Drugs | Cells | Pairs   | Purpose             |
|----------------------------|------------------------------------|-------|-------|---------|---------------------|
| Custom 7-Modalities        | Mut, Meth, Expr, Chrom, RPPA, Met, Pathway | 298   | 425   | 109,446 | Model Development   |
| Custom 3-Omics             | Mutation, Expression, Methylation  | 298   | 456   | 117,330 | Benchmarking        |
| GraphCDR Benchmark         | Mutation, Expression, Methylation  | 222   | 561   | 99,147  | Standard Benchmark  |

Note: Pathway refers to GSVA-derived features; Chrom = chromatin accessibility.

------------------------------------------------------------
S2. PREPROCESSING PIPELINE ( CUSTOM DATASET )
------------------------------------------------------------

Cell-line modalities:
All omics data are processed into a consistent cell-line × feature format. After DepMap ID mapping, non-feature columns are removed. Features and samples with >10% missing values are discarded. Remaining missing values are imputed using KNN (k=5). Low-variance features are filtered using VarianceThreshold = 0.03.

Pathway features (50-dimensional) are generated using GSVA over MSigDB Hallmark gene sets. PROGENy pathway scores are computed separately for similarity graph construction.

Drug data:
SMILES are obtained via PubChem CID mapping. Molecular graphs are generated using DeepChem ConvMolFeaturizer. Physicochemical descriptors are computed using Mordred, filtered to non-constant features, reduced to top 64 by variance, and z-score standardized.

Response labels:
For custom datasets, labels are defined as:
LN_IC50 < log(MAX_CONC)

Duplicate drug–cell pairs are resolved via majority voting (ties favor positive). Missing or unmapped entries are removed.

For the benchmark dataset, drug-specific thresholds are used following the GraphCDR protocol.

Feature scaling:
No additional normalization is applied at runtime, as datasets are already normalized during preprocessing. Scaling is applied globally where required.

Missing data handling:
Entities with incomplete modality coverage are removed during dataset construction. No runtime masking is used.

------------------------------------------------------------
S3. GRAPH CONSTRUCTION
------------------------------------------------------------

A directed top-k (k = 10) similarity graph is constructed using cosine similarity over raw features.

- Cell–cell similarity: PROGENy pathway scores
- Drug–drug similarity: physicochemical descriptors

For inductive settings:
- Similarity graphs are recomputed over in-scope entities
- Similarity edges for validation/test nodes are retained
- Response edges are strictly limited to training pairs

------------------------------------------------------------
S4. DATA SPLITTING AND EVALUATION PROTOCOL ( BENCHMARK EVALUATION )
------------------------------------------------------------

Five-fold cross-validation is performed using shuffled KFold with a fixed seed (0). Splits are shared across all models.

Inductive protocols:
Splits are constructed at the entity level.

Example: Unseen Both
- Cells and drugs are independently partitioned into 5 folds
- Train/validation/test entity sets are disjoint
- Training pairs: (train_cells × train_drugs)
- Validation pairs: (val_cells × val_drugs)
- Test pairs: (test_cells × test_drugs)

At runtime:
- Validation/test scopes include held-out entities for inference
- Training supervision is restricted to training pairs only

This ensures strict inductive evaluation: held-out entities are never seen with labels during training.

------------------------------------------------------------
S5. HYPERPARAMETER TUNING STRATEGY ( BENCHMARK EVALUATION )
------------------------------------------------------------

Hyperparameters are selected using two-stage successive halving on the random split.

Stage 1:
- All configurations evaluated with reduced budget
- Top 50% selected based on validation AUC

Stage 2:
- Remaining configurations evaluated with higher budget
- Best configuration selected

Key details:
- Tuning performed only on dataset-1 (benchmark dataset)
- Same configurations reused across all evaluation settings
- Predefined candidate sets per model (e.g., SOULCDR: 8 configs)
- Ties resolved deterministically

------------------------------------------------------------
S6. HYPERPARAMETER TUNING PROTOCOL ( BENCHMARK EVALUATION )
------------------------------------------------------------

In the current 3OmicsBenchmarking pipeline, three models—SOULCDR, GraphCDR, and RedCDR—undergo explicit hyperparameter tuning. The tuning logic is implemented in tuning.py, while protocol-level reuse behavior is handled in experiment.py.

GENERAL TUNING STRATEGY:
- Tuning Protocol:
  Hyperparameter optimization is performed only on the random split protocol.

- Inductive Protocol Reuse:
  For inductive settings (unseen_cells, unseen_drugs, unseen_both), no independent tuning is performed.
  Instead, the best configuration from the random split is reused with:
    "tuned": false
    "reused_from_protocol": "random"

- Search Method:
  A predefined candidate set is evaluated using a two-stage successive halving strategy:
    Stage 1:
      All candidates evaluated on fold 1 only.
    Selection:
      Top 50% (by validation AUC) retained.
    Stage 2:
      Survivors evaluated on folds 1 and 2.
    Final Selection:
      Configuration with highest validation AUC selected.

- Tie-breaking:
  Sorting is done using (AUC, config_json) in descending order.
  Equal AUC values are broken lexicographically.


SOULCDR:
- Tuning Scope:
  Tuned only on dataset-1. Dataset-2 tuning is skipped by policy.

- Tuned Parameters:
  lr, hidden_channels, output_channels, fusion_channels

- Fixed Defaults:
  dropout=0.2, num_layers=2, heads=4, drug_num_gnn_layers=3,
  top_k=10, contrastive_weight=0.005, temperature=0.05, warmup_epochs=10

- Successive Halving:
  Stage 1: 40 epochs
  Stage 2: 80 epochs

- Candidate Set Size:
  8 configurations

- Stage 2 Best (from tuning logs):
  hidden=512, output=64, fusion=128 (AUC = 0.802868)

- Saved Best Config (best_config.json):
  lr=0.001
  hidden_channels=256
  output_channels=128
  fusion_channels=256
  selected_score=0.8028683427295737

- Note:
  There is an inconsistency between tuning logs and best_config.json.
  For reporting, either treat best_config.json as final or explicitly mention the inconsistency.

- Dataset-2:
  Tuning skipped; best_config.json contains empty config with "tuned": false.

GRAPHCDR:

- Tuned Parameters:
  alpha, beta

- Fixed Defaults:
  epochs=350, lr=0.001, output_channels=100

- Successive Halving:
  Stage 1: 60 epochs
  Stage 2: 120 epochs

- Candidate Set Size:
  6 configurations

- Dataset-1 Best Config:
  alpha=0.2, beta=0.2

- Dataset-2 Best Config:
  alpha=0.2, beta=0.3

RedCDR:

- Tuned Parameters:
  lr, numk, rd, pd_weight

- Fixed Defaults:
  epochs=400, dropout=0.4, dim_feat=100, layers=2, alpha=8.0

- Successive Halving:
  Stage 1: 60 epochs
  Stage 2: 120 epochs

- Candidate Set Size:
  6 configurations

- Dataset-1 Best Config:
  lr=0.001, numk=5, rd=0.25, pd_weight=1.0

- Dataset-2 Best Config:
  lr=0.001, numk=5, rd=0.25, pd_weight=1.0

SUMMARY:
- SOULCDR:
  Tuned only on dataset-1; dataset-2 skipped. Contains inconsistency between tuning logs and saved config.

- GraphCDR:
  Tuned alpha and beta. Best configs:
    dataset-1 → (0.2, 0.2)
    dataset-2 → (0.2, 0.3)

- RedCDR:
  Tuned lr, numk, rd, pd_weight. Same best config across both datasets:
    lr=0.001, numk=5, rd=0.25, pd_weight=1.0


------------------------------------------------------------
S7. IMPLEMENTATION DETAILS
------------------------------------------------------------

- Framework: PyTorch + PyTorch Geometric
- Hardware: Google Colab (Tesla T4, 12GB)
- Training: full-batch graph training

No learning rate scheduler or gradient clipping is used.

Training epochs:
- SOULCDR: 200 (development), 400 (benchmarking)
- Baselines: follow original configurations

------------------------------------------------------------
S8. EVALUATION AND STATISTICAL TESTING
------------------------------------------------------------

Metrics are reported as mean ± standard deviation across folds.

Threshold selection:
- F1 maximized over candidate thresholds derived from prediction scores (~≤999 candidates)

Statistical testing:
- Paired Wilcoxon signed-rank test across folds

------------------------------------------------------------
S9. REPRODUCIBILITY
------------------------------------------------------------

- Random seed: 0
- Seeds fixed for Python, NumPy, and PyTorch
- cuDNN deterministic mode enabled (full determinism not guaranteed)

The complete implementation, including preprocessing and benchmarking pipelines, will be released publicly.