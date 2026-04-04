# Drug-Cell Pair Processing Module Testing Code

This folder contains the later-stage SOULCDR experiments built on top of the prepared custom dataset. The focus here is on the **full drug-cell pair processing pipeline**, including graph-construction choices, contrastive learning, hyperparameter tuning, and final benchmarking/ablation studies.

## Main files

- `main.py`: entry point for the SOULCDR experiments in this folder
- `model.py`: main SOULCDR-style model used in these experiments
- `model_GraphCDR.py`: GraphCDR-related reference model code used for comparison
- `export_final_dataset.py`: export script for the final custom dataset used by these runs
- `contrastive_loss.py`: supervised contrastive objective
- `plot_training_curves.py` and `plot_roc_curves.py`: visualization utilities

## Phase structure

- **Phase 1**: node-representation synchronization experiments
  - `run_phase1_node_representation.sh`
- **Phase 2**: graph/global-GNN/graph-transformer variation experiments
  - `run_phase2_graph_variations.sh`
- **Phase 3**: contrastive learning and hyperparameter tuning
  - `run_phase3_contrastive_and_hyperparameter.sh`
- **Phase 4**: final 5-fold benchmarking and ablations
  - `run_phase4_final_benchmarking.sh`

## Results to read first

- `soulcdr_results_phase_wise.txt`: consolidated readable results across the main experimental phases
- `soulcdr_phase3_results_labeled.txt`: labeled Phase 3 contrastive-learning results
- `logs/phase3_hp_summary.txt`: validation-set hyperparameter tuning summary
- `logs/phase4_cl_ablation.csv`: contrastive-learning ablation outputs
- `logs/phase4_omics_ablation.csv`: omics drop-one ablation outputs

## Supporting artifacts

- `edge similarity experiment results/`: pathway-based cell similarity and physicochemical drug similarity analysis files
- `notes.txt`: design notes on similarity measures, graph choices, and argument meanings
- `logs/`: raw training and tuning logs

## Notes

- This folder is the main experimental space where the final SOULCDR design choices were tested before being translated into the benchmark-ready pipeline.
- Phase 3 tuning decisions should be taken from `logs/phase3_hp_summary.txt`, because that file contains the validation-set ranking used during tuning.
- Phase 4 results are the main final benchmarking outputs in this folder, while the earlier phases explain how the final design was selected.
