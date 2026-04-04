# Node Representation Modules Testing Code

This folder contains the intermediate experimental code used to test **node-representation design choices** before moving to the later SOULCDR drug-cell pair processing experiments. The focus here is on modifying the GraphCDR-style architecture and comparing modality combinations, GNN backbones, cell-line encoders, and drug encoders.

## Main files

- `main.py`: entry point for all experiments in this folder
- `model.py`: modified architecture under study
- `model_baseline.py`: baseline GraphCDR-style model
- `data_load.py` and `data_process.py`: dataset loading and split preparation
- `run_phase0_modified_experiments.sh`: 1-epoch smoke pass over all planned modified configs
- `run_phase1_shortlist.sh`: first scored shortlist experiments
- `run_phase2_rank_best_configs.sh`: final 5-fold comparison of shortlisted configs

## Phase structure

- **Phase 0**: smoke testing of all planned modified-architecture configurations
- **Phase 1**: shortlist comparison across modality sets, drug GNNs, cell-line modules, and drug representation variants
- **Phase 2**: final 5-fold evaluation of shortlisted modified configurations

## Results to read first

- `phase_wise_results_summary.txt`: phase-wise formatted summary of what was tested and what performed best
- `baseline_results.txt`: separate 5-fold baseline GraphCDR reference result
- `notes.txt`: shortlist notes and rationale for which configs were carried forward
- `logs/`: raw logs and per-phase CSV summaries

## Notes

- Phase 0 should be interpreted as **runtime feasibility only**, not as a real performance comparison.
- The main scored summaries for this folder are the per-phase CSV files in `logs/` and the consolidated `phase_wise_results_summary.txt`.
- A nominal Phase 2 baseline run failed in the logs, so the usable baseline reference is the separate `baseline_results.txt` artifact.
