# Local-only, HGT-only, and HGT-depth Study

This experiment compares four FUSE-CDR graph configurations while holding all
other settings fixed:

| Variant | GraphSAGE layers | HGT layers |
| --- | ---: | ---: |
| `local_only` | 2 | 0 |
| `hgt_2_only` | 0 | 2 |
| `hgt_2` | 2 | 2 |
| `hgt_3` | 2 | 3 |

The `hgt_2_only` model contains no GraphSAGE layers and performs no branch
fusion. This makes its comparison with `local_only` a direct two-layer HGT
versus two-layer GraphSAGE branch ablation. The `hgt_2` and `hgt_3` variants
retain both branches and isolate the effect of HGT depth in the full model.

The runner reuses each dataset's selected strict FUSE-CDR hyperparameters and
reads the canonical five-fold random splits directly from
`3OmicsStrictBenchmarking/splits/random/<dataset>`. These splits use the
historical seed-0 split policy; model training still uses a distinct seed for
each fold. No hyperparameter tuning is performed.

Before training, every canonical split is required to exactly reproduce the
historical algorithm, partition all prepared response pairs without overlap,
and match all available saved benchmark test predictions. SHA-256 hashes and
row counts are written to `split_audits/<dataset>.json` under the study output.

## Run

```bash
python3 3OmicsStrictBenchmarking/run_hgt_depth_study.py --device auto
```

The full run trains 40 fold models:

```text
2 datasets x 4 variants x 5 folds = 40 fold models
```

Results are written under:

```text
3OmicsStrictBenchmarking/results/hgt_depth_study_historical_v1/
```

The runner is resumable. A fold is reused only when both its metrics and best
checkpoint exist.

## Analyses

`variant_summary.csv` reports overall test performance. `distance_summary.csv`
reports AUC and AUPR for test drug-cell pairs whose shortest directed training
graph path is exactly two or three edges.

`propagation_summary.csv` reports the causal source-perturbation probe on
dataset-2. It uses exact directed cell-similarity paths and holds the other graph
branch and direct predictor endpoint fixed. The expected reach pattern is:

| Variant | Exact 2-hop influence | Exact 3-hop influence |
| --- | --- | --- |
| `local_only` | possible | impossible |
| `hgt_2_only` | possible | impossible |
| `hgt_2` | possible | impossible |
| `hgt_3` | possible | possible |

An influence beyond the expected receptive field indicates a graph or probe
implementation error. Reachable but negligible effects mean the architecture
can carry the signal but the trained model did not use it materially.
