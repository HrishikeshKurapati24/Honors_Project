import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_claim_utils import (
    DATASET_CLAIM_OUTPUTS_DIR,
    average_neighbor_overlap,
    load_real_omics_frames,
    modality_similarity_matrix,
    pairwise_matrix_correlation,
    write_json,
)
from flexibility_utils import FINAL_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset-claim experiment 5: static modality redundancy analysis.")
    parser.add_argument("--dataset-root", type=str, default=str(FINAL_DATASET_DIR))
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = DATASET_CLAIM_OUTPUTS_DIR / "dc5_redundancy_analysis" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    frames = load_real_omics_frames(dataset_root)
    stems = sorted(frames.keys())
    similarity_matrices = {
        stem: modality_similarity_matrix(frame)
        for stem, frame in frames.items()
    }

    corr_matrix = pd.DataFrame(index=stems, columns=stems, dtype=float)
    overlap_matrix = pd.DataFrame(index=stems, columns=stems, dtype=float)
    pair_rows = []
    for stem_a in stems:
        for stem_b in stems:
            if stem_a == stem_b:
                corr = 1.0
                overlap = 1.0
            else:
                corr = pairwise_matrix_correlation(similarity_matrices[stem_a], similarity_matrices[stem_b])
                overlap = average_neighbor_overlap(
                    similarity_matrices[stem_a],
                    similarity_matrices[stem_b],
                    top_k=args.neighbor_k,
                )
            corr_matrix.loc[stem_a, stem_b] = corr
            overlap_matrix.loc[stem_a, stem_b] = overlap
            if stem_a < stem_b:
                pair_rows.append(
                    {
                        "modality_a": stem_a,
                        "modality_b": stem_b,
                        "similarity_matrix_correlation": corr,
                        "average_neighbor_overlap": overlap,
                    }
                )

    corr_matrix.to_csv(experiment_dir / "similarity_matrix_correlation.csv")
    overlap_matrix.to_csv(experiment_dir / "neighbor_overlap.csv")
    pd.DataFrame(pair_rows).sort_values(
        ["similarity_matrix_correlation", "average_neighbor_overlap"],
        ascending=False,
    ).to_csv(experiment_dir / "pairwise_redundancy.csv", index=False)

    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "dc5_redundancy_analysis",
            "dataset_root": str(dataset_root),
            "neighbor_k": args.neighbor_k,
            "modalities": stems,
            "n_cells": int(next(iter(frames.values())).shape[0]) if frames else 0,
            "pairwise_redundancy_csv": str(experiment_dir / "pairwise_redundancy.csv"),
            "similarity_matrix_correlation_csv": str(experiment_dir / "similarity_matrix_correlation.csv"),
            "neighbor_overlap_csv": str(experiment_dir / "neighbor_overlap.csv"),
        },
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
