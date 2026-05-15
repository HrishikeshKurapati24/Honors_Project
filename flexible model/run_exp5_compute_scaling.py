import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from flexibility_utils import FLEXIBILITY_OUTPUTS_DIR, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 5: aggregate compute scaling outputs from experiments 1-4.")
    parser.add_argument("--exp1-dir", type=str, default=None)
    parser.add_argument("--exp2-dir", type=str, default=None)
    parser.add_argument("--exp3-dir", type=str, default=None)
    parser.add_argument("--exp4-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    args = parse_args()
    tag = args.tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = FLEXIBILITY_OUTPUTS_DIR / "exp5_compute_scaling" / tag
    experiment_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    if args.exp1_dir:
        exp1_dir = Path(args.exp1_dir).resolve()
        frame = _load_csv_if_exists(exp1_dir / "results.csv")
        if not frame.empty:
            frame["experiment"] = "exp1_subset_cardinality"
            frames.append(frame)

    if args.exp2_dir:
        exp2_dir = Path(args.exp2_dir).resolve()
        frame = _load_csv_if_exists(exp2_dir / "results.csv")
        if not frame.empty:
            frame["experiment"] = "exp2_pathway_sharding"
            frames.append(frame)

    if args.exp3_dir:
        exp3_dir = Path(args.exp3_dir).resolve()
        frame = _load_csv_if_exists(exp3_dir / "aggregate_metrics.csv")
        if not frame.empty:
            frame["experiment"] = "exp3_missing_modality"
            frames.append(frame)

    if args.exp4_dir:
        exp4_dir = Path(args.exp4_dir).resolve()
        frame = _load_csv_if_exists(exp4_dir / "results.csv")
        if not frame.empty:
            frame["experiment"] = "exp4_noisy_modality"
            frames.append(frame)

    if not frames:
        raise ValueError("No experiment directories with readable CSV outputs were provided.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(experiment_dir / "combined_compute_scaling.csv", index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            "experiment": "exp5_compute_scaling",
            "sources": {
                "exp1_dir": args.exp1_dir,
                "exp2_dir": args.exp2_dir,
                "exp3_dir": args.exp3_dir,
                "exp4_dir": args.exp4_dir,
            },
            "combined_csv": str(experiment_dir / "combined_compute_scaling.csv"),
        },
    )
    print(f"Saved experiment outputs to: {experiment_dir}")


if __name__ == "__main__":
    main()
