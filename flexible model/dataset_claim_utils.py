import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from flexibility_utils import (
    FINAL_DATASET_DIR,
    REAL_OMICS_STEMS,
    FLEXIBLE_DIR,
    build_dataset_view,
    canonicalize_response_pairs,
    list_real_omics_stems,
    load_response_pairs,
    read_json,
    run_main_flexible,
    save_flexible_random_folds,
    summarize_flexible_summary,
    write_json,
)


DATASET_CLAIM_OUTPUTS_DIR = FLEXIBLE_DIR / "dataset_claim_outputs"
DATASET_CLAIM_VIEWS_DIR = FLEXIBLE_DIR / "dataset_claim_views"
DATASET_CLAIM_SPLITS_DIR = DATASET_CLAIM_OUTPUTS_DIR / "shared_splits"

DEFAULT_CURATED_CONFIGS = [
    {"name": "pathway", "omics": ["pathway"]},
    {
        "name": "mut_meth_exp",
        "omics": ["genomics_mutation", "epigenomics_methylation", "transcriptomics_expression"],
    },
    {
        "name": "exp_prot",
        "omics": ["transcriptomics_expression", "proteomics_reverse_phase"],
    },
    {
        "name": "full_7omics",
        "omics": list(REAL_OMICS_STEMS),
    },
    {"name": "expression_only", "omics": ["transcriptomics_expression"]},
    {
        "name": "meth_exp",
        "omics": ["epigenomics_methylation", "transcriptomics_expression"],
    },
    {
        "name": "mut_exp",
        "omics": ["genomics_mutation", "transcriptomics_expression"],
    },
    {
        "name": "prot_metab",
        "omics": ["proteomics_reverse_phase", "metabolomics_profile"],
    },
    {
        "name": "mut_chrom_exp",
        "omics": ["genomics_mutation", "epigenomics_chromatin", "transcriptomics_expression"],
    },
    {
        "name": "pathway_exp_prot",
        "omics": ["pathway", "transcriptomics_expression", "proteomics_reverse_phase"],
    },
]

ADD_ONE_IN_BASES = [
    {"name": "pathway", "omics": ["pathway"]},
    {"name": "exp_prot", "omics": ["transcriptomics_expression", "proteomics_reverse_phase"]},
    {
        "name": "mut_meth_exp",
        "omics": ["genomics_mutation", "epigenomics_methylation", "transcriptomics_expression"],
    },
]


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def dataset_name(dataset_root: Path | str) -> str:
    return Path(dataset_root).resolve().name


def default_split_dir(dataset_root: Path | str, seed: int, k_fold: int) -> Path:
    name = dataset_name(dataset_root)
    return DATASET_CLAIM_SPLITS_DIR / f"{name}_seed{seed}_k{k_fold}"


def ensure_dataset_claim_split_dir(
    *,
    dataset_root: Path | str = FINAL_DATASET_DIR,
    split_dir: Path | str | None = None,
    seed: int = 0,
    k_fold: int = 5,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    resolved_split_dir = Path(split_dir).resolve() if split_dir else default_split_dir(dataset_root, seed, k_fold)
    manifest_path = resolved_split_dir / "split_manifest.json"
    expected_fold = resolved_split_dir / "fold_1" / "train.csv"
    if manifest_path.is_file() and expected_fold.is_file():
        return resolved_split_dir
    ensure_dir(resolved_split_dir)
    save_flexible_random_folds(
        response_pairs_path=dataset_root / "response_pairs.csv",
        output_dir=resolved_split_dir,
        seed=seed,
        n_splits=k_fold,
    )
    return resolved_split_dir


def load_progress(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def save_progress(frame: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path


def run_or_reuse_flexible(
    *,
    run_name: str,
    dataset_root: Path | str,
    omics: Sequence[str],
    epoch: int,
    k_fold: int,
    seed: int,
    device: str,
    split_dir: Path | str,
    save_checkpoints: bool = False,
    **extra_kwargs,
) -> Dict:
    run_dir = FLEXIBLE_DIR / "outputs" / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    return run_main_flexible(
        run_name=run_name,
        dataset_root=dataset_root,
        omics=omics,
        epoch=epoch,
        k_fold=k_fold,
        seed=seed,
        device=device,
        split_dir=split_dir,
        save_checkpoints=save_checkpoints,
        **extra_kwargs,
    )


def result_row_from_summary(
    *,
    experiment: str,
    config_name: str,
    omics: Sequence[str],
    run_name: str,
    summary: Dict,
    extra_fields: Dict | None = None,
) -> Dict:
    metrics = summarize_flexible_summary(summary)
    row = {
        "experiment": experiment,
        "config_name": config_name,
        "omics_stems": "|".join(omics),
        "run_name": run_name,
        "run_dir": summary["run_dir"],
        **metrics,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def save_incremental_experiment_artifacts(
    *,
    experiment_dir: Path | str,
    rows: List[Dict],
    metadata: Dict,
    results_name: str = "results.csv",
    leaderboard_metric: str = "auc",
) -> None:
    experiment_dir = Path(experiment_dir)
    ensure_dir(experiment_dir)
    frame = pd.DataFrame(rows)
    if not frame.empty and leaderboard_metric in frame.columns:
        frame = frame.sort_values(leaderboard_metric, ascending=False).reset_index(drop=True)
    frame.to_csv(experiment_dir / results_name, index=False)
    write_json(
        experiment_dir / "summary.json",
        {
            **metadata,
            "completed_runs": int(len(frame)),
            "results_csv": str(experiment_dir / results_name),
        },
    )


def build_permuted_modality_view(
    *,
    output_dir: Path | str,
    dataset_root: Path | str = FINAL_DATASET_DIR,
    permute_stem: str,
    seed: int,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    omics_stems = list_real_omics_stems(dataset_root)
    view_dir = build_dataset_view(
        output_dir=output_dir,
        base_dataset_root=dataset_root,
        include_stems=omics_stems,
    )
    path = view_dir / f"{permute_stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot permute missing modality file: {path}")
    frame = pd.read_csv(path, index_col=0)
    rng = np.random.default_rng(seed)
    permuted_values = frame.to_numpy(copy=True)
    permuted_values = permuted_values[rng.permutation(permuted_values.shape[0])]
    pd.DataFrame(permuted_values, index=frame.index.copy(), columns=frame.columns.copy()).to_csv(path)
    return view_dir


def get_leave_one_out_configs(dataset_root: Path | str = FINAL_DATASET_DIR) -> List[Dict]:
    real_stems = list_real_omics_stems(dataset_root)
    configs = [{"name": "full_7omics", "omics": real_stems}]
    for stem in real_stems:
        configs.append(
            {
                "name": f"full_minus_{stem}",
                "omics": [candidate for candidate in real_stems if candidate != stem],
                "removed_stem": stem,
            }
        )
    return configs


def get_add_one_in_configs(dataset_root: Path | str = FINAL_DATASET_DIR) -> List[Dict]:
    available = list_real_omics_stems(dataset_root)
    configs: List[Dict] = []
    for base in ADD_ONE_IN_BASES:
        base_stems = [stem for stem in base["omics"] if stem in available]
        configs.append({"name": base["name"], "omics": base_stems, "base_name": base["name"], "added_stem": ""})
        for stem in available:
            if stem in base_stems:
                continue
            configs.append(
                {
                    "name": f"{base['name']}_plus_{stem}",
                    "omics": base_stems + [stem],
                    "base_name": base["name"],
                    "added_stem": stem,
                }
            )
    return configs


def get_curated_configs(dataset_root: Path | str = FINAL_DATASET_DIR) -> List[Dict]:
    available = set(list_real_omics_stems(dataset_root))
    configs = []
    for config in DEFAULT_CURATED_CONFIGS:
        if set(config["omics"]).issubset(available):
            configs.append(config)
    return configs


def modality_similarity_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    values = values / std
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = values / norms
    return normalized @ normalized.T


def pairwise_matrix_correlation(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    tri = np.triu_indices_from(matrix_a, k=1)
    vec_a = matrix_a[tri]
    vec_b = matrix_b[tri]
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return 0.0
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def average_neighbor_overlap(matrix_a: np.ndarray, matrix_b: np.ndarray, top_k: int = 10) -> float:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Neighbor overlap requires equal-shaped similarity matrices.")
    n = matrix_a.shape[0]
    if n <= 1:
        return 0.0
    k = max(1, min(top_k, n - 1))
    a = matrix_a.copy()
    b = matrix_b.copy()
    np.fill_diagonal(a, -np.inf)
    np.fill_diagonal(b, -np.inf)
    neigh_a = np.argsort(a, axis=1)[:, -k:]
    neigh_b = np.argsort(b, axis=1)[:, -k:]
    overlaps = []
    for row in range(n):
        set_a = set(neigh_a[row].tolist())
        set_b = set(neigh_b[row].tolist())
        overlaps.append(len(set_a & set_b) / float(len(set_a | set_b)))
    return float(np.mean(overlaps))


def load_real_omics_frames(dataset_root: Path | str = FINAL_DATASET_DIR) -> Dict[str, pd.DataFrame]:
    dataset_root = Path(dataset_root).resolve()
    stems = list_real_omics_stems(dataset_root)
    frames = {
        stem: pd.read_csv(dataset_root / f"{stem}.csv", index_col=0)
        for stem in stems
    }
    common_index = sorted(set.intersection(*[set(frame.index.astype(str)) for frame in frames.values()]))
    out = {}
    for stem, frame in frames.items():
        aligned = frame.copy()
        aligned.index = aligned.index.astype(str)
        out[stem] = aligned.loc[common_index]
    return out
