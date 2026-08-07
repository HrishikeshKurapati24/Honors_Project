import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
FLEXIBLE_DIR = REPO_ROOT / "flexible model"
FINAL_DATASET_DIR = REPO_ROOT / "final_dataset"
FLEXIBILITY_OUTPUTS_DIR = FLEXIBLE_DIR / "flexibility_outputs"
FLEXIBILITY_VIEWS_DIR = FLEXIBLE_DIR / "flexibility_views"
MAIN_FLEXIBLE_PATH = FLEXIBLE_DIR / "main_flexible.py"

REAL_OMICS_STEMS = [
    "genomics_mutation",
    "epigenomics_chromatin",
    "epigenomics_methylation",
    "transcriptomics_expression",
    "proteomics_reverse_phase",
    "metabolomics_profile",
    "pathway",
]
GRAPHCDR_MATCHED_OMICS_STEMS = [
    "genomics_mutation",
    "transcriptomics_expression",
    "epigenomics_methylation",
]
SHARED_NON_OMICS_FILES = [
    "response_pairs.csv",
    "similarity.csv",
    "physicochemical.csv",
]


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_json(path: Path | str) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Dict) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path_obj


def load_response_pairs(dataset_root: Path | str = FINAL_DATASET_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(dataset_root) / "response_pairs.csv")


def list_real_omics_stems(dataset_root: Path | str = FINAL_DATASET_DIR) -> List[str]:
    dataset_root = Path(dataset_root)
    available = []
    for stem in REAL_OMICS_STEMS:
        if (dataset_root / f"{stem}.csv").is_file():
            available.append(stem)
    return available


def summarize_flexible_summary(summary: Dict) -> Dict[str, float]:
    return {
        "auc": float(summary["mean"]["test_auc"]),
        "aupr": float(summary["mean"]["test_aupr"]),
        "f1": float(summary["mean"]["test_f1"]),
        "acc": float(summary["mean"]["test_acc"]),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "mean_fold_elapsed_seconds": float(summary.get("system", {}).get("mean_fold_elapsed_seconds", 0.0)),
        "max_peak_gpu_memory_bytes": float(summary.get("system", {}).get("max_peak_gpu_memory_bytes", 0.0)),
        "parameter_count": float(summary.get("system", {}).get("parameter_count", 0.0)),
    }


def _normalize_identifier(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return text


def _normalize_label(value) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = 0
    if numeric == -1:
        return 0
    return 1 if numeric > 0 else 0


def canonicalize_response_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cell_id": df["cell_id"].map(_normalize_identifier)
            if "cell_id" in df.columns
            else df["cell_line_id"].map(_normalize_identifier),
            "drug_id": df["drug_id"].map(_normalize_identifier)
            if "drug_id" in df.columns
            else df["pubchem_id"].map(_normalize_identifier),
            "label": df["label"].map(_normalize_label),
        }
    )
    out = out[(out["cell_id"] != "") & (out["drug_id"] != "")]
    out = out.sort_values(["cell_id", "drug_id", "label"], ascending=[True, True, False])
    out = out.drop_duplicates(["cell_id", "drug_id"], keep="first")
    return out.sort_values(["cell_id", "drug_id"]).reset_index(drop=True)


def build_dataset_view(
    output_dir: Path | str,
    *,
    base_dataset_root: Path | str = FINAL_DATASET_DIR,
    include_stems: Optional[Sequence[str]] = None,
    exclude_stems: Optional[Sequence[str]] = None,
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)
    base_dataset_root = Path(base_dataset_root)
    source_stems = (
        list_real_omics_stems(base_dataset_root)
        if include_stems is None
        else include_stems
    )
    include_set = set(source_stems)
    exclude_set = set(exclude_stems or [])
    selected_stems = sorted(include_set - exclude_set)

    for filename in SHARED_NON_OMICS_FILES:
        shutil.copy2(base_dataset_root / filename, output_dir / filename)

    source_graph_dir = base_dataset_root / "drug_graph_feat"
    target_graph_dir = output_dir / "drug_graph_feat"
    if target_graph_dir.exists():
        shutil.rmtree(target_graph_dir)
    shutil.copytree(source_graph_dir, target_graph_dir)

    for stem in selected_stems:
        src = base_dataset_root / f"{stem}.csv"
        if not src.is_file():
            raise FileNotFoundError(f"Missing omics source file: {src}")
        shutil.copy2(src, output_dir / f"{stem}.csv")

    for stem, frame in (extra_tables or {}).items():
        frame.to_csv(output_dir / f"{stem}.csv")

    manifest = {
        "base_dataset_root": str(base_dataset_root),
        "selected_stems": selected_stems,
        "extra_tables": sorted((extra_tables or {}).keys()),
    }
    write_json(output_dir / "view_manifest.json", manifest)
    return output_dir


def build_pathway_shards(
    *,
    base_dataset_root: Path | str = FINAL_DATASET_DIR,
    shard_count: int,
) -> Dict[str, pd.DataFrame]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    pathway = pd.read_csv(Path(base_dataset_root) / "pathway.csv", index_col=0)
    if shard_count > pathway.shape[1]:
        raise ValueError(
            f"shard_count ({shard_count}) cannot exceed the number of pathway "
            f"features ({pathway.shape[1]})"
        )
    column_shards = np.array_split(pathway.columns.to_numpy(), shard_count)
    shards: Dict[str, pd.DataFrame] = {}
    for shard_index, columns in enumerate(column_shards, start=1):
        if len(columns) == 0:
            continue
        shards[f"pathway_shard_{shard_index:02d}"] = pathway.loc[:, list(columns)].copy()
    return shards


def build_noisy_pathway_tables(
    *,
    base_dataset_root: Path | str = FINAL_DATASET_DIR,
    count: int,
    seed: int,
    mode: str = "permute",
) -> Dict[str, pd.DataFrame]:
    if count <= 0:
        raise ValueError("count must be positive")
    pathway = pd.read_csv(Path(base_dataset_root) / "pathway.csv", index_col=0)
    rng = np.random.default_rng(seed)
    tables: Dict[str, pd.DataFrame] = {}
    values = pathway.to_numpy(dtype=np.float32)
    for idx in range(1, count + 1):
        if mode == "permute":
            noisy_values = values.copy()
            for col_idx in range(noisy_values.shape[1]):
                noisy_values[:, col_idx] = rng.permutation(noisy_values[:, col_idx])
        elif mode == "gaussian":
            noisy_values = rng.normal(loc=0.0, scale=1.0, size=values.shape).astype(np.float32)
        elif mode == "hybrid":
            noisy_values = values.copy()
            for col_idx in range(noisy_values.shape[1]):
                noisy_values[:, col_idx] = rng.permutation(noisy_values[:, col_idx])
            noisy_values = noisy_values + 0.1 * rng.normal(size=noisy_values.shape).astype(np.float32)
        else:
            raise ValueError(f"Unsupported noise mode: {mode}")
        tables[f"pathway_noise_{idx:02d}"] = pd.DataFrame(
            noisy_values,
            index=pathway.index.copy(),
            columns=pathway.columns.copy(),
        )
    return tables


def run_main_flexible(
    *,
    run_name: str,
    dataset_root: Path | str = FINAL_DATASET_DIR,
    omics: Optional[Sequence[str]] = None,
    epoch: int = 400,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_channels: int = 256,
    output_channels: int = 64,
    fusion_dim: int = 512,
    dropout: float = 0.2,
    num_layers: int = 2,
    heads: int = 4,
    drug_num_gnn_layers: int = 3,
    drug_batch_size: int = 0,
    k_fold: int = 5,
    seed: int = 0,
    top_k: int = 10,
    contrastive_weight: float = 0.005,
    temperature: float = 0.05,
    warmup_epochs: int = 10,
    max_contrastive_pairs: int = 2048,
    device: str = "auto",
    split_dir: Optional[Path | str] = None,
    fold_ids: Optional[Sequence[int]] = None,
    save_checkpoints: bool = False,
) -> Dict:
    cmd = [
        sys.executable,
        str(MAIN_FLEXIBLE_PATH),
        "--run_name",
        run_name,
        "--dataset_root",
        str(Path(dataset_root)),
        "--epoch",
        str(epoch),
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--hidden_channels",
        str(hidden_channels),
        "--output_channels",
        str(output_channels),
        "--fusion_dim",
        str(fusion_dim),
        "--dropout",
        str(dropout),
        "--num_layers",
        str(num_layers),
        "--heads",
        str(heads),
        "--drug_num_gnn_layers",
        str(drug_num_gnn_layers),
        "--drug_batch_size",
        str(drug_batch_size),
        "--k_fold",
        str(k_fold),
        "--seed",
        str(seed),
        "--top_k",
        str(top_k),
        "--contrastive_weight",
        str(contrastive_weight),
        "--temperature",
        str(temperature),
        "--warmup_epochs",
        str(warmup_epochs),
        "--max_contrastive_pairs",
        str(max_contrastive_pairs),
        "--device",
        device,
    ]
    if omics:
        cmd.extend(["--omics", *omics])
    if split_dir is not None:
        cmd.extend(["--split_dir", str(Path(split_dir))])
    if fold_ids:
        cmd.extend(["--fold_ids", *[str(fold_id) for fold_id in fold_ids]])
    if save_checkpoints:
        cmd.append("--save_checkpoints")

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    run_dir = FLEXIBLE_DIR / "outputs" / run_name
    return read_json(run_dir / "summary.json")


def save_flexible_random_folds(
    *,
    response_pairs_path: Path | str,
    output_dir: Path | str,
    seed: int = 0,
    n_splits: int = 5,
    val_ratio_of_full: float = 0.1,
) -> Path:
    output_dir = ensure_dir(output_dir)
    canonical = canonicalize_response_pairs(pd.read_csv(response_pairs_path))
    allpairs = canonical[["cell_id", "drug_id", "label"]].to_numpy(dtype=object)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    manifest = {
        "protocol": "random",
        "seed": seed,
        "n_splits": n_splits,
        "val_ratio_of_full": val_ratio_of_full,
        "source": "flexible_process_flexible_compatible",
    }
    write_json(output_dir / "split_manifest.json", manifest)

    for fold_id, (train_val_idx, test_idx) in enumerate(splitter.split(allpairs), start=1):
        test_df = canonical.iloc[test_idx].reset_index(drop=True)
        val_size = int(len(allpairs) * val_ratio_of_full)
        val_size = max(1, min(val_size, len(train_val_idx) - 1))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_val_idx))
        val_idx = train_val_idx[perm[:val_size]]
        train_idx = train_val_idx[perm[val_size:]]

        train_df = canonical.iloc[train_idx].reset_index(drop=True)
        val_df = canonical.iloc[val_idx].reset_index(drop=True)
        fold_dir = ensure_dir(output_dir / f"fold_{fold_id}")
        train_df.to_csv(fold_dir / "train.csv", index=False)
        val_df.to_csv(fold_dir / "val.csv", index=False)
        test_df.to_csv(fold_dir / "test.csv", index=False)
        write_json(
            fold_dir / "entities.json",
            {
                "train_cells": sorted(train_df["cell_id"].astype(str).unique().tolist()),
                "val_cells": sorted(val_df["cell_id"].astype(str).unique().tolist()),
                "test_cells": sorted(test_df["cell_id"].astype(str).unique().tolist()),
                "train_drugs": sorted(train_df["drug_id"].astype(str).unique().tolist()),
                "val_drugs": sorted(val_df["drug_id"].astype(str).unique().tolist()),
                "test_drugs": sorted(test_df["drug_id"].astype(str).unique().tolist()),
            },
        )
    return output_dir


def build_graphcdr_prepared_view(
    output_dir: Path | str,
    *,
    base_dataset_root: Path | str = FINAL_DATASET_DIR,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)
    base_dataset_root = Path(base_dataset_root)
    for stem in GRAPHCDR_MATCHED_OMICS_STEMS:
        shutil.copy2(base_dataset_root / f"{stem}.csv", output_dir / f"{stem}.csv")
    for filename in SHARED_NON_OMICS_FILES:
        shutil.copy2(base_dataset_root / filename, output_dir / filename)
    source_graph_dir = base_dataset_root / "drug_graph_feat"
    target_graph_dir = output_dir / "drug_graph_feat"
    if target_graph_dir.exists():
        shutil.rmtree(target_graph_dir)
    shutil.copytree(source_graph_dir, target_graph_dir)
    response_pairs = canonicalize_response_pairs(pd.read_csv(base_dataset_root / "response_pairs.csv"))
    metadata = {
        "benchmark": "3OmicsStrictBenchmarking",
        "dataset": "final_dataset_graphcdr_baseline",
        "models": [
            "FUSECDR",
            "GraphCDR",
            "RedCDR",
            "GADRP",
            "DeepTTC",
            "GraphDRP",
        ],
        "omics_for_fusecdr": GRAPHCDR_MATCHED_OMICS_STEMS,
        "cell_graph_source": "similarity.csv",
        "drug_similarity_graph_source": "physicochemical.csv",
        "drug_structure_source": "drug_graph_feat",
        "response_graph_source": "train_pairs",
        "graph_builder": "topk_directed_cosine",
        "strict_predictive_inputs": [
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ],
        "strict_graph_inputs": [
            "similarity.csv",
            "physicochemical.csv",
            "train_pairs",
        ],
        "disabled_models_pending_strict_alignment": [],
        "cell_count": int(response_pairs["cell_id"].nunique()),
        "drug_count": int(response_pairs["drug_id"].nunique()),
        "pair_count": int(len(response_pairs)),
        "notes": [
            "Temporary strict-style prepared view built from final_dataset for flexibility baseline comparison."
        ],
    }
    write_json(output_dir / "metadata.json", metadata)
    response_pairs.to_csv(output_dir / "response_pairs.csv", index=False)
    return output_dir
