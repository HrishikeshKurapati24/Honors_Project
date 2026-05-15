import os
from typing import Dict, List

from benchmark_wrappers import deepttc_shared_graph_runner as base_runner
from benchmarking_common import read_json, write_json


MODEL_KEY = "DeepTTC_largebatch"


def _rewrite_summary_metadata(results_dir: str, batch_size: int) -> Dict | None:
    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        return None
    summary = read_json(summary_path)
    metadata = dict(summary.get("metadata", {}))
    metadata["model"] = MODEL_KEY
    metadata["base_model"] = "DeepTTC"
    metadata["training_regime"] = "large_batch_minibatch"
    config = dict(metadata.get("config", {}))
    config["batch_size"] = batch_size
    metadata["config"] = config
    summary["metadata"] = metadata
    write_json(summary_path, summary)
    return summary


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 80,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_size: int = 512,
    top_k: int = 10,
    fold_ids: List[int] | None = None,
) -> Dict:
    summary = base_runner.run(
        root_dir=root_dir,
        prepared_dir=prepared_dir,
        split_dir=split_dir,
        results_dir=results_dir,
        device=device,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        top_k=top_k,
        fold_ids=fold_ids,
    )
    rewritten = _rewrite_summary_metadata(results_dir, batch_size)
    return rewritten or summary
