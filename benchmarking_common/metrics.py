from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _to_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like.astype(float)
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy().astype(float)
    return np.asarray(array_like, dtype=float)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(-np.trapz(precision, recall))


def _best_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.5

    real_score = np.atleast_2d(np.asarray(y_true).flatten())
    predict_score = np.atleast_2d(np.asarray(y_score).flatten())
    sorted_predict_score = np.array(sorted(list(set(np.asarray(predict_score).flatten()))))
    if sorted_predict_score.size == 0:
        return 0.0, 0.0, 0.5

    threshold_indices = np.int32(sorted_predict_score.size * np.arange(1, 1000) / 1000)
    threshold_indices = np.clip(threshold_indices, 0, sorted_predict_score.size - 1)
    thresholds = np.atleast_2d(np.asarray(sorted_predict_score[threshold_indices]).flatten())

    predict_score_matrix = np.tile(predict_score, (thresholds.shape[1], 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    tp = predict_score_matrix.dot(real_score.T)
    fp = predict_score_matrix.sum(axis=1, keepdims=True) - tp
    fn = real_score.sum() - tp
    total = real_score.shape[1]
    tn = total - tp - fp - fn

    f1_scores = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) != 0,
    )
    acc_scores = np.divide(
        tp + tn,
        total,
        out=np.zeros_like(tp, dtype=float),
        where=total != 0,
    )
    max_index = int(np.argmax(f1_scores))
    threshold = float(thresholds.flat[max_index]) if thresholds.size else 0.5
    return (
        float(np.clip(f1_scores.flat[max_index], 0.0, 1.0)),
        float(np.clip(acc_scores.flat[max_index], 0.0, 1.0)),
        threshold,
    )


def compute_binary_metrics(y_true, y_score) -> Dict[str, float]:
    y_true_np = _to_numpy(y_true).reshape(-1)
    y_score_np = _to_numpy(y_score).reshape(-1)
    auc = _safe_auc(y_true_np, y_score_np)
    aupr = _safe_aupr(y_true_np, y_score_np)
    f1, acc, threshold = _best_threshold_metrics(y_true_np, y_score_np)
    return {
        "auc": auc,
        "aupr": aupr,
        "f1": f1,
        "acc": acc,
        "threshold": threshold,
    }


def summarize_metric_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    if not rows:
        return {"mean": {}, "std": {}}
    keys = [key for key in rows[0].keys() if key != "fold"]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    std = {key: float(np.std([row[key] for row in rows])) for key in keys}
    return {"mean": mean, "std": std}
