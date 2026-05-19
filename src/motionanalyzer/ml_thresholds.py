"""Threshold selection for balanced precision/recall targets (pretrain + real refine)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score


def aggregate_dataset_scores(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    *,
    path_col: str = "dataset_path",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bundle max score and label (dataset-level evaluation)."""
    if path_col not in features_df.columns:
        return scores, np.zeros(len(scores), dtype=int)
    df = features_df[[path_col]].copy()
    df["score"] = scores
    if "label" in features_df.columns:
        df["label"] = features_df["label"].values
    else:
        df["label"] = 0
    agg = df.groupby(path_col, sort=False).agg({"score": "max", "label": "max"}).reset_index(drop=True)
    return agg["score"].to_numpy(dtype=float), agg["label"].to_numpy(dtype=int)


def select_threshold_target_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float = 0.9,
    target_recall: float = 0.7,
    min_precision: float = 0.85,
    min_recall: float = 0.55,
) -> tuple[float, float, float, dict[str, int]]:
    """
    Pick threshold minimizing distance to (target_precision, target_recall) on PR curve.

    Prefers candidates with precision >= min_precision and recall >= min_recall when possible.
  """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or (labels == 0).all():
        return float(np.percentile(scores, 95)), 0.0, 0.0, {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    prec, rec, thresh = precision_recall_curve(labels, scores)
    best_t = float(np.percentile(scores[labels == 0], 95)) if (labels == 0).any() else 0.5
    best_loss = float("inf")
    best_p, best_r = 0.0, 0.0

    for i in range(len(thresh)):
        t = float(thresh[i])
        pred = (scores >= t).astype(int)
        p = float(precision_score(labels, pred, zero_division=0))
        r = float(recall_score(labels, pred, zero_division=0))
        loss = (p - target_precision) ** 2 + (r - target_recall) ** 2
        feasible = p >= min_precision and r >= min_recall
        if feasible and loss < best_loss:
            best_loss = loss
            best_t, best_p, best_r = t, p, r

    if best_loss == float("inf"):
        for i in range(len(thresh)):
            t = float(thresh[i])
            pred = (scores >= t).astype(int)
            p = float(precision_score(labels, pred, zero_division=0))
            r = float(recall_score(labels, pred, zero_division=0))
            loss = (p - target_precision) ** 2 + (r - target_recall) ** 2
            if loss < best_loss:
                best_loss = loss
                best_t, best_p, best_r = t, p, r

    pred = (scores >= best_t).astype(int)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    counts = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    return best_t, best_p, best_r, counts


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (np.asarray(scores) >= float(threshold)).astype(int)
    y = np.asarray(labels, dtype=int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tp, fp, fn, tn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0]), int(cm[0, 0])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
