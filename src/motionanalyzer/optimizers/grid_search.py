"""
Grid search over CrackModelParams for FPCB crack detection.

Evaluates each parameter combination by loading normal/crack datasets with that
params and computing AUC-ROC on per-dataset max crack_risk (crack=1, normal=0).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from motionanalyzer.auto_optimize import load_dataset
from motionanalyzer.crack_model import CrackModelParams, crack_risk_global


def evaluate_params(
    params: CrackModelParams,
    normal_paths: list[Path],
    crack_paths: list[Path],
    *,
    fps: float | None = None,
) -> tuple[float, list[float], list[int]]:
    """
    Compute AUC-ROC for a single CrackModelParams on given datasets.

    Each dataset is loaded with the given params; one score per dataset
    (max crack_risk) and label 0 (normal) or 1 (crack). AUC-ROC is computed
    over these (score, label) pairs.

    Returns:
        (auc_roc, scores, labels). If no valid pairs (e.g. only one class), auc_roc is 0.5.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        raise ImportError("Grid search requires scikit-learn. Install with: pip install -e '.[ml]'") from e

    scores: list[float] = []
    labels_list: list[int] = []

    for path in normal_paths:
        try:
            info = load_dataset(path, label=0, fps=fps, crack_params=params)
            scores.append(crack_risk_global(info.vectors))
            labels_list.append(0)
        except Exception:
            continue

    for path in crack_paths:
        try:
            info = load_dataset(path, label=1, fps=fps, crack_params=params)
            scores.append(crack_risk_global(info.vectors))
            labels_list.append(1)
        except Exception:
            continue

    if not scores or len(set(labels_list)) < 2:
        return 0.5, scores, labels_list

    auc = float(roc_auc_score(labels_list, scores))
    return auc, scores, labels_list


def _param_grid_small() -> list[dict[str, Any]]:
    """Small default grid: a few values per key parameter (for speed)."""
    grid: list[dict[str, Any]] = []
    for sig_center in [0.4, 0.5]:
        for sig_steep in [6.0, 10.0]:
            grid.append({
                "sigmoid_center": sig_center,
                "sigmoid_steepness": sig_steep,
            })
    return grid


def run_grid_search(
    normal_paths: list[Path],
    crack_paths: list[Path],
    *,
    fps: float | None = None,
    param_grid: list[dict[str, Any]] | None = None,
    base_params: CrackModelParams | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Run grid search over CrackModelParams; maximize AUC-ROC.

    Args:
        normal_paths: Paths to normal dataset directories.
        crack_paths: Paths to crack dataset directories.
        fps: FPS (if None, read from each dataset).
        param_grid: List of dicts; each dict overrides base_params. If None, use small default grid.
        base_params: Base parameters; grid items override. If None, use CrackModelParams().
        log: Optional log callback.
        progress: Optional progress callback.

    Returns:
        Dict with success, message, best_params (CrackModelParams), best_auc, all_results.
    """
    def no_log(_: str) -> None:
        pass

    log_fn = log or no_log

    if not normal_paths and not crack_paths:
        return {
            "success": False,
            "message": "At least one normal or crack dataset path is required.",
        }

    base = base_params or CrackModelParams()
    grid = param_grid or _param_grid_small()
    all_results: list[tuple[CrackModelParams, float]] = []
    best_auc = -1.0
    best_params: CrackModelParams | None = None

    for i, override in enumerate(grid):
        if progress:
            progress()
        d = asdict(base)
        d.update(override)
        try:
            params = CrackModelParams(**d)
        except TypeError as e:
            log_fn(f"Skipping invalid grid point {override}: {e}")
            continue
        try:
            auc, _, _ = evaluate_params(params, normal_paths, crack_paths, fps=fps)
        except Exception as e:
            log_fn(f"Evaluate failed for {override}: {e}")
            continue
        all_results.append((params, auc))
        log_fn(f"Params {override} -> AUC-ROC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_params = params

    if best_params is None:
        return {
            "success": False,
            "message": "No valid grid point could be evaluated.",
        }

    log_fn(f"Best AUC-ROC: {best_auc:.4f}")
    return {
        "success": True,
        "message": f"Grid search complete. Best AUC-ROC = {best_auc:.4f}",
        "best_params": best_params,
        "best_auc": best_auc,
        "all_results": all_results,
    }
