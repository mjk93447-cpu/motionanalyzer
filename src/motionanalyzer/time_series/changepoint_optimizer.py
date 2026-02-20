"""
Change Point Detection parameter optimization.

Provides automatic parameter tuning for CUSUM, Window-based, and PELT detectors
using Grid Search and Bayesian Optimization (Optuna).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from motionanalyzer.time_series.changepoint import (
    ChangePointDetector,
    ChangePointResult,
    CUSUMDetector,
    WindowBasedDetector,
    detect_change_points_pelt,
)


@dataclass
class CPDOptimizationResult:
    """Result of change point detection parameter optimization."""

    best_params: dict[str, Any]
    """Best parameters found."""
    best_score: float
    """Best score achieved."""
    method: str
    """Detection method name."""
    all_trials: list[dict[str, Any]] | None = None
    """All trial results (for analysis)."""


def optimize_cusum_parameters(
    signal: np.ndarray | pd.Series,
    expected_change_range: tuple[int, int] | None = None,
    threshold_range: tuple[float, float] = (0.5, 10.0),
    sensitivity_range: tuple[float, float] | None = None,
    n_trials: int = 20,
    optimization_method: str = "grid",
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float] | None = None,
) -> CPDOptimizationResult:
    """
    Optimize CUSUM parameters using Grid Search or Bayesian Optimization.
    
    Args:
        signal: Time series signal
        expected_change_range: Expected change point range (frame_start, frame_end) for scoring
        threshold_range: Range for threshold parameter (min, max)
        sensitivity_range: Range for sensitivity parameter (min, max). If None, uses (0.1*std, 2.0*std)
        n_trials: Number of optimization trials
        optimization_method: "grid" or "bayesian"
        score_function: Custom scoring function. If None, uses default (detection in range = 1.0, else 0.0)
    
    Returns:
        CPDOptimizationResult with best parameters and score
    """
    if isinstance(signal, pd.Series):
        signal = signal.values
    signal = np.asarray(signal, dtype=np.float64)
    
    if sensitivity_range is None:
        std_val = float(np.std(signal))
        sensitivity_range = (0.1 * std_val, 2.0 * std_val)
    
    if score_function is None:
        def default_score(result: ChangePointResult, expected_range: tuple[int, int] | None) -> float:
            if expected_range is None:
                return 1.0 if len(result.change_points) > 0 else 0.0
            if len(result.change_points) == 0:
                return 0.0
            # Check if any change point is in expected range
            in_range = any(expected_range[0] <= cp <= expected_range[1] for cp in result.change_points)
            return 1.0 if in_range else 0.0
        score_function = default_score
    
    if optimization_method == "grid":
        return _optimize_cusum_grid(
            signal, expected_change_range, threshold_range, sensitivity_range, score_function
        )
    elif optimization_method == "bayesian":
        return _optimize_cusum_bayesian(
            signal, expected_change_range, threshold_range, sensitivity_range, n_trials, score_function
        )
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


def _optimize_cusum_grid(
    signal: np.ndarray,
    expected_range: tuple[int, int] | None,
    threshold_range: tuple[float, float],
    sensitivity_range: tuple[float, float],
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float],
) -> CPDOptimizationResult:
    """Grid search for CUSUM parameters."""
    threshold_values = np.linspace(threshold_range[0], threshold_range[1], 10)
    sensitivity_values = np.linspace(sensitivity_range[0], sensitivity_range[1], 10)
    
    best_score = -1.0
    best_params: dict[str, Any] = {}
    all_trials = []
    
    detector = CUSUMDetector()
    
    for thresh in threshold_values:
        for sens in sensitivity_values:
            result = detector.detect(signal, threshold=thresh, sensitivity=sens)
            score = score_function(result, expected_range)
            
            trial = {
                "threshold": thresh,
                "sensitivity": sens,
                "score": score,
                "change_points": result.change_points,
            }
            all_trials.append(trial)
            
            if score > best_score:
                best_score = score
                best_params = {"threshold": thresh, "sensitivity": sens}
    
    return CPDOptimizationResult(
        best_params=best_params,
        best_score=best_score,
        method="CUSUM",
        all_trials=all_trials,
    )


def _optimize_cusum_bayesian(
    signal: np.ndarray,
    expected_range: tuple[int, int] | None,
    threshold_range: tuple[float, float],
    sensitivity_range: tuple[float, float],
    n_trials: int,
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float],
) -> CPDOptimizationResult:
    """Bayesian optimization for CUSUM parameters using Optuna."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Bayesian optimization requires Optuna. Install with: pip install optuna")
    
    detector = CUSUMDetector()
    all_trials = []
    
    def objective(trial: Any) -> float:
        threshold = trial.suggest_float("threshold", threshold_range[0], threshold_range[1], log=True)
        sensitivity = trial.suggest_float("sensitivity", sensitivity_range[0], sensitivity_range[1], log=True)
        
        result = detector.detect(signal, threshold=threshold, sensitivity=sensitivity)
        score = score_function(result, expected_range)
        
        all_trials.append({
            "threshold": threshold,
            "sensitivity": sensitivity,
            "score": score,
            "change_points": result.change_points,
        })
        
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    return CPDOptimizationResult(
        best_params=best_params,
        best_score=best_score,
        method="CUSUM",
        all_trials=all_trials,
    )


def optimize_window_parameters(
    signal: np.ndarray | pd.Series,
    expected_change_range: tuple[int, int] | None = None,
    window_size_range: tuple[int, int] = (5, 20),
    threshold_ratio_range: tuple[float, float] = (1.2, 3.0),
    n_trials: int = 20,
    optimization_method: str = "grid",
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float] | None = None,
) -> CPDOptimizationResult:
    """
    Optimize Window-based detector parameters.
    
    Args:
        signal: Time series signal
        expected_change_range: Expected change point range for scoring
        window_size_range: Range for window_size parameter (min, max)
        threshold_ratio_range: Range for threshold_ratio parameter (min, max)
        n_trials: Number of optimization trials (for Bayesian)
        optimization_method: "grid" or "bayesian"
        score_function: Custom scoring function
    
    Returns:
        CPDOptimizationResult with best parameters
    """
    if isinstance(signal, pd.Series):
        signal = signal.values
    signal = np.asarray(signal, dtype=np.float64)
    
    if score_function is None:
        def default_score(result: ChangePointResult, expected_range: tuple[int, int] | None) -> float:
            if expected_range is None:
                return 1.0 if len(result.change_points) > 0 else 0.0
            if len(result.change_points) == 0:
                return 0.0
            in_range = any(expected_range[0] <= cp <= expected_range[1] for cp in result.change_points)
            return 1.0 if in_range else 0.0
        score_function = default_score
    
    if optimization_method == "grid":
        return _optimize_window_grid(
            signal, expected_change_range, window_size_range, threshold_ratio_range, score_function
        )
    elif optimization_method == "bayesian":
        return _optimize_window_bayesian(
            signal, expected_change_range, window_size_range, threshold_ratio_range, n_trials, score_function
        )
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


def _optimize_window_grid(
    signal: np.ndarray,
    expected_range: tuple[int, int] | None,
    window_size_range: tuple[int, int],
    threshold_ratio_range: tuple[float, float],
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float],
) -> CPDOptimizationResult:
    """Grid search for Window-based parameters."""
    window_sizes = np.arange(window_size_range[0], window_size_range[1] + 1, 2)
    threshold_ratios = np.linspace(threshold_ratio_range[0], threshold_ratio_range[1], 10)
    
    best_score = -1.0
    best_params: dict[str, Any] = {}
    all_trials = []
    
    detector = WindowBasedDetector()
    
    for ws in window_sizes:
        for tr in threshold_ratios:
            result = detector.detect(signal, window_size=ws, threshold_ratio=tr)
            score = score_function(result, expected_range)
            
            trial = {
                "window_size": ws,
                "threshold_ratio": tr,
                "score": score,
                "change_points": result.change_points,
            }
            all_trials.append(trial)
            
            if score > best_score:
                best_score = score
                best_params = {"window_size": ws, "threshold_ratio": tr}
    
    return CPDOptimizationResult(
        best_params=best_params,
        best_score=best_score,
        method="WindowBased",
        all_trials=all_trials,
    )


def _optimize_window_bayesian(
    signal: np.ndarray,
    expected_range: tuple[int, int] | None,
    window_size_range: tuple[int, int],
    threshold_ratio_range: tuple[float, float],
    n_trials: int,
    score_function: Callable[[ChangePointResult, tuple[int, int] | None], float],
) -> CPDOptimizationResult:
    """Bayesian optimization for Window-based parameters."""
    try:
        import optuna
    except ImportError:
        raise ImportError("Bayesian optimization requires Optuna. Install with: pip install optuna")
    
    detector = WindowBasedDetector()
    all_trials = []
    
    def objective(trial: Any) -> float:
        window_size = trial.suggest_int("window_size", window_size_range[0], window_size_range[1])
        threshold_ratio = trial.suggest_float(
            "threshold_ratio", threshold_ratio_range[0], threshold_ratio_range[1], log=True
        )
        
        result = detector.detect(signal, window_size=window_size, threshold_ratio=threshold_ratio)
        score = score_function(result, expected_range)
        
        all_trials.append({
            "window_size": window_size,
            "threshold_ratio": threshold_ratio,
            "score": score,
            "change_points": result.change_points,
        })
        
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    return CPDOptimizationResult(
        best_params=best_params,
        best_score=best_score,
        method="WindowBased",
        all_trials=all_trials,
    )


def detect_change_points_multi_feature(
    features_df: pd.DataFrame,
    feature_names: list[str],
    method: str = "cusum",
    combine_strategy: str = "union",
    **method_kwargs: Any,
) -> ChangePointResult:
    """
    Detect change points using multiple features simultaneously.
    
    Args:
        features_df: DataFrame with 'frame' column and feature columns
        feature_names: List of feature column names to analyze
        method: Detection method ("cusum", "window", "pelt")
        combine_strategy: How to combine results ("union", "intersection", "majority")
        **method_kwargs: Method-specific parameters
    
    Returns:
        Combined ChangePointResult
    """
    if "frame" not in features_df.columns:
        raise ValueError("features_df must have 'frame' column")
    
    all_change_points: list[list[int]] = []
    
    for feature_name in feature_names:
        if feature_name not in features_df.columns:
            continue
        
        signal = features_df[feature_name].values
        signal = signal[~np.isnan(signal)]
        
        if len(signal) < 2:
            continue
        
        if method == "cusum":
            detector = CUSUMDetector(**{k: v for k, v in method_kwargs.items() if k in ["threshold", "sensitivity", "min_size"]})
            result = detector.detect(signal)
        elif method == "window":
            detector = WindowBasedDetector(**{k: v for k, v in method_kwargs.items() if k in ["window_size", "threshold_ratio", "min_size"]})
            result = detector.detect(signal)
        elif method == "pelt":
            result = detect_change_points_pelt(signal, **{k: v for k, v in method_kwargs.items() if k in ["min_size", "pen", "jump"]})
        else:
            raise ValueError(f"Unknown method: {method}")
        
        all_change_points.append(result.change_points)
    
    if not all_change_points:
        return ChangePointResult(change_points=[], method=f"MultiFeature-{method}")
    
    # Combine change points based on strategy
    if combine_strategy == "union":
        # All unique change points from any feature
        combined = sorted(set(cp for cps in all_change_points for cp in cps))
    elif combine_strategy == "intersection":
        # Only change points detected by all features
        if len(all_change_points) == 1:
            combined = all_change_points[0]
        else:
            combined = sorted(set.intersection(*[set(cps) for cps in all_change_points]))
    elif combine_strategy == "majority":
        # Change points detected by majority of features
        from collections import Counter
        all_cps = [cp for cps in all_change_points for cp in cps]
        cp_counts = Counter(all_cps)
        threshold = len(all_change_points) / 2.0
        combined = sorted([cp for cp, count in cp_counts.items() if count > threshold])
    else:
        raise ValueError(f"Unknown combine_strategy: {combine_strategy}")
    
    return ChangePointResult(
        change_points=combined,
        method=f"MultiFeature-{method}-{combine_strategy}",
    )


def ensemble_change_point_detection(
    features_df: pd.DataFrame,
    feature_names: list[str],
    methods: list[str] = ["cusum", "window"],
    combine_strategy: str = "union",
    **method_kwargs: Any,
) -> ChangePointResult:
    """
    Ensemble change point detection using multiple methods and features.
    
    Args:
        features_df: DataFrame with 'frame' column and feature columns
        feature_names: List of feature column names
        methods: List of detection methods to use
        combine_strategy: How to combine results ("union", "intersection", "majority")
        **method_kwargs: Method-specific parameters
    
    Returns:
        Ensemble ChangePointResult
    """
    all_results: list[ChangePointResult] = []
    
    for method in methods:
        result = detect_change_points_multi_feature(
            features_df, feature_names, method=method, combine_strategy="union", **method_kwargs
        )
        all_results.append(result)
    
    # Combine results from different methods
    all_change_points = [r.change_points for r in all_results]
    
    if combine_strategy == "union":
        combined = sorted(set(cp for cps in all_change_points for cp in cps))
    elif combine_strategy == "intersection":
        if len(all_change_points) == 1:
            combined = all_change_points[0]
        else:
            combined = sorted(set.intersection(*[set(cps) for cps in all_change_points]))
    elif combine_strategy == "majority":
        from collections import Counter
        all_cps = [cp for cps in all_change_points for cp in cps]
        cp_counts = Counter(all_cps)
        threshold = len(all_change_points) / 2.0
        combined = sorted([cp for cp, count in cp_counts.items() if count > threshold])
    else:
        raise ValueError(f"Unknown combine_strategy: {combine_strategy}")
    
    return ChangePointResult(
        change_points=combined,
        method=f"Ensemble-{'-'.join(methods)}-{combine_strategy}",
    )
