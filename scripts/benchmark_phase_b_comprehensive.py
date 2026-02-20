"""
Phase B Comprehensive Benchmark: All ML Models on Enhanced Synthetic Data.

This script validates all Phase B improvements:
- DREAM (baseline)
- PatchCore (baseline)
- Ensemble (DREAM + PatchCore)
- Temporal (LSTM/GRU)
- Advanced Features (with/without)

Metrics: ROC AUC, PR AUC (AUCPR), Precision, Recall, F1-score

Usage:
  pip install -e ".[ml]"
  python scripts/benchmark_phase_b_comprehensive.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure package on path (src layout)
repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from motionanalyzer.auto_optimize import (
    FeatureExtractionConfig,
    normalize_features,
    prepare_training_data,
)
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def _generate_enhanced_synthetic_datasets(tmp_path: Path, n_normal: int = 5, n_crack: int = 5) -> tuple[list[Path], list[Path]]:
    """
    Generate enhanced synthetic datasets with shockwave and vibration patterns.
    
    Returns:
        (normal_dirs, crack_dirs)
    """
    normal_dirs = []
    crack_dirs = []
    
    for i in range(n_normal):
        normal_dir = tmp_path / f"normal_{i}"
        normal_dir.mkdir(parents=True, exist_ok=True)
        cfg = SyntheticConfig(
            frames=120,
            points_per_frame=230,
            fps=30.0,
            seed=42 + i,
            scenario="normal",
        )
        generate_synthetic_bundle(normal_dir, cfg)
        normal_dirs.append(normal_dir)
    
    for i in range(n_crack):
        crack_dir = tmp_path / f"crack_{i}"
        crack_dir.mkdir(parents=True, exist_ok=True)
        cfg = SyntheticConfig(
            frames=120,
            points_per_frame=230,
            fps=30.0,
            seed=100 + i,
            scenario="crack",
        )
        generate_synthetic_bundle(crack_dir, cfg)
        crack_dirs.append(crack_dir)
    
    return normal_dirs, crack_dirs


def _prepare_data(
    normal_paths: list[Path],
    crack_paths: list[Path],
    include_advanced_stats: bool = False,
    include_frequency_domain: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare training data with optional advanced features.
    
    Returns:
        (X_normalized, labels_array, feature_cols)
    """
    features_df, labels = prepare_training_data(
        normal_datasets=normal_paths,
        crack_datasets=crack_paths,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,  # Prevent label leakage
            include_advanced_stats=include_advanced_stats,
            include_frequency_domain=include_frequency_domain,
        ),
    )
    
    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [
        c for c in features_df.columns
        if c not in exclude and c in features_df.select_dtypes(include=["number"]).columns
    ]
    if not feature_cols:
        feature_cols = [c for c in features_df.columns if c not in exclude]
    
    # Fit normalization on normal-only to avoid leakage
    normal_mask = np.asarray(labels, dtype=int) == 0
    normalized = normalize_features(features_df, exclude_cols=exclude, fit_df=features_df.loc[normal_mask])
    X_norm = normalized[feature_cols].fillna(0).to_numpy(dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    
    return X_norm, labels_arr, feature_cols


def _compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    """Compute comprehensive metrics."""
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError:
        return {}
    
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.5, "pr_auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.5}
    
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))
    
    if threshold is None:
        # Use optimal threshold based on F1-score
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
        best_f1 = 0.0
        best_threshold = thresholds[0]
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        threshold = best_threshold
    
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def _run_dream_benchmark(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_features: int
) -> dict[str, Any] | None:
    """Train DREAM and return metrics."""
    try:
        import torch  # noqa: F401
        from motionanalyzer.ml_models.dream import DREAMPyTorch
    except ImportError:
        return None
    
    normal_mask = np.asarray(y_train) == 0
    X_normal = np.asarray(X_train)[normal_mask]
    
    if len(X_normal) < 2:
        return None
    
    model = DREAMPyTorch(input_dim=n_features, hidden_dims=[64, 32, 16], learning_rate=1e-3)
    model.fit(X_normal, epochs=20)
    model.set_threshold_from_normal(X_normal, percentile=95.0)
    
    scores = model.predict(X_test)
    metrics = _compute_metrics(y_test, scores)
    metrics["model"] = "DREAM"
    
    return metrics


def _run_patchcore_benchmark(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_features: int
) -> dict[str, Any] | None:
    """Train PatchCore and return metrics."""
    try:
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError:
        return None
    
    normal_mask = np.asarray(y_train) == 0
    X_normal = np.asarray(X_train)[normal_mask]
    
    if len(X_normal) < 2:
        return None
    
    model = PatchCoreScikitLearn(feature_dim=n_features, coreset_size=min(100, len(X_normal)), k_neighbors=1)
    model.fit(X_normal)
    model.set_threshold_from_normal(X_normal, percentile=95.0)
    
    scores = model.predict(X_test)
    metrics = _compute_metrics(y_test, scores)
    metrics["model"] = "PatchCore"
    
    return metrics


def _run_ensemble_benchmark(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_features: int
) -> dict[str, Any] | None:
    """Train Ensemble (DREAM + PatchCore) and return metrics."""
    try:
        import torch  # noqa: F401
        from motionanalyzer.ml_models.dream import DREAMPyTorch
        from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError:
        return None
    
    normal_mask = np.asarray(y_train) == 0
    X_normal = np.asarray(X_train)[normal_mask]
    X_anomaly = np.asarray(X_train)[~normal_mask]
    
    if len(X_normal) < 2 or len(X_anomaly) < 1:
        return None
    
    # Train base models
    dream_model = DREAMPyTorch(input_dim=n_features, hidden_dims=[64, 32, 16], learning_rate=1e-3)
    dream_model.fit(X_normal, epochs=20)
    
    patchcore_model = PatchCoreScikitLearn(feature_dim=n_features, coreset_size=min(100, len(X_normal)), k_neighbors=1)
    patchcore_model.fit(X_normal)
    
    # Create ensemble
    ensemble = EnsembleAnomalyDetector(
        dream_model=dream_model,
        patchcore_model=patchcore_model,
        strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
        dream_weight=0.5,
        patchcore_weight=0.5,
    )
    
    # Optimize weights
    dream_weight, patchcore_weight, opt_metrics = ensemble.optimize_weights(
        normal_data=X_normal, anomaly_data=X_anomaly, target_metric="f1"
    )
    ensemble.dream_weight = dream_weight
    ensemble.patchcore_weight = patchcore_weight
    
    ensemble.set_threshold_from_normal(X_normal, percentile=95.0)
    
    scores = ensemble.predict(X_test)
    metrics = _compute_metrics(y_test, scores)
    metrics["model"] = "Ensemble"
    metrics["dream_weight"] = float(dream_weight)
    metrics["patchcore_weight"] = float(patchcore_weight)
    
    return metrics


def _run_temporal_benchmark(
    normal_paths: list[Path],
    crack_paths: list[Path],
    include_advanced_stats: bool = False,
    include_frequency_domain: bool = False,
) -> dict[str, Any] | None:
    """
    Train Temporal model with proper time-series train/test split.
    
    Key: Split at dataset level BEFORE sequence construction to avoid data leakage.
    """
    try:
        import torch  # noqa: F401
        from motionanalyzer.ml_models.dream_temporal import TemporalAnomalyDetector
    except ImportError:
        return None
    
    # CRITICAL: Split datasets BEFORE feature extraction to avoid leakage
    # This ensures sequences are constructed only from training data
    np.random.seed(42)
    all_normal = list(normal_paths)
    all_crack = list(crack_paths)
    np.random.shuffle(all_normal)
    np.random.shuffle(all_crack)
    
    # 70/30 split at dataset level
    split_normal = int(len(all_normal) * 0.7)
    split_crack = int(len(all_crack) * 0.7)
    
    train_normal = all_normal[:split_normal]
    test_normal = all_normal[split_normal:]
    train_crack = all_crack[:split_crack]
    test_crack = all_crack[split_crack:]
    
    # Prepare training data (normal only for training)
    train_features_df, train_labels = prepare_training_data(
        normal_datasets=train_normal,
        crack_datasets=[],  # Temporal model trains on normal only
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,
            include_advanced_stats=include_advanced_stats,
            include_frequency_domain=include_frequency_domain,
        ),
    )
    
    # Prepare test data (normal + crack)
    test_features_df, test_labels = prepare_training_data(
        normal_datasets=test_normal,
        crack_datasets=test_crack,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,
            include_advanced_stats=include_advanced_stats,
            include_frequency_domain=include_frequency_domain,
        ),
    )
    
    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [
        c for c in train_features_df.columns
        if c not in exclude and c in train_features_df.select_dtypes(include=["number"]).columns
    ]
    
    # Normalize: fit on training normal data only, apply to test
    train_normal_mask = np.asarray(train_labels, dtype=int) == 0
    train_normalized = normalize_features(
        train_features_df, exclude_cols=exclude, fit_df=train_features_df.loc[train_normal_mask]
    )
    train_normalized = train_normalized.fillna(0)
    
    # Apply same normalization to test data
    test_normalized = normalize_features(
        test_features_df, exclude_cols=exclude, fit_df=train_features_df.loc[train_normal_mask]
    )
    test_normalized = test_normalized.fillna(0)
    
    # Training: use only normal data
    train_normal_df = train_normalized[train_normal_mask].copy()
    
    if len(train_normal_df) < 20:  # Need enough frames for sequences
        return None
    
    # Train temporal model
    model = TemporalAnomalyDetector(
        feature_dim=len(feature_cols),
        sequence_length=10,
        hidden_dim=64,
        num_layers=2,
        cell_type="LSTM",
        learning_rate=1e-3,
        batch_size=32,
    )
    
    model.fit(train_normal_df, feature_cols, epochs=20)
    model.set_threshold_from_normal(train_normal_df, feature_cols, percentile=95.0)
    
    # Predict on test data
    test_pred_df = model.predict(test_normalized, feature_cols, aggregation="max")
    
    # Merge predictions with labels
    test_scores_df = test_normalized[["dataset_path", "frame"]].copy()
    test_scores_df = test_scores_df.merge(
        test_pred_df[["dataset_path", "frame", "anomaly_score"]],
        on=["dataset_path", "frame"],
        how="left",
    )
    test_scores_df = test_scores_df.fillna(0.0)
    
    # Align with test labels
    test_labels_df = test_normalized[["dataset_path", "frame"]].copy()
    test_labels_df["label"] = test_labels
    
    merged = test_scores_df.merge(test_labels_df, on=["dataset_path", "frame"], how="inner")
    
    if len(merged) == 0:
        return None
    
    scores = merged["anomaly_score"].values
    labels = merged["label"].values
    
    metrics = _compute_metrics(labels, scores)
    metrics["model"] = "Temporal"
    
    return metrics


def main() -> None:
    """Run comprehensive Phase B benchmark."""
    print("Phase B Comprehensive Benchmark")
    print("=" * 80)
    print("Testing: DREAM, PatchCore, Ensemble, Temporal (with/without Advanced Features)")
    print("=" * 80)
    
    results: list[dict[str, Any]] = []
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        print("\n[1/4] Generating enhanced synthetic datasets...")
        normal_paths, crack_paths = _generate_enhanced_synthetic_datasets(tmp_path, n_normal=5, n_crack=5)
        print(f"  Generated {len(normal_paths)} normal and {len(crack_paths)} crack datasets")
        
        # Split into train/test
        train_normal = normal_paths[:3] + crack_paths[:2]  # Mix for training
        train_crack = crack_paths[:2]
        test_normal = normal_paths[3:]
        test_crack = crack_paths[2:]
        
        print("\n[2/4] Preparing data (baseline features)...")
        X_train_base, y_train_base, feature_cols_base = _prepare_data(
            train_normal, train_crack, include_advanced_stats=False, include_frequency_domain=False
        )
        X_test_base, y_test_base, _ = _prepare_data(
            test_normal, test_crack, include_advanced_stats=False, include_frequency_domain=False
        )
        print(f"  Train: {len(X_train_base)} samples, {len(feature_cols_base)} features")
        print(f"  Test: {len(X_test_base)} samples")
        
        print("\n[3/4] Running benchmarks (baseline features)...")
        
        # DREAM
        print("  - DREAM...")
        dream_result = _run_dream_benchmark(X_train_base, y_train_base, X_test_base, y_test_base, len(feature_cols_base))
        if dream_result:
            dream_result["features"] = "baseline"
            results.append(dream_result)
            print(f"    ROC AUC: {dream_result['roc_auc']:.3f}, PR AUC: {dream_result['pr_auc']:.3f}")
        
        # PatchCore
        print("  - PatchCore...")
        patchcore_result = _run_patchcore_benchmark(
            X_train_base, y_train_base, X_test_base, y_test_base, len(feature_cols_base)
        )
        if patchcore_result:
            patchcore_result["features"] = "baseline"
            results.append(patchcore_result)
            print(f"    ROC AUC: {patchcore_result['roc_auc']:.3f}, PR AUC: {patchcore_result['pr_auc']:.3f}")
        
        # Ensemble
        print("  - Ensemble...")
        ensemble_result = _run_ensemble_benchmark(
            X_train_base, y_train_base, X_test_base, y_test_base, len(feature_cols_base)
        )
        if ensemble_result:
            ensemble_result["features"] = "baseline"
            results.append(ensemble_result)
            print(f"    ROC AUC: {ensemble_result['roc_auc']:.3f}, PR AUC: {ensemble_result['pr_auc']:.3f}")
        
        # Temporal
        print("  - Temporal...")
        temporal_result = _run_temporal_benchmark(
            train_normal + train_crack, test_normal + test_crack, include_advanced_stats=False, include_frequency_domain=False
        )
        if temporal_result:
            temporal_result["features"] = "baseline"
            results.append(temporal_result)
            print(f"    ROC AUC: {temporal_result['roc_auc']:.3f}, PR AUC: {temporal_result['pr_auc']:.3f}")
        
        print("\n[4/4] Testing with Advanced Features...")
        X_train_adv, y_train_adv, feature_cols_adv = _prepare_data(
            train_normal, train_crack, include_advanced_stats=True, include_frequency_domain=True
        )
        X_test_adv, y_test_adv, _ = _prepare_data(
            test_normal, test_crack, include_advanced_stats=True, include_frequency_domain=True
        )
        print(f"  Advanced features: {len(feature_cols_adv)} (vs {len(feature_cols_base)} baseline)")
        
        # DREAM with advanced features
        print("  - DREAM (advanced features)...")
        dream_adv_result = _run_dream_benchmark(X_train_adv, y_train_adv, X_test_adv, y_test_adv, len(feature_cols_adv))
        if dream_adv_result:
            dream_adv_result["features"] = "advanced"
            dream_adv_result["model"] = "DREAM+Advanced"
            results.append(dream_adv_result)
            print(f"    ROC AUC: {dream_adv_result['roc_auc']:.3f}, PR AUC: {dream_adv_result['pr_auc']:.3f}")
        
        # Temporal with advanced features
        print("  - Temporal (advanced features)...")
        temporal_adv_result = _run_temporal_benchmark(
            train_normal + train_crack, test_normal + test_crack, include_advanced_stats=True, include_frequency_domain=True
        )
        if temporal_adv_result:
            temporal_adv_result["features"] = "advanced"
            temporal_adv_result["model"] = "Temporal+Advanced"
            results.append(temporal_adv_result)
            print(f"    ROC AUC: {temporal_adv_result['roc_auc']:.3f}, PR AUC: {temporal_adv_result['pr_auc']:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    if not results:
        print("No results available. Check ML dependencies (PyTorch, scikit-learn).")
        return
    
    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df[["model", "features", "roc_auc", "pr_auc", "precision", "recall", "f1"]].to_string(index=False))
    
    # Save results
    output_file = repo_root / "reports" / "phase_b_benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)
    
    baseline_results = results_df[results_df["features"] == "baseline"]
    if len(baseline_results) > 0:
        best_baseline = baseline_results.loc[baseline_results["roc_auc"].idxmax()]
        print(f"\nBest baseline model: {best_baseline['model']} (ROC AUC: {best_baseline['roc_auc']:.3f})")
    
    if "Ensemble" in results_df["model"].values:
        ensemble_roc = results_df[results_df["model"] == "Ensemble"]["roc_auc"].values[0]
        dream_roc = results_df[results_df["model"] == "DREAM"]["roc_auc"].values[0] if "DREAM" in results_df["model"].values else None
        if dream_roc:
            improvement = (ensemble_roc - dream_roc) / dream_roc * 100
            print(f"Ensemble vs DREAM improvement: {improvement:+.1f}%")
    
    advanced_results = results_df[results_df["features"] == "advanced"]
    if len(advanced_results) > 0:
        print("\nAdvanced Features Impact:")
        for model_name in ["DREAM", "Temporal"]:
            base_name = model_name
            adv_name = f"{model_name}+Advanced"
            if base_name in results_df["model"].values and adv_name in results_df["model"].values:
                base_roc = results_df[results_df["model"] == base_name]["roc_auc"].values[0]
                adv_roc = results_df[results_df["model"] == adv_name]["roc_auc"].values[0]
                improvement = (adv_roc - base_roc) / base_roc * 100
                print(f"  {model_name}: {base_roc:.3f} -> {adv_roc:.3f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    main()
