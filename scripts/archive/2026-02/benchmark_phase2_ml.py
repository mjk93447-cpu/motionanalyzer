"""
Phase 2 ML benchmark on synthetic data: DREAM and PatchCore.

Generates synthetic normal/crack datasets, trains each model on normal data,
evaluates AUC-ROC on normal+crack. Results are for internal comparison and
regression checks; domain differs from MVTec AD (see docs/PHASE2_ML_REVIEW.md).

Usage (from repo root, with ML deps):
  pip install -e ".[ml]"
  python scripts/benchmark_phase2_ml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def _generate_synthetic_datasets(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """Create minimal normal and crack synthetic datasets. Returns (normal_dirs, crack_dirs)."""
    normal_dir = tmp_path / "normal"
    crack_dir = tmp_path / "crack"
    normal_dir.mkdir()
    crack_dir.mkdir()
    cfg_n = SyntheticConfig(frames=12, points_per_frame=80, fps=24.0, seed=42, scenario="normal")
    cfg_c = SyntheticConfig(frames=12, points_per_frame=80, fps=24.0, seed=123, scenario="crack")
    generate_synthetic_bundle(normal_dir, cfg_n)
    generate_synthetic_bundle(crack_dir, cfg_c)
    return [normal_dir], [crack_dir]


def _prepare_data(normal_paths: list[Path], crack_paths: list[Path]):
    """Return (X_normalized, labels_array, feature_cols)."""
    features_df, labels = prepare_training_data(
        normal_datasets=normal_paths,
        crack_datasets=crack_paths,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
        ),
    )
    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [c for c in features_df.columns if c not in exclude and c in features_df.select_dtypes(include=["number"]).columns]
    if not feature_cols:
        feature_cols = [c for c in features_df.columns if c not in exclude]
    # Fit normalization on normal-only to avoid leakage
    normal_mask = np.asarray(labels, dtype=int) == 0
    normalized = normalize_features(features_df, exclude_cols=exclude, fit_df=features_df.loc[normal_mask])
    X_norm = normalized[feature_cols].fillna(0).to_numpy(dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    return X_norm, labels_arr, feature_cols


def _run_patchcore_benchmark(X_train, y_train, X_test, y_test, n_features: int) -> float | None:
    """Train PatchCore on normal, predict on test, return AUC-ROC or None if sklearn missing."""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
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
    if len(np.unique(y_test)) < 2:
        return 0.5
    return float(roc_auc_score(y_test, scores))


def _run_dream_benchmark(X_train, y_train, X_test, y_test, n_features: int) -> float | None:
    """Train DREAM on normal, predict on test, return AUC-ROC or None if torch missing."""
    try:
        import torch  # noqa: F401
        from sklearn.metrics import roc_auc_score
        from motionanalyzer.ml_models.dream import DREAMPyTorch
    except ImportError:
        return None
    normal_mask = np.asarray(y_train) == 0
    X_normal = np.asarray(X_train)[normal_mask]
    if len(X_normal) < 2:
        return None
    try:
        model = DREAMPyTorch(input_dim=n_features, hidden_dims=[32, 16], latent_dim=4, batch_size=min(16, len(X_normal)))
    except ImportError:
        return None
    model.fit(X_normal.astype(np.float32), epochs=25)
    model.set_threshold_from_normal(X_normal.astype(np.float32), percentile=95.0)
    scores = model.predict(X_test.astype(np.float32))
    if len(np.unique(y_test)) < 2:
        return 0.5
    return float(roc_auc_score(y_test, scores))


def main() -> None:
    import tempfile
    import numpy as np

    print("Phase 2 ML benchmark (synthetic data)")
    print("--------------------------------------")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        normal_paths, crack_paths = _generate_synthetic_datasets(tmp_path)
        X, labels, feature_cols = _prepare_data(normal_paths, crack_paths)
        n_features = len(feature_cols)
        # Same data for train/test (quick benchmark; for strict eval use train/test split)
        X_train = X
        y_train = np.asarray(labels)
        X_test = X
        y_test = np.asarray(labels)

        results = {}
        try:
            auc_p = _run_patchcore_benchmark(X_train, y_train, X_test, y_test, n_features)
        except Exception:
            auc_p = None
        if auc_p is not None:
            results["PatchCore"] = auc_p
            print(f"  PatchCore (synthetic) AUC-ROC: {auc_p:.4f}")
        else:
            print("  PatchCore: skipped (scikit-learn not installed or error)")

        try:
            auc_d = _run_dream_benchmark(X_train, y_train, X_test, y_test, n_features)
        except Exception:
            auc_d = None
        if auc_d is not None:
            results["DREAM"] = auc_d
            print(f"  DREAM (synthetic) AUC-ROC: {auc_d:.4f}")
        else:
            print("  DREAM: skipped (PyTorch not installed or error)")

    print("--------------------------------------")
    print("Literature (different domain): PatchCore ~99.6% image AUROC on MVTec AD; AE-based ~99%.")
    print("Our domain: FPCB tabular/series; use this script for regression and relative comparison only.")
    if results:
        print(f"Summary: {results}")


if __name__ == "__main__":
    main()
