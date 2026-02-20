"""
DREAM (DRAEM strategy) validation on synthetic data.

Reports accuracy, precision, recall, F1, and AUC-ROC. DRAEM paper reports
98.1% image-level ROC AUC on MVTec AD; our domain is FPCB tabular—use this
script for regression and relative comparison only.

Usage (from repo root):
  pip install -e ".[ml]"
  python scripts/validate_dream_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.auto_optimize import (
    FeatureExtractionConfig,
    normalize_features,
    prepare_training_data,
)
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def _generate_synthetic_datasets(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    normal_dir = tmp_path / "normal"
    crack_dir = tmp_path / "crack"
    normal_dir.mkdir()
    crack_dir.mkdir()
    generate_synthetic_bundle(normal_dir, SyntheticConfig(frames=16, points_per_frame=80, fps=24.0, seed=42, scenario="normal"))
    generate_synthetic_bundle(crack_dir, SyntheticConfig(frames=16, points_per_frame=80, fps=24.0, seed=123, scenario="crack"))
    return [normal_dir], [crack_dir]


def _prepare_data(normal_paths: list[Path], crack_paths: list[Path]):
    features_df, labels = prepare_training_data(
        normal_datasets=normal_paths,
        crack_datasets=crack_paths,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            # Avoid Physics-derived crack_risk features (leakage risk)
            include_crack_risk_features=False,
        ),
    )
    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [c for c in features_df.columns if c not in exclude and c in features_df.select_dtypes(include=["number"]).columns]
    if not feature_cols:
        feature_cols = [c for c in features_df.columns if c not in exclude]
    # Fit normalization on normal-only to avoid leakage
    normal_mask = np.asarray(labels, dtype=int) == 0
    normalized = normalize_features(features_df, exclude_cols=exclude, fit_df=features_df.loc[normal_mask])
    X = normalized[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    return X, y, feature_cols


def main() -> None:
    try:
        import torch
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )
        from motionanalyzer.ml_models.dream import DREAMPyTorch
    except ImportError as e:
        print("DREAM validation requires PyTorch and scikit-learn: pip install -e '.[ml]'")
        raise SystemExit(1) from e

    import tempfile
    print("DREAM (DRAEM strategy) validation on synthetic data")
    # Avoid non-CP949 punctuation on some Windows consoles
    print("Reference: Zavrtanik et al., ICCV 2021 - MVTec image-level ROC AUC 98.1%")
    print("Our domain: FPCB tabular; results not directly comparable.\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        normal_paths, crack_paths = _generate_synthetic_datasets(tmp_path)
        X, y, feature_cols = _prepare_data(normal_paths, crack_paths)
        n_features = len(feature_cols)

        normal_mask = y == 0
        X_normal = X[normal_mask]
        X_crack = X[~normal_mask]
        if len(X_normal) < 2 or len(X_crack) < 1:
            print("Not enough normal/crack samples.")
            return

        # Prepare feature names for crack-like anomaly generation
        feature_cols_list = feature_cols

        model = DREAMPyTorch(
            input_dim=n_features,
            hidden_dims=[32, 16],
            latent_dim=4,
            use_discriminative=True,
            synthetic_noise_std=0.3,  # Fallback if feature names don't match
            discriminator_weight=0.5,
            batch_size=min(16, len(X_normal)),
        )
        # Use DataFrame to pass feature names for crack-like anomaly generation
        normal_df = pd.DataFrame(X_normal, columns=feature_cols_list)
        print(f"Feature columns ({len(feature_cols_list)}): {feature_cols_list[:5]}..." if len(feature_cols_list) > 5 else f"Feature columns: {feature_cols_list}")
        model.fit(normal_df, epochs=60, feature_names=feature_cols_list)  # More epochs for better learning

        # Optimize threshold for precision-recall balance
        crack_df = pd.DataFrame(X_crack, columns=feature_cols_list)
        thresh, opt_metrics = model.optimize_threshold_for_precision_recall(
            normal_df,
            crack_df,
            target_metric="balanced",  # F1 with recall >= 0.7
        )
        print(f"\nThreshold optimization (balanced):")
        print(f"  Optimal threshold: {thresh:.4f}")
        print(f"  Precision: {opt_metrics['precision']:.4f}, Recall: {opt_metrics['recall']:.4f}, F1: {opt_metrics['f1']:.4f}")

        scores_all = model.predict(X)
        pred_binary = model.predict_binary(X)

        acc = accuracy_score(y, pred_binary)
        prec = precision_score(y, pred_binary, zero_division=0)
        rec = recall_score(y, pred_binary, zero_division=0)
        f1 = f1_score(y, pred_binary, zero_division=0)
        auc = roc_auc_score(y, scores_all) if len(np.unique(y)) == 2 else 0.5

    print("Metrics (synthetic normal vs crack):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print("\nDRAEM paper (MVTec AD): image-level ROC AUC 98.1%. Compare trend only; domain differs.")


if __name__ == "__main__":
    main()
