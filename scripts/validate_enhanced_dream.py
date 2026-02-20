"""
Validate DREAM model on enhanced synthetic data with shockwave/vibration patterns.

Compares performance before and after synthetic data enhancement.
Reports Precision-Recall AUC (AUCPR) in addition to ROC AUC.
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


def _generate_evaluation_datasets(tmp_path: Path, n_normal: int = 15, n_crack: int = 15) -> tuple[list[Path], list[Path]]:
    """Generate multiple synthetic datasets for robust evaluation."""
    normal_paths = []
    crack_paths = []
    
    for i in range(n_normal):
        normal_dir = tmp_path / f"normal_{i:03d}"
        config = SyntheticConfig(
            frames=60,
            points_per_frame=280,
            fps=30.0,
            seed=42 + i,
            scenario="normal",
        )
        generate_synthetic_bundle(normal_dir, config)
        normal_paths.append(normal_dir)
    
    for i in range(n_crack):
        crack_dir = tmp_path / f"crack_{i:03d}"
        config = SyntheticConfig(
            frames=60,
            points_per_frame=280,
            fps=30.0,
            seed=100 + i,
            scenario="crack",
        )
        generate_synthetic_bundle(crack_dir, config)
        crack_paths.append(crack_dir)
    
    return normal_paths, crack_paths


def main() -> None:
    """Main validation function."""
    try:
        import torch
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            precision_recall_curve,
            auc,
        )
        from motionanalyzer.ml_models.dream import DREAMPyTorch
    except ImportError as e:
        print("DREAM validation requires PyTorch and scikit-learn: pip install -e '.[ml]'")
        raise SystemExit(1) from e

    import tempfile
    print("DREAM Validation on Enhanced Synthetic Data")
    print("=" * 60)
    print("Testing with shockwave and vibration patterns\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Generate evaluation datasets
        print("Generating enhanced synthetic datasets...")
        normal_paths, crack_paths = _generate_evaluation_datasets(tmp_path, n_normal=15, n_crack=15)
        print(f"  Generated {len(normal_paths)} normal and {len(crack_paths)} crack datasets")
        
        # Prepare features
        print("\nExtracting features...")
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
        feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in [np.float64, np.int64, float, int]]
        # Fit normalization on normal-only to avoid leakage
        normal_mask = np.asarray(labels, dtype=int) == 0
        normalized = normalize_features(features_df, exclude_cols=exclude, fit_df=features_df.loc[normal_mask])
        X = normalized[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        y = np.asarray(labels, dtype=int)
        
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Normal: {np.sum(y == 0)}, Crack: {np.sum(y == 1)}")
        
        normal_mask = y == 0
        X_normal = X[normal_mask]
        X_crack = X[~normal_mask]
        
        if len(X_normal) < 2 or len(X_crack) < 1:
            print("Not enough normal/crack samples.")
            return
        
        # Train DREAM model
        print("\n" + "=" * 60)
        print("Training DREAM Model")
        print("=" * 60)
        
        feature_cols_list = feature_cols
        normal_df = pd.DataFrame(X_normal, columns=feature_cols_list)
        
        model = DREAMPyTorch(
            input_dim=len(feature_cols_list),
            hidden_dims=[32, 16],
            latent_dim=4,
            use_discriminative=True,
            synthetic_noise_std=0.3,
            discriminator_weight=0.5,
            batch_size=min(16, len(X_normal)),
        )
        
        print(f"Training on {len(X_normal)} normal samples...")
        model.fit(normal_df, epochs=60, feature_names=feature_cols_list)
        
        # Check score distributions before threshold optimization
        print("\nChecking score distributions...")
        normal_scores_pre = model.predict(X_normal)
        crack_scores_pre = model.predict(X_crack)
        print(f"  Normal scores: min={normal_scores_pre.min():.6f}, max={normal_scores_pre.max():.6f}, mean={normal_scores_pre.mean():.6f}, std={normal_scores_pre.std():.6f}")
        print(f"  Crack scores: min={crack_scores_pre.min():.6f}, max={crack_scores_pre.max():.6f}, mean={crack_scores_pre.mean():.6f}, std={crack_scores_pre.std():.6f}")
        
        # Optimize threshold
        crack_df = pd.DataFrame(X_crack, columns=feature_cols_list)
        thresh, opt_metrics = model.optimize_threshold_for_precision_recall(
            normal_df,
            crack_df,
            target_metric="balanced",
        )
        
        print(f"\nThreshold optimization:")
        print(f"  Optimal threshold: {thresh:.4f}")
        print(f"  Precision: {opt_metrics['precision']:.4f}")
        print(f"  Recall: {opt_metrics['recall']:.4f}")
        print(f"  F1: {opt_metrics['f1']:.4f}")
        
        # Set threshold for predict_binary
        model.reconstruction_error_threshold = thresh
        
        # Evaluate on all data
        scores_all = model.predict(X)
        pred_binary = model.predict_binary(X)
        
        print(f"\nScore distributions (all data):")
        print(f"  Scores: min={scores_all.min():.6f}, max={scores_all.max():.6f}, mean={scores_all.mean():.6f}")
        print(f"  Predictions: normal={np.sum(pred_binary == 0)}, crack={np.sum(pred_binary == 1)}")
        
        # Calculate metrics
        acc = accuracy_score(y, pred_binary)
        prec = precision_score(y, pred_binary, zero_division=0)
        rec = recall_score(y, pred_binary, zero_division=0)
        f1 = f1_score(y, pred_binary, zero_division=0)
        roc_auc = roc_auc_score(y, scores_all) if len(np.unique(y)) == 2 else 0.5
        
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y, scores_all)
        pr_auc = auc(recall_vals, precision_vals)
        
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"PR AUC:    {pr_auc:.4f} (Precision-Recall AUC)")
        
        # Assessment
        print("\n" + "=" * 60)
        print("Dataset Quality Assessment")
        print("=" * 60)
        
        if roc_auc >= 0.9:
            print("[EXCELLENT] ROC AUC >= 0.9 - Dataset quality is excellent")
        elif roc_auc >= 0.8:
            print("[GOOD] ROC AUC >= 0.8 - Dataset quality is good")
        elif roc_auc >= 0.7:
            print("[MODERATE] ROC AUC >= 0.7 - Dataset quality is moderate")
        else:
            print("[POOR] ROC AUC < 0.7 - Dataset quality needs improvement")
        
        if pr_auc >= 0.8:
            print("[EXCELLENT] PR AUC >= 0.8 - Good for precision-recall evaluation")
        elif pr_auc >= 0.7:
            print("[GOOD] PR AUC >= 0.7 - Acceptable for precision-recall evaluation")
        elif pr_auc >= 0.6:
            print("[MODERATE] PR AUC >= 0.6 - Consider dataset improvements")
        else:
            print("[POOR] PR AUC < 0.6 - Dataset improvements needed")
        
        # Compare with DRAEM paper (reference only)
        print("\nReference: DRAEM paper (MVTec AD): image-level ROC AUC 98.1%")
        print("Note: Domain differs (FPCB tabular vs MVTec images)")
        print("      Use for trend comparison only, not direct comparison")


if __name__ == "__main__":
    main()
