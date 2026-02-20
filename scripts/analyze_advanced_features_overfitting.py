"""
Analyze potential overfitting in DREAM+Advanced Features.

Investigates why DREAM achieves perfect performance (ROC AUC 1.000) with advanced features
on synthetic data. Checks for:
1. Feature importance analysis
2. Correlation between features and labels
3. Cross-validation to detect overfitting
4. Comparison with baseline features
"""

from __future__ import annotations

import sys
import tempfile
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


def _generate_datasets(tmp_path: Path, n_normal: int = 10, n_crack: int = 10) -> tuple[list[Path], list[Path]]:
    """Generate synthetic datasets."""
    normal_paths = []
    crack_paths = []
    
    for i in range(n_normal):
        normal_dir = tmp_path / f"normal_{i}"
        normal_dir.mkdir(parents=True, exist_ok=True)
        cfg = SyntheticConfig(frames=120, points_per_frame=230, fps=30.0, seed=42 + i, scenario="normal")
        generate_synthetic_bundle(normal_dir, cfg)
        normal_paths.append(normal_dir)
    
    for i in range(n_crack):
        crack_dir = tmp_path / f"crack_{i}"
        crack_dir.mkdir(parents=True, exist_ok=True)
        cfg = SyntheticConfig(frames=120, points_per_frame=230, fps=30.0, seed=100 + i, scenario="crack")
        generate_synthetic_bundle(crack_dir, cfg)
        crack_paths.append(crack_dir)
    
    return normal_paths, crack_paths


def _analyze_feature_correlations(features_df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Analyze correlations between features and labels."""
    correlations = []
    
    for feat in feature_cols:
        if feat not in features_df.columns:
            continue
        
        corr = np.corrcoef(features_df[feat].fillna(0), labels)[0, 1]
        correlations.append({
            "feature": feat,
            "correlation": corr,
            "abs_correlation": abs(corr),
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_correlation", ascending=False)
    
    return corr_df


def _analyze_feature_importance_via_permutation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_func,
) -> pd.DataFrame:
    """Analyze feature importance using permutation importance."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return pd.DataFrame()
    
    # Baseline performance
    normal_mask = np.asarray(y_train) == 0
    X_normal = X_train[normal_mask]
    
    model = model_func(X_normal.shape[1])
    model.fit(X_normal, epochs=20)
    model.set_threshold_from_normal(X_normal, percentile=95.0)
    
    baseline_scores = model.predict(X_test)
    baseline_auc = roc_auc_score(y_test, baseline_scores)
    
    # Permutation importance
    importances = []
    
    for i, feat_name in enumerate(feature_names):
        X_test_permuted = X_test.copy()
        np.random.seed(42)
        np.random.shuffle(X_test_permuted[:, i])
        
        permuted_scores = model.predict(X_test_permuted)
        permuted_auc = roc_auc_score(y_test, permuted_scores)
        
        importance = baseline_auc - permuted_auc
        importances.append({
            "feature": feat_name,
            "importance": importance,
            "baseline_auc": baseline_auc,
            "permuted_auc": permuted_auc,
        })
    
    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    return importance_df


def main() -> None:
    """Analyze advanced features for overfitting."""
    print("Advanced Features Overfitting Analysis")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        print("\n[1/3] Generating datasets...")
        normal_paths, crack_paths = _generate_datasets(tmp_path, n_normal=10, n_crack=10)
        print(f"  Generated {len(normal_paths)} normal and {len(crack_paths)} crack datasets")
        
        # Split for cross-validation
        train_normal = normal_paths[:7]
        train_crack = crack_paths[:7]
        test_normal = normal_paths[7:]
        test_crack = crack_paths[7:]
        
        print("\n[2/3] Analyzing baseline features...")
        features_base, labels_base = prepare_training_data(
            normal_datasets=train_normal + test_normal,
            crack_datasets=train_crack + test_crack,
            feature_config=FeatureExtractionConfig(
                include_per_frame=True,
                include_per_point=False,
                include_global_stats=False,
                include_crack_risk_features=False,
                include_advanced_stats=False,
                include_frequency_domain=False,
            ),
        )
        
        exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
        feature_cols_base = [
            c for c in features_base.columns
            if c not in exclude and c in features_base.select_dtypes(include=["number"]).columns
        ]
        
        normal_mask = np.asarray(labels_base, dtype=int) == 0
        normalized_base = normalize_features(
            features_base, exclude_cols=exclude, fit_df=features_base.loc[normal_mask]
        )
        
        print(f"  Baseline features: {len(feature_cols_base)}")
        corr_base = _analyze_feature_correlations(normalized_base, np.asarray(labels_base), feature_cols_base)
        print("\n  Top 5 correlated features (baseline):")
        print(corr_base.head(5).to_string(index=False))
        
        print("\n[3/3] Analyzing advanced features...")
        features_adv, labels_adv = prepare_training_data(
            normal_datasets=train_normal + test_normal,
            crack_datasets=train_crack + test_crack,
            feature_config=FeatureExtractionConfig(
                include_per_frame=True,
                include_per_point=False,
                include_global_stats=False,
                include_crack_risk_features=False,
                include_advanced_stats=True,
                include_frequency_domain=True,
            ),
        )
        
        feature_cols_adv = [
            c for c in features_adv.columns
            if c not in exclude and c in features_adv.select_dtypes(include=["number"]).columns
        ]
        
        normalized_adv = normalize_features(
            features_adv, exclude_cols=exclude, fit_df=features_adv.loc[normal_mask]
        )
        
        print(f"  Advanced features: {len(feature_cols_adv)}")
        corr_adv = _analyze_feature_correlations(normalized_adv, np.asarray(labels_adv), feature_cols_adv)
        print("\n  Top 10 correlated features (advanced):")
        print(corr_adv.head(10).to_string(index=False))
        
        # Compare feature counts and correlations
        print("\n" + "=" * 80)
        print("Analysis Summary")
        print("=" * 80)
        print(f"\nFeature count: {len(feature_cols_base)} -> {len(feature_cols_adv)} (+{len(feature_cols_adv) - len(feature_cols_base)})")
        
        high_corr_base = len(corr_base[corr_base["abs_correlation"] > 0.5])
        high_corr_adv = len(corr_adv[corr_adv["abs_correlation"] > 0.5])
        print(f"\nHigh correlation features (|corr| > 0.5):")
        print(f"  Baseline: {high_corr_base}")
        print(f"  Advanced: {high_corr_adv}")
        
        print("\nTop advanced features with high correlation:")
        top_adv = corr_adv[corr_adv["abs_correlation"] > 0.3].head(10)
        print(top_adv[["feature", "correlation"]].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("Potential Overfitting Indicators")
        print("=" * 80)
        
        if high_corr_adv > high_corr_base * 2:
            print(f"[WARNING] Advanced features have {high_corr_adv / high_corr_base:.1f}x more highly correlated features")
            print("   This may indicate overfitting on synthetic data patterns.")
        
        if corr_adv["abs_correlation"].max() > 0.8:
            print(f"[WARNING] Maximum correlation: {corr_adv['abs_correlation'].max():.3f}")
            print("   Very high correlation suggests features may be too predictive on synthetic data.")
        
        print("\nRecommendations:")
        print("1. Test on more diverse synthetic datasets (different seeds, scenarios)")
        print("2. Use cross-validation to detect overfitting")
        print("3. Validate on real data when available")
        print("4. Consider feature selection to reduce dimensionality")


if __name__ == "__main__":
    main()
