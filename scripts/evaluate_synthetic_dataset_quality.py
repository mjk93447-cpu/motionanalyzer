"""
Evaluate synthetic dataset quality for ML model training and validation.

Assesses whether synthetic datasets are suitable for precision-recall evaluation
by checking:
1. Class balance (normal vs crack)
2. Feature separability
3. Temporal patterns (shockwave, vibration)
4. Statistical properties

Reference: ADBench (Han et al., 2022), TimeSeAD evaluation protocols
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

from motionanalyzer.analysis import run_analysis
from motionanalyzer.auto_optimize import (
    FeatureExtractionConfig,
    normalize_features,
    prepare_training_data,
)
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def _generate_evaluation_datasets(tmp_path: Path, n_normal: int = 10, n_crack: int = 10) -> tuple[list[Path], list[Path]]:
    """Generate multiple synthetic datasets for evaluation."""
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


def _evaluate_class_balance(features_df: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """Evaluate class balance in dataset."""
    n_normal = np.sum(labels == 0)
    n_crack = np.sum(labels == 1)
    total = len(labels)
    
    return {
        "n_normal": n_normal,
        "n_crack": n_crack,
        "normal_ratio": n_normal / total if total > 0 else 0.0,
        "crack_ratio": n_crack / total if total > 0 else 0.0,
        "balance_ratio": min(n_normal, n_crack) / max(n_normal, n_crack) if max(n_normal, n_crack) > 0 else 0.0,
    }


def _evaluate_feature_separability(features_df: pd.DataFrame, labels: np.ndarray) -> dict[str, any]:
    """Evaluate feature separability between normal and crack classes."""
    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [c for c in features_df.columns if c not in exclude and features_df[c].dtype in [np.float64, np.int64, float, int]]
    
    if not feature_cols:
        return {"separability_score": 0.0, "top_separable_features": []}
    
    normal_mask = labels == 0
    crack_mask = labels == 1
    
    if not (normal_mask.any() and crack_mask.any()):
        return {"separability_score": 0.0, "top_separable_features": []}
    
    # Calculate separability for each feature (effect size: Cohen's d)
    separability_scores = {}
    for col in feature_cols:
        normal_vals = features_df.loc[normal_mask, col].values
        crack_vals = features_df.loc[crack_mask, col].values
        
        if len(normal_vals) == 0 or len(crack_vals) == 0:
            continue
        
        mean_normal = np.mean(normal_vals)
        mean_crack = np.mean(crack_vals)
        std_normal = np.std(normal_vals)
        std_crack = np.std(crack_vals)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((std_normal**2 + std_crack**2) / 2)
        
        if pooled_std > 1e-6:
            cohens_d = abs(mean_crack - mean_normal) / pooled_std
            separability_scores[col] = cohens_d
    
    # Overall separability (mean of top features)
    if separability_scores:
        scores_sorted = sorted(separability_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = scores_sorted[:5]
        overall_separability = np.mean([s[1] for s in scores_sorted])
        
        return {
            "separability_score": overall_separability,
            "top_separable_features": top_features,
        }
    
    return {"separability_score": 0.0, "top_separable_features": []}


def _evaluate_temporal_patterns(normal_paths: list[Path], crack_paths: list[Path]) -> dict[str, any]:
    """Evaluate temporal patterns (shockwave, vibration) in crack scenarios."""
    results = {
        "shockwave_detected": False,
        "vibration_detected": False,
        "crack_frame_consistency": 0.0,
    }
    
    if not crack_paths:
        return results
    
    # Analyze first crack dataset
    crack_path = crack_paths[0]
    output_dir = crack_path.parent / f"{crack_path.name}_analysis"
    run_analysis(crack_path, output_dir, fps=30.0)
    
    vectors_path = output_dir / "vectors.csv"
    if not vectors_path.exists():
        return results
    
    vectors = pd.read_csv(vectors_path)
    frame_accel = vectors.groupby("frame")["acceleration"].agg(["mean", "max", "std"]).reset_index()
    
    # Expected crack frame (crack_center_ratio ~0.72, frames=60)
    expected_crack_frame = int(0.72 * 59)  # ~43
    
    # Check for acceleration spike (shockwave)
    crack_window = frame_accel[
        (frame_accel["frame"] >= expected_crack_frame - 3) &
        (frame_accel["frame"] <= expected_crack_frame + 5)
    ]
    if len(crack_window) > 0:
        spike_max = crack_window["max"].max()
        baseline_max = frame_accel[frame_accel["frame"] < expected_crack_frame - 3]["max"].mean()
        if spike_max > baseline_max * 1.5:
            results["shockwave_detected"] = True
    
    # Check for vibration (increased std after crack)
    before_std = frame_accel[frame_accel["frame"] < expected_crack_frame]["std"].mean()
    after_std = frame_accel[
        (frame_accel["frame"] >= expected_crack_frame) &
        (frame_accel["frame"] <= expected_crack_frame + 15)
    ]["std"].mean()
    if after_std > before_std * 1.2:
        results["vibration_detected"] = True
    
    return results


def main() -> None:
    """Main evaluation function."""
    import tempfile
    
    print("Synthetic Dataset Quality Evaluation")
    print("=" * 60)
    print("Assessing dataset suitability for precision-recall evaluation\n")
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Generate evaluation datasets
        print("Generating evaluation datasets...")
        normal_paths, crack_paths = _generate_evaluation_datasets(tmp_path, n_normal=10, n_crack=10)
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
                # ML validation should avoid Physics-derived crack_risk features (leakage risk)
                include_crack_risk_features=False,
            ),
        )
        print(f"  Total samples: {len(features_df)}")
        
        # Evaluate class balance
        print("\n" + "=" * 60)
        print("1. Class Balance Evaluation")
        print("=" * 60)
        balance = _evaluate_class_balance(features_df, labels)
        print(f"  Normal samples: {balance['n_normal']}")
        print(f"  Crack samples: {balance['n_crack']}")
        print(f"  Normal ratio: {balance['normal_ratio']:.2%}")
        print(f"  Crack ratio: {balance['crack_ratio']:.2%}")
        print(f"  Balance ratio: {balance['balance_ratio']:.2f}")
        
        if balance['balance_ratio'] < 0.3:
            print("  [WARN] Severe class imbalance - may affect precision-recall evaluation")
        elif balance['balance_ratio'] < 0.5:
            print("  [INFO] Moderate class imbalance - consider using AUCPR in addition to ROC AUC")
        else:
            print("  [OK] Good class balance")
        
        # Evaluate feature separability
        print("\n" + "=" * 60)
        print("2. Feature Separability Evaluation")
        print("=" * 60)
        separability = _evaluate_feature_separability(features_df, labels)
        print(f"  Overall separability score (mean Cohen's d): {separability['separability_score']:.3f}")
        
        if separability['top_separable_features']:
            print("\n  Top 5 separable features:")
            for feat, score in separability['top_separable_features']:
                print(f"    {feat}: {score:.3f}")
        
        if separability['separability_score'] < 0.5:
            print("  [WARN] Low separability - features may not distinguish normal/crack well")
        elif separability['separability_score'] < 1.0:
            print("  [INFO] Moderate separability - ML models should be able to learn patterns")
        else:
            print("  [OK] High separability - good for ML model training")
        
        # Evaluate temporal patterns
        print("\n" + "=" * 60)
        print("3. Temporal Pattern Evaluation")
        print("=" * 60)
        temporal = _evaluate_temporal_patterns(normal_paths, crack_paths)
        print(f"  Shockwave detected: {temporal['shockwave_detected']}")
        print(f"  Vibration detected: {temporal['vibration_detected']}")
        
        if temporal['shockwave_detected'] and temporal['vibration_detected']:
            print("  [OK] Enhanced physics patterns (shockwave + vibration) detected")
        elif temporal['shockwave_detected'] or temporal['vibration_detected']:
            print("  [INFO] Some physics patterns detected")
        else:
            print("  [WARN] Physics patterns may be weak - consider enhancing synthetic data")
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("Overall Dataset Quality Assessment")
        print("=" * 60)
        
        quality_score = 0.0
        quality_factors = []
        
        # Class balance (0-30 points)
        if balance['balance_ratio'] >= 0.5:
            quality_score += 30
            quality_factors.append("Good class balance")
        elif balance['balance_ratio'] >= 0.3:
            quality_score += 15
            quality_factors.append("Moderate class balance")
        else:
            quality_factors.append("Poor class balance")
        
        # Feature separability (0-40 points)
        if separability['separability_score'] >= 1.0:
            quality_score += 40
            quality_factors.append("High feature separability")
        elif separability['separability_score'] >= 0.5:
            quality_score += 20
            quality_factors.append("Moderate feature separability")
        else:
            quality_factors.append("Low feature separability")
        
        # Temporal patterns (0-30 points)
        if temporal['shockwave_detected'] and temporal['vibration_detected']:
            quality_score += 30
            quality_factors.append("Enhanced physics patterns")
        elif temporal['shockwave_detected'] or temporal['vibration_detected']:
            quality_score += 15
            quality_factors.append("Some physics patterns")
        else:
            quality_factors.append("Weak physics patterns")
        
        print(f"\nQuality Score: {quality_score}/100")
        print("\nQuality Factors:")
        for factor in quality_factors:
            print(f"  - {factor}")
        
        if quality_score >= 80:
            print("\n[OK] Dataset quality is EXCELLENT for precision-recall evaluation")
        elif quality_score >= 60:
            print("\n[INFO] Dataset quality is GOOD for precision-recall evaluation")
        elif quality_score >= 40:
            print("\n[WARN] Dataset quality is MODERATE - consider improvements")
        else:
            print("\n[WARN] Dataset quality is POOR - improvements needed before ML evaluation")
        
        print("\nRecommendations:")
        if balance['balance_ratio'] < 0.5:
            print("  - Use AUCPR (Precision-Recall AUC) in addition to ROC AUC")
            print("  - Consider stratified sampling or class weighting")
        if separability['separability_score'] < 0.5:
            print("  - Consider feature engineering or advanced feature extraction")
            print("  - Review synthetic data generation parameters")
        if not (temporal['shockwave_detected'] and temporal['vibration_detected']):
            print("  - Enhance synthetic data with stronger physics patterns")
            print("  - Increase shockwave_amplitude or vibration_frequency_hz")


if __name__ == "__main__":
    main()
