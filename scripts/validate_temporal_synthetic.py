"""
Validate Temporal (LSTM/GRU) model on enhanced synthetic data.

Tests temporal anomaly detection with sequence-based reconstruction error.
Reports ROC AUC and PR AUC for comprehensive evaluation.
"""

from __future__ import annotations

import json
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
        from motionanalyzer.ml_models.dream_temporal import TemporalAnomalyDetector
    except ImportError as e:
        print("Temporal validation requires PyTorch and scikit-learn: pip install -e '.[ml]'")
        raise SystemExit(1) from e

    import tempfile
    print("Temporal Model Validation on Enhanced Synthetic Data")
    print("=" * 60)
    print("Testing with sequence-based reconstruction error\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Generate evaluation datasets
        print("Generating enhanced synthetic datasets...")
        normal_paths, crack_paths = _generate_evaluation_datasets(tmp_path, n_normal=15, n_crack=15)
        print(f"  Generated {len(normal_paths)} normal and {len(crack_paths)} crack datasets")

        # Prepare features (per-frame for temporal model)
        print("\nExtracting per-frame features...")
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
        feature_cols = [
            c
            for c in features_df.columns
            if c not in exclude and features_df[c].dtype in [np.float64, np.int64, float, int]
        ]

        # Normalize features (fit on normal-only to avoid leakage)
        normal_mask = np.asarray(labels, dtype=int) == 0
        normalized = normalize_features(features_df, exclude_cols=exclude, fit_df=features_df.loc[normal_mask])

        print(f"  Total frames: {len(features_df)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Normal: {np.sum(labels == 0)}, Crack: {np.sum(labels == 1)}")

        normal_df = normalized.loc[normal_mask].copy()
        crack_df = normalized.loc[~normal_mask].copy()

        if len(normal_df) < 2 or len(crack_df) < 1:
            print("Not enough normal/crack samples.")
            return

        # Train Temporal model
        print("\n" + "=" * 60)
        print("Training Temporal Model (LSTM)")
        print("=" * 60)

        sequence_length = 10  # Optimal window length per literature
        model = TemporalAnomalyDetector(
            feature_dim=len(feature_cols),
            sequence_length=sequence_length,
            hidden_dim=64,
            num_layers=2,
            cell_type="LSTM",
            batch_size=min(16, len(normal_df) // sequence_length),
        )

        print(f"Training on {len(normal_df)} normal frames (sequence length: {sequence_length})...")
        model.fit(normal_df, feature_cols, epochs=50)

        # Set threshold
        model.set_threshold_from_normal(normal_df, feature_cols, percentile=95.0)
        print(f"Threshold set to: {model.reconstruction_threshold:.4f}")

        # Evaluate on all data
        print("\nEvaluating on all data...")
        all_scores_df = model.predict(normalized, feature_cols, aggregation="max")

        # Merge scores back with labels
        scores_with_labels = normalized[["dataset_path", "frame", "label"]].merge(
            all_scores_df, on=["dataset_path", "frame"], how="left"
        )
        scores_with_labels = scores_with_labels.fillna(0.0)

        scores = scores_with_labels["anomaly_score"].values
        y_true = scores_with_labels["label"].values

        # Binary predictions
        pred_binary = (scores > model.reconstruction_threshold).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true, pred_binary)
        prec = precision_score(y_true, pred_binary, zero_division=0)
        rec = recall_score(y_true, pred_binary, zero_division=0)
        f1 = f1_score(y_true, pred_binary, zero_division=0)
        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else 0.5

        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
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

        # Save results
        results = {
            "model": "temporal_lstm",
            "sequence_length": sequence_length,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
            },
            "threshold": float(model.reconstruction_threshold),
            "n_normal": int(np.sum(y_true == 0)),
            "n_crack": int(np.sum(y_true == 1)),
        }

        results_path = tmp_path / "temporal_validation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
