"""
Ensemble anomaly detection combining DREAM and PatchCore.

Implements multiple ensemble strategies:
- Weighted Average: Combine scores with optimized weights (α)
- Maximum: Take maximum score (recall-oriented)
- Stacking: Meta-classifier on base model predictions

References:
- Ensemble anomaly detection: Weighted averaging and greedy selection strategies
- PatchCore (Roth et al., CVPR 2022): Memory-based anomaly detection
- DRAEM (Zavrtanik et al., ICCV 2021): Discriminative reconstruction embedding
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class EnsembleStrategy(str, Enum):
    """Ensemble combination strategy."""

    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    STACKING = "stacking"


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining DREAM and PatchCore.

    Combines predictions from multiple base models to improve robustness and performance.
    """

    def __init__(
        self,
        dream_model: Any,
        patchcore_model: Any,
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        dream_weight: float = 0.5,
        patchcore_weight: float = 0.5,
    ) -> None:
        """
        Initialize ensemble detector.

        Args:
            dream_model: Trained DREAM model (DREAMPyTorch instance)
            patchcore_model: Trained PatchCore model (PatchCoreScikitLearn instance)
            strategy: Ensemble combination strategy
            dream_weight: Weight for DREAM scores (used in weighted_average)
            patchcore_weight: Weight for PatchCore scores (used in weighted_average)
        """
        self.dream_model = dream_model
        self.patchcore_model = patchcore_model
        self.strategy = strategy
        self.dream_weight = dream_weight
        self.patchcore_weight = patchcore_weight
        self.ensemble_threshold: Optional[float] = None
        self._meta_classifier: Optional[Any] = None  # For stacking strategy

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict ensemble anomaly scores.

        Args:
            data: Test data (same format as base models)

        Returns:
            Ensemble anomaly scores (higher = more anomalous). Shape: (n_samples,)
        """
        dream_scores = self.dream_model.predict(data)
        patchcore_scores = self.patchcore_model.predict(data)

        # Normalize scores to [0, 1] range for combination
        dream_scores_norm = self._normalize_scores(dream_scores)
        patchcore_scores_norm = self._normalize_scores(patchcore_scores)

        if self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            # Weighted average: α * dream + (1-α) * patchcore
            total_weight = self.dream_weight + self.patchcore_weight
            if total_weight > 0:
                ensemble_scores = (
                    self.dream_weight * dream_scores_norm + self.patchcore_weight * patchcore_scores_norm
                ) / total_weight
            else:
                ensemble_scores = (dream_scores_norm + patchcore_scores_norm) / 2.0
        elif self.strategy == EnsembleStrategy.MAXIMUM:
            # Maximum: Take max score (recall-oriented)
            ensemble_scores = np.maximum(dream_scores_norm, patchcore_scores_norm)
        elif self.strategy == EnsembleStrategy.STACKING:
            # Stacking: Meta-classifier on base predictions
            if self._meta_classifier is None:
                # Fallback to weighted average if meta-classifier not trained
                total_weight = self.dream_weight + self.patchcore_weight
                if total_weight > 0:
                    ensemble_scores = (
                        self.dream_weight * dream_scores_norm + self.patchcore_weight * patchcore_scores_norm
                    ) / total_weight
                else:
                    ensemble_scores = (dream_scores_norm + patchcore_scores_norm) / 2.0
            else:
                # Stack features: [dream_score, patchcore_score]
                stack_features = np.column_stack([dream_scores_norm, patchcore_scores_norm])
                ensemble_scores = self._meta_classifier.predict_proba(stack_features)[:, 1]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return ensemble_scores

    def predict_binary(self, data: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels (0=normal, 1=anomaly).

        Args:
            data: Test data
            threshold: Anomaly threshold (if None, use self.ensemble_threshold)

        Returns:
            Binary labels (0 or 1). Shape: (n_samples,)
        """
        scores = self.predict(data)
        thresh = threshold if threshold is not None else self.ensemble_threshold
        if thresh is None:
            raise ValueError("Threshold not set. Call set_threshold_from_normal() or optimize_threshold() first.")
        return (scores > thresh).astype(int)

    def set_threshold_from_normal(self, normal_data: pd.DataFrame | np.ndarray, percentile: float = 95.0) -> None:
        """
        Set ensemble threshold from normal data scores.

        Args:
            normal_data: Normal data (validation set)
            percentile: Percentile for threshold (e.g., 95 = p95)
        """
        scores = self.predict(normal_data)
        self.ensemble_threshold = float(np.percentile(scores, percentile))

    def optimize_threshold_for_precision_recall(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        anomaly_data: pd.DataFrame | np.ndarray,
        target_metric: str = "f1",
    ) -> tuple[float, dict[str, float]]:
        """
        Optimize threshold to maximize F1 or balance precision-recall.

        Args:
            normal_data: Normal validation data
            anomaly_data: Anomaly validation data
            target_metric: "f1", "precision", "recall", or "balanced" (F1 with recall >= 0.7)

        Returns:
            (optimal_threshold, metrics_dict) where metrics_dict has precision, recall, f1, accuracy
        """
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        except ImportError:
            raise ImportError("Threshold optimization requires scikit-learn")

        normal_scores = self.predict(normal_data)
        anomaly_scores = self.predict(anomaly_data)
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

        # Try thresholds from p50 to p99.9 of normal scores
        thresholds = np.percentile(normal_scores, np.linspace(50, 99.9, 100))
        best_metric = -1.0
        best_threshold = float(np.percentile(normal_scores, 95.0))
        best_metrics = {}

        for thresh in thresholds:
            pred = (all_scores > thresh).astype(int)
            prec = precision_score(all_labels, pred, zero_division=0)
            rec = recall_score(all_labels, pred, zero_division=0)
            f1 = f1_score(all_labels, pred, zero_division=0)
            acc = accuracy_score(all_labels, pred)

            if target_metric == "f1":
                metric_val = f1
            elif target_metric == "precision":
                metric_val = prec
            elif target_metric == "recall":
                metric_val = rec
            elif target_metric == "balanced":
                # F1 with recall >= 0.7 constraint
                metric_val = f1 if rec >= 0.7 else 0.0
            else:
                raise ValueError(f"Unknown target_metric: {target_metric}")

            if metric_val > best_metric:
                best_metric = metric_val
                best_threshold = thresh
                best_metrics = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

        self.ensemble_threshold = best_threshold
        return best_threshold, best_metrics

    def optimize_weights(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        anomaly_data: pd.DataFrame | np.ndarray,
        target_metric: str = "f1",
    ) -> tuple[float, float, dict[str, float]]:
        """
        Optimize ensemble weights (α) for weighted_average strategy.

        Searches over α ∈ [0, 1] to find optimal dream_weight = α, patchcore_weight = 1-α.

        Args:
            normal_data: Normal validation data
            anomaly_data: Anomaly validation data
            target_metric: "f1", "precision", "recall", or "balanced"

        Returns:
            (optimal_dream_weight, optimal_patchcore_weight, best_metrics)
        """
        if self.strategy != EnsembleStrategy.WEIGHTED_AVERAGE:
            raise ValueError("Weight optimization only applies to weighted_average strategy")

        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        except ImportError:
            raise ImportError("Weight optimization requires scikit-learn")

        # Try α values from 0.0 to 1.0 in steps of 0.1
        alpha_values = np.linspace(0.0, 1.0, 11)
        best_metric = -1.0
        best_alpha = 0.5
        best_metrics = {}

        original_dream_weight = self.dream_weight
        original_patchcore_weight = self.patchcore_weight

        for alpha in alpha_values:
            self.dream_weight = alpha
            self.patchcore_weight = 1.0 - alpha

            # Set threshold from normal data
            self.set_threshold_from_normal(normal_data, percentile=95.0)

            # Evaluate on anomaly data
            normal_scores = self.predict(normal_data)
            anomaly_scores = self.predict(anomaly_data)
            all_scores = np.concatenate([normal_scores, anomaly_scores])
            all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])

            thresh = self.ensemble_threshold
            pred = (all_scores > thresh).astype(int)
            prec = precision_score(all_labels, pred, zero_division=0)
            rec = recall_score(all_labels, pred, zero_division=0)
            f1 = f1_score(all_labels, pred, zero_division=0)
            acc = accuracy_score(all_labels, pred)

            if target_metric == "f1":
                metric_val = f1
            elif target_metric == "precision":
                metric_val = prec
            elif target_metric == "recall":
                metric_val = rec
            elif target_metric == "balanced":
                metric_val = f1 if rec >= 0.7 else 0.0
            else:
                raise ValueError(f"Unknown target_metric: {target_metric}")

            if metric_val > best_metric:
                best_metric = metric_val
                best_alpha = alpha
                best_metrics = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

        # Restore optimal weights
        self.dream_weight = best_alpha
        self.patchcore_weight = 1.0 - best_alpha

        # Restore original weights if optimization failed
        if best_metric <= 0:
            self.dream_weight = original_dream_weight
            self.patchcore_weight = original_patchcore_weight

        return self.dream_weight, self.patchcore_weight, best_metrics

    def fit_stacking(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        anomaly_data: pd.DataFrame | np.ndarray,
    ) -> None:
        """
        Train meta-classifier for stacking strategy.

        Args:
            normal_data: Normal training data
            anomaly_data: Anomaly training data
        """
        if self.strategy != EnsembleStrategy.STACKING:
            raise ValueError("fit_stacking() only applies to stacking strategy")

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("Stacking requires scikit-learn")

        # Get base model predictions
        dream_normal = self.dream_model.predict(normal_data)
        dream_anomaly = self.dream_model.predict(anomaly_data)
        patchcore_normal = self.patchcore_model.predict(normal_data)
        patchcore_anomaly = self.patchcore_model.predict(anomaly_data)

        # Normalize scores
        dream_normal_norm = self._normalize_scores(dream_normal)
        dream_anomaly_norm = self._normalize_scores(dream_anomaly)
        patchcore_normal_norm = self._normalize_scores(patchcore_normal)
        patchcore_anomaly_norm = self._normalize_scores(patchcore_anomaly)

        # Stack features: [dream_score, patchcore_score]
        X_normal = np.column_stack([dream_normal_norm, patchcore_normal_norm])
        X_anomaly = np.column_stack([dream_anomaly_norm, patchcore_anomaly_norm])
        X = np.vstack([X_normal, X_anomaly])
        y = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])

        # Train meta-classifier
        self._meta_classifier = LogisticRegression(random_state=42, max_iter=1000)
        self._meta_classifier.fit(X, y)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling."""
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        return np.zeros_like(scores)

    def save(self, path: Path) -> None:
        """
        Save ensemble configuration (not base models).

        Args:
            path: Path to save ensemble config JSON
        """
        import json

        config = {
            "strategy": self.strategy.value,
            "dream_weight": self.dream_weight,
            "patchcore_weight": self.patchcore_weight,
            "ensemble_threshold": self.ensemble_threshold,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: Path) -> None:
        """
        Load ensemble configuration (base models must be loaded separately).

        Args:
            path: Path to ensemble config JSON
        """
        import json

        with open(path, "r") as f:
            config = json.load(f)
        self.strategy = EnsembleStrategy(config["strategy"])
        self.dream_weight = config.get("dream_weight", 0.5)
        self.patchcore_weight = config.get("patchcore_weight", 0.5)
        self.ensemble_threshold = config.get("ensemble_threshold")
