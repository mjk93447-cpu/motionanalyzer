"""
PatchCore-inspired anomaly detection for FPCB crack detection.

Memory bank-based few-shot anomaly detection: builds a memory bank of normal
feature vectors, detects anomalies via distance to nearest normal (k-NN).

Reference:
- Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022).
  "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. arXiv:2106.08265.

Implementation note: The original PatchCore uses CNN patch features and greedy
coreset sampling; this implementation uses tabular/time-series features and
random coreset sampling, suitable for FPCB bending data. See docs/PHASE2_ML_REVIEW.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class PatchCoreAnomalyDetector(ABC):
    """
    PatchCore anomaly detector for FPCB crack detection.

    Builds a memory bank of normal feature patches, detects anomalies via
    distance to nearest normal patch in feature space.
    """

    def __init__(
        self,
        feature_dim: int,
        coreset_size: int = 1000,
        k_neighbors: int = 1,
    ) -> None:
        """
        Initialize PatchCore model.

        Args:
            feature_dim: Feature vector dimension
            coreset_size: Size of coreset (memory bank subset)
            k_neighbors: Number of nearest neighbors for distance computation
        """
        self.feature_dim = feature_dim
        self.coreset_size = coreset_size
        self.k_neighbors = k_neighbors
        self.memory_bank: Optional[np.ndarray] = None
        self.is_trained = False
        self.anomaly_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, normal_data: pd.DataFrame | np.ndarray) -> None:
        """
        Build memory bank from normal data.

        Args:
            normal_data: Normal FPCB bending data (feature vectors or vectors.csv-like)
        """
        raise NotImplementedError("Subclass must implement fit()")

    @abstractmethod
    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (distance to nearest normal patch).

        Args:
            data: Test data (same format as fit)

        Returns:
            Anomaly scores (higher = more anomalous). Shape: (n_samples,)
        """
        raise NotImplementedError("Subclass must implement predict()")

    @abstractmethod
    def predict_binary(self, data: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels (0=normal, 1=anomaly).

        Args:
            data: Test data
            threshold: Anomaly threshold (if None, use self.anomaly_threshold)

        Returns:
            Binary labels (0 or 1). Shape: (n_samples,)
        """
        raise NotImplementedError("Subclass must implement predict_binary()")

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save memory bank and config."""
        raise NotImplementedError("Subclass must implement save()")

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load memory bank and config."""
        raise NotImplementedError("Subclass must implement load()")

    def set_threshold_from_normal(self, normal_data: pd.DataFrame | np.ndarray, percentile: float = 95.0) -> None:
        """
        Set anomaly threshold from normal data distances.

        Args:
            normal_data: Normal data (validation set)
            percentile: Percentile for threshold (e.g., 95 = p95)
        """
        scores = self.predict(normal_data)
        self.anomaly_threshold = float(np.percentile(scores, percentile))


def _to_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert DataFrame or array to float32 2D array."""
    if isinstance(data, pd.DataFrame):
        return data.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class PatchCoreScikitLearn(PatchCoreAnomalyDetector):
    """
    Scikit-learn based PatchCore: memory bank of normal features + NearestNeighbors.

    Coreset: random sample of size coreset_size from normal data (or all if smaller).
    Anomaly score = mean distance to k_neighbors in memory bank (higher = more anomalous).
    """

    def __init__(
        self,
        feature_dim: int,
        coreset_size: int = 1000,
        k_neighbors: int = 1,
    ) -> None:
        super().__init__(feature_dim, coreset_size, k_neighbors)
        self._nn: Optional[Any] = None  # sklearn NearestNeighbors

    def fit(self, normal_data: pd.DataFrame | np.ndarray) -> None:
        """Build memory bank (coreset) and fit NearestNeighbors."""
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as e:
            raise ImportError(
                "PatchCore requires scikit-learn. Install with: pip install -e '.[ml]'"
            ) from e

        X = _to_array(normal_data)
        if X.shape[1] != self.feature_dim:
            self.feature_dim = X.shape[1]

        n = len(X)
        if n <= self.coreset_size:
            self.memory_bank = X
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.coreset_size, replace=False)
            self.memory_bank = X[idx]

        self._nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(self.memory_bank)),
            algorithm="auto",
            metric="minkowski",
            p=2,
        )
        self._nn.fit(self.memory_bank)
        self.is_trained = True

    def predict(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Anomaly score = mean distance to k_neighbors in memory bank."""
        if not self.is_trained or self._nn is None:
            raise ValueError("Model not trained. Call fit() first.")
        X = _to_array(data)
        distances, _ = self._nn.kneighbors(X)
        # Mean distance to k neighbors (higher = more anomalous)
        scores = np.mean(distances, axis=1).astype(np.float64)
        return scores

    def predict_binary(self, data: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary labels: 1 if score > threshold else 0."""
        scores = self.predict(data)
        t = threshold if threshold is not None else self.anomaly_threshold
        if t is None:
            raise ValueError("Threshold not set. Call set_threshold_from_normal() or pass threshold.")
        return (scores > t).astype(np.int64)

    def save(self, path: Path) -> None:
        """Save memory bank and config as .npz; re-fit NearestNeighbors on load."""
        if not self.is_trained or self.memory_bank is None:
            raise ValueError("Model not trained. Cannot save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank,
            feature_dim=self.feature_dim,
            coreset_size=self.coreset_size,
            k_neighbors=self.k_neighbors,
            anomaly_threshold=np.array(self.anomaly_threshold) if self.anomaly_threshold is not None else None,
        )

    def load(self, path: Path) -> None:
        """Load memory bank and config; fit NearestNeighbors on loaded bank."""
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as e:
            raise ImportError("scikit-learn required for PatchCore load") from e

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        data = np.load(path, allow_pickle=True)
        self.memory_bank = data["memory_bank"]
        self.feature_dim = int(data["feature_dim"])
        self.coreset_size = int(data["coreset_size"])
        self.k_neighbors = int(data["k_neighbors"])
        self.anomaly_threshold = None
        if "anomaly_threshold" in data:
            t = data["anomaly_threshold"]
            self.anomaly_threshold = None if t is None else float(np.asarray(t).flat[0])
        self.is_trained = True

        self._nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(self.memory_bank)),
            algorithm="auto",
            metric="minkowski",
            p=2,
        )
        self._nn.fit(self.memory_bank)


# Default implementation
PatchCoreAnomalyDetector = PatchCoreScikitLearn
