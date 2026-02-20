"""
Change Point Detection for FPCB bending crack detection.

Detects the exact frame where a crack occurs during bending by analyzing
time-series patterns in acceleration, curvature, strain, and other features.

References:
- CUSUM: Page, E.S. (1954). "Continuous Inspection Schemes". Biometrika, 41(1/2), 100-115.
- PELT: Killick, R., Fearnhead, P., & Eckley, I.A. (2012). "Optimal detection of changepoints
  with a linear computational cost". Journal of the American Statistical Association, 107(500), 1590-1598.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ChangePointResult:
    """Result of change point detection."""

    change_points: list[int]
    """Frame indices where change points are detected (0-indexed)."""

    scores: np.ndarray | None = None
    """Anomaly scores for each frame (if available)."""

    method: str = ""
    """Detection method name."""

    def __post_init__(self) -> None:
        """Validate change points."""
        if self.change_points:
            if min(self.change_points) < 0:
                raise ValueError("Change points must be non-negative")
            if len(set(self.change_points)) != len(self.change_points):
                raise ValueError("Change points must be unique")


class ChangePointDetector(ABC):
    """Base class for change point detection algorithms."""

    @abstractmethod
    def detect(self, signal: np.ndarray | pd.Series, **kwargs: Any) -> ChangePointResult:
        """
        Detect change points in a time series signal.

        Args:
            signal: 1D time series signal (e.g., acceleration, curvature, strain)
            **kwargs: Algorithm-specific parameters

        Returns:
            ChangePointResult with detected change point frame indices
        """
        pass


class CUSUMDetector(ChangePointDetector):
    """
    CUSUM (Cumulative Sum) change point detector.

    Detects changes by monitoring cumulative deviations from a target value.
    When the cumulative sum exceeds a threshold, a change point is detected.

    Algorithm:
    - Upper: C_i^+ = max[0, x_i − (T+K) + C_{i-1}^+]
    - Lower: C_i^- = max[0, (T−K) − x_i + C_{i-1}^-]
    where T is target value and K is sensitivity parameter.

    Reference: Page, E.S. (1954). "Continuous Inspection Schemes". Biometrika.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        target: float | None = None,
        sensitivity: float | None = None,
        min_size: int = 3,
    ) -> None:
        """
        Initialize CUSUM detector.

        Args:
            threshold: Detection threshold (h). Higher = fewer false alarms, lower = more sensitive.
            target: Target value (T). If None, uses signal mean.
            sensitivity: Sensitivity parameter (K). If None, uses 0.5 * std(signal).
            min_size: Minimum distance between change points (frames).
        """
        self.threshold = threshold
        self.target = target
        self.sensitivity = sensitivity
        self.min_size = min_size

    def detect(self, signal: np.ndarray | pd.Series, **kwargs: Any) -> ChangePointResult:
        """
        Detect change points using CUSUM algorithm.

        Args:
            signal: 1D time series signal
            **kwargs: Override parameters (threshold, target, sensitivity, min_size)

        Returns:
            ChangePointResult with detected change points
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        signal = np.asarray(signal, dtype=np.float64)

        if len(signal) < 2:
            return ChangePointResult(change_points=[], method="CUSUM")

        # Use kwargs to override instance parameters
        threshold = kwargs.get("threshold", self.threshold)
        target = kwargs.get("target", self.target)
        sensitivity = kwargs.get("sensitivity", self.sensitivity)
        min_size = kwargs.get("min_size", self.min_size)

        # Set default target and sensitivity if not provided
        if target is None:
            target = float(np.mean(signal))
        if sensitivity is None:
            std_val = float(np.std(signal))
            sensitivity = 0.5 * std_val if std_val > 0 else 1.0

        # CUSUM algorithm
        n = len(signal)
        c_upper = np.zeros(n)
        c_lower = np.zeros(n)
        scores = np.zeros(n)

        for i in range(1, n):
            # Upper CUSUM
            c_upper[i] = max(0.0, signal[i] - (target + sensitivity) + c_upper[i - 1])
            # Lower CUSUM
            c_lower[i] = max(0.0, (target - sensitivity) - signal[i] + c_lower[i - 1])
            # Combined score (maximum of upper and lower)
            scores[i] = max(c_upper[i], c_lower[i])

        # Detect change points where score exceeds threshold
        change_points = []
        last_cp = -min_size  # Ensure minimum distance

        for i in range(min_size, n):
            if scores[i] > threshold and (i - last_cp) >= min_size:
                change_points.append(int(i))
                last_cp = i

        return ChangePointResult(
            change_points=change_points,
            scores=scores,
            method="CUSUM",
        )


class WindowBasedDetector(ChangePointDetector):
    """
    Window-based change point detector using statistical tests.

    Detects changes by comparing statistics (mean, variance) between sliding windows.
    """

    def __init__(
        self,
        window_size: int = 10,
        step_size: int = 1,
        threshold_ratio: float = 2.0,
        min_size: int = 3,
    ) -> None:
        """
        Initialize window-based detector.

        Args:
            window_size: Size of sliding window (frames).
            step_size: Step size for sliding window (frames).
            threshold_ratio: Ratio threshold for detecting change (e.g., 2.0 = 2x increase).
            min_size: Minimum distance between change points (frames).
        """
        self.window_size = window_size
        self.step_size = step_size
        self.threshold_ratio = threshold_ratio
        self.min_size = min_size

    def detect(self, signal: np.ndarray | pd.Series, **kwargs: Any) -> ChangePointResult:
        """
        Detect change points using window-based statistical comparison.

        Args:
            signal: 1D time series signal
            **kwargs: Override parameters (window_size, step_size, threshold_ratio, min_size)

        Returns:
            ChangePointResult with detected change points
        """
        if isinstance(signal, pd.Series):
            signal = signal.values
        signal = np.asarray(signal, dtype=np.float64)

        if len(signal) < 2 * self.window_size:
            return ChangePointResult(change_points=[], method="WindowBased")

        # Use kwargs to override instance parameters
        window_size = kwargs.get("window_size", self.window_size)
        step_size = kwargs.get("step_size", self.step_size)
        threshold_ratio = kwargs.get("threshold_ratio", self.threshold_ratio)
        min_size = kwargs.get("min_size", self.min_size)

        n = len(signal)
        scores = np.zeros(n)
        change_points = []

        # Sliding window comparison
        for i in range(window_size, n - window_size, step_size):
            window_before = signal[i - window_size : i]
            window_after = signal[i : i + window_size]

            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            std_before = np.std(window_before) + 1e-6
            std_after = np.std(window_after) + 1e-6

            # Ratio of means (detect spikes)
            mean_ratio = mean_after / (mean_before + 1e-6)
            # Ratio of std (detect variance changes)
            std_ratio = std_after / (std_before + 1e-6)

            # Score: maximum of mean ratio and std ratio (normalized)
            score = max(mean_ratio, std_ratio, 1.0 / mean_ratio, 1.0 / std_ratio)
            scores[i] = score

            # Detect change point if ratio exceeds threshold
            if score > threshold_ratio:
                if not change_points or (i - change_points[-1]) >= min_size:
                    change_points.append(int(i))

        return ChangePointResult(
            change_points=change_points,
            scores=scores,
            method="WindowBased",
        )


def detect_change_points_pelt(
    signal: np.ndarray | pd.Series,
    min_size: int = 3,
    jump: int = 5,
    pen: float = 3.0,
    model: str = "l2",
) -> ChangePointResult:
    """
    Detect change points using PELT (Pruned Exact Linear Time) algorithm.

    Requires 'ruptures' library: pip install ruptures

    Reference: Killick et al. (2012). "Optimal detection of changepoints with a linear computational cost".

    Args:
        signal: 1D time series signal
        min_size: Minimum segment size (frames)
        jump: Grid of possible change points (e.g., jump=5 considers only k, 2k, 3k...)
        pen: Penalty parameter (higher = fewer change points)
        model: Cost function model ("l1", "l2", "rbf")

    Returns:
        ChangePointResult with detected change points

    Raises:
        ImportError: If ruptures library is not installed
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError(
            "PELT detection requires 'ruptures' library. Install with: pip install ruptures"
        ) from e

    if isinstance(signal, pd.Series):
        signal = signal.values
    signal = np.asarray(signal, dtype=np.float64).reshape(-1, 1)

    if len(signal) < 2 * min_size:
        return ChangePointResult(change_points=[], method="PELT")

    # Fit PELT algorithm
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
    change_points = algo.predict(pen=pen)

    # Remove last point if it's the end of signal (not a change point)
    if change_points and change_points[-1] == len(signal):
        change_points = change_points[:-1]

    return ChangePointResult(
        change_points=[int(cp) for cp in change_points],
        scores=None,
        method="PELT",
    )
