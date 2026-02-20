"""
Unit tests for advanced feature engineering functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from motionanalyzer.auto_optimize import (
    _compute_advanced_stats,
    _compute_frequency_domain_features,
    _compute_temporal_features,
)


def test_compute_advanced_stats_basic() -> None:
    """Test basic advanced stats computation."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _compute_advanced_stats(series)
    
    assert "skewness" in stats
    assert "kurtosis" in stats
    assert "autocorr_lag1" in stats
    assert "autocorr_lag2" in stats
    assert isinstance(stats["skewness"], float)
    assert isinstance(stats["kurtosis"], float)


def test_compute_advanced_stats_short_series() -> None:
    """Test advanced stats with short series (edge case)."""
    series = pd.Series([1.0, 2.0])
    stats = _compute_advanced_stats(series, min_samples=3)
    
    # Should return zeros for insufficient samples
    assert stats["skewness"] == 0.0
    assert stats["kurtosis"] == 0.0


def test_compute_temporal_features() -> None:
    """Test temporal feature computation."""
    df = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "acceleration_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
        "curvature_mean": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    
    result = _compute_temporal_features(df, ["acceleration_mean", "curvature_mean"])
    
    assert "acceleration_mean_change_rate" in result.columns
    assert "acceleration_mean_change_rate_abs" in result.columns
    assert "acceleration_mean_change_accel" in result.columns
    assert len(result) == len(df)


def test_compute_frequency_domain_features() -> None:
    """Test frequency-domain feature computation."""
    # Create a simple sine wave
    t = np.linspace(0, 1, 30)
    signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
    series = pd.Series(signal)
    
    freq_features = _compute_frequency_domain_features(series, fps=30.0)
    
    assert "dominant_frequency" in freq_features
    assert "spectral_power" in freq_features
    assert "spectral_entropy" in freq_features
    assert freq_features["dominant_frequency"] > 0.0
    assert freq_features["spectral_power"] > 0.0


def test_compute_frequency_domain_features_short() -> None:
    """Test frequency-domain features with short series (edge case)."""
    series = pd.Series([1.0, 2.0, 3.0])
    freq_features = _compute_frequency_domain_features(series, fps=30.0, min_samples=10)
    
    # Should return zeros for insufficient samples
    assert freq_features["dominant_frequency"] == 0.0
    assert freq_features["spectral_power"] == 0.0
