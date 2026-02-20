"""
Auto-optimization data preparation pipeline for FPCB crack detection.

Loads normal and crack datasets, extracts features, prepares data for
deep learning models (DREAM, PatchCore) and parameter optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from motionanalyzer.analysis import load_bundle, compute_vectors
from motionanalyzer.crack_model import load_frame_metrics, compute_crack_risk, CrackModelParams


def _compute_advanced_stats(series: pd.Series, min_samples: int = 3) -> dict[str, float]:
    """
    Compute advanced statistical features for a time series.
    
    Args:
        series: Time series values
        min_samples: Minimum samples required for computation
    
    Returns:
        Dictionary with skewness, kurtosis, autocorr_lag1, autocorr_lag2
    """
    if len(series) < min_samples:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "autocorr_lag1": 0.0,
            "autocorr_lag2": 0.0,
        }
    
    values = series.dropna().values
    if len(values) < min_samples:
        return {
            "skewness": 0.0,
            "kurtosis": 0.0,
            "autocorr_lag1": 0.0,
            "autocorr_lag2": 0.0,
        }
    
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    
    # Skewness: measure of asymmetry
    if std_val > 1e-9:
        skew = float(np.mean(((values - mean_val) / std_val) ** 3))
    else:
        skew = 0.0
    
    # Kurtosis: measure of tail weight (excess kurtosis, so subtract 3)
    if std_val > 1e-9:
        kurt = float(np.mean(((values - mean_val) / std_val) ** 4) - 3.0)
    else:
        kurt = 0.0
    
    # Autocorrelation (lag-1, lag-2)
    autocorr_lag1 = 0.0
    autocorr_lag2 = 0.0
    if len(values) >= 2 and std_val > 1e-9:
        # Lag-1 autocorrelation
        if len(values) >= 2:
            corr1 = np.corrcoef(values[:-1], values[1:])[0, 1]
            autocorr_lag1 = float(corr1) if not np.isnan(corr1) else 0.0
        # Lag-2 autocorrelation
        if len(values) >= 3:
            corr2 = np.corrcoef(values[:-2], values[2:])[0, 1]
            autocorr_lag2 = float(corr2) if not np.isnan(corr2) else 0.0
    
    return {
        "skewness": skew,
        "kurtosis": kurt,
        "autocorr_lag1": autocorr_lag1,
        "autocorr_lag2": autocorr_lag2,
    }


def _compute_temporal_features(
    frame_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute temporal features (frame-to-frame change rates).
    
    Args:
        frame_features: Per-frame features DataFrame (must have 'frame' column)
        feature_cols: List of feature column names to compute change rates for
    
    Returns:
        DataFrame with temporal features added
    """
    result = frame_features.copy()
    result = result.sort_values("frame").reset_index(drop=True)
    
    for col in feature_cols:
        if col not in result.columns:
            continue
        
        # Frame-to-frame change rate (first-order difference)
        change_rate = result[col].diff()
        result[f"{col}_change_rate"] = change_rate.fillna(0.0)
        
        # Change rate magnitude (absolute value)
        result[f"{col}_change_rate_abs"] = change_rate.abs().fillna(0.0)
        
        # Second-order change (acceleration of change)
        if len(result) >= 2:
            change_accel = change_rate.diff()
            result[f"{col}_change_accel"] = change_accel.fillna(0.0)
        else:
            result[f"{col}_change_accel"] = 0.0
    
    return result


def _compute_frequency_domain_features(
    series: pd.Series,
    fps: float,
    min_samples: int = 10,
) -> dict[str, float]:
    """
    Compute frequency-domain features using FFT.
    
    Args:
        series: Time series values
        fps: Frames per second (for frequency calculation)
        min_samples: Minimum samples required for FFT
    
    Returns:
        Dictionary with dominant_frequency, spectral_power, spectral_entropy
    """
    if len(series) < min_samples:
        return {
            "dominant_frequency": 0.0,
            "spectral_power": 0.0,
            "spectral_entropy": 0.0,
        }
    
    values = series.dropna().values
    if len(values) < min_samples:
        return {
            "dominant_frequency": 0.0,
            "spectral_power": 0.0,
            "spectral_entropy": 0.0,
        }
    
    # Remove DC component (mean)
    values_centered = values - np.mean(values)
    
    # FFT
    fft_vals = np.fft.rfft(values_centered)
    fft_magnitude = np.abs(fft_vals)
    fft_power = fft_magnitude ** 2
    
    # Frequency bins
    freqs = np.fft.rfftfreq(len(values), d=1.0 / fps)
    
    # Dominant frequency (frequency with maximum power)
    if len(fft_power) > 1:
        dominant_idx = np.argmax(fft_power[1:]) + 1  # Skip DC component
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = 0.0
    
    # Total spectral power (excluding DC)
    spectral_power = float(np.sum(fft_power[1:])) if len(fft_power) > 1 else 0.0
    
    # Spectral entropy (normalized entropy of power spectrum)
    if len(fft_power) > 1 and spectral_power > 1e-9:
        power_normalized = fft_power[1:] / spectral_power
        power_normalized = power_normalized[power_normalized > 1e-9]  # Avoid log(0)
        if len(power_normalized) > 0:
            entropy = -np.sum(power_normalized * np.log2(power_normalized))
            spectral_entropy = float(entropy)
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0
    
    return {
        "dominant_frequency": dominant_freq,
        "spectral_power": spectral_power,
        "spectral_entropy": spectral_entropy,
    }


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    path: Path
    label: int  # 0=normal, 1=crack
    vectors: pd.DataFrame
    frame_metrics: pd.DataFrame | None
    fps: float
    meters_per_pixel: float | None


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    include_per_frame: bool = True
    include_per_point: bool = True
    include_global_stats: bool = True
    window_size: int | None = None  # For time-series windows
    feature_columns: list[str] | None = None  # Specific columns to extract
    include_crack_risk_features: bool = True
    """
    Whether to include Physics-derived `crack_risk`/`crack_risk_*` features.

    IMPORTANT:
    - For Physics parameter tuning/diagnostics, these can be useful.
    - For ML anomaly detection evaluation (DREAM/PatchCore), including `crack_risk_*`
      can cause label leakage / circular evaluation. Prefer False for ML validation.
    """
    include_advanced_stats: bool = False
    """
    Whether to include advanced statistical features:
    - Higher-order statistics: skewness, kurtosis
    - Autocorrelation (lag-1, lag-2)
    - Temporal features: frame-to-frame change rates
    """
    include_frequency_domain: bool = False
    """
    Whether to include frequency-domain features:
    - FFT magnitude/phase (dominant frequency, power spectral density)
    - Wavelet transform coefficients (optional, requires PyWavelets)
    """


def load_dataset(
    dataset_path: Path,
    label: int,
    fps: float | None = None,
    crack_params: CrackModelParams | None = None,
) -> DatasetInfo:
    """
    Load a dataset (normal or crack) and compute vectors with crack risk.

    Args:
        dataset_path: Directory containing frame_*.txt and optionally frame_metrics.csv
        label: 0 for normal, 1 for crack
        fps: FPS (if None, read from fps.txt)
        crack_params: Crack model parameters (if None, use default/user config)

    Returns:
        DatasetInfo with vectors and metadata
    """
    df, fps_val, meters_per_pixel = load_bundle(input_dir=dataset_path, fps=fps)
    vectors = compute_vectors(df=df, fps=fps_val, meters_per_pixel=meters_per_pixel)
    frame_metrics = load_frame_metrics(dataset_path / "frame_metrics.csv")
    
    dt_s = 1.0 / fps_val
    vectors = compute_crack_risk(
        vectors,
        frame_metrics,
        dt_s,
        meters_per_pixel=meters_per_pixel,
        params=crack_params,
    )
    
    return DatasetInfo(
        path=dataset_path,
        label=label,
        vectors=vectors,
        frame_metrics=frame_metrics,
        fps=fps_val,
        meters_per_pixel=meters_per_pixel,
    )


def extract_features(
    dataset: DatasetInfo,
    config: FeatureExtractionConfig,
) -> pd.DataFrame:
    """
    Extract features from dataset for ML model training/optimization.

    Args:
        dataset: Loaded dataset
        config: Feature extraction configuration

    Returns:
        DataFrame with extracted features (one row per frame or per point, depending on config)
    """
    vectors = dataset.vectors
    features_list = []

    if config.include_per_frame:
        # Per-frame features: aggregate per frame
        agg_spec: dict[str, list[str]] = {
            "strain_surrogate": ["mean", "max", "std"],
            "stress_surrogate": ["mean", "max", "std"],
            "impact_surrogate": ["mean", "max", "std"],
            "curvature_like": ["mean", "max", "std"],
            "acceleration": ["mean", "max", "std"],
            "speed": ["mean", "max", "std"],
        }
        if config.include_crack_risk_features and "crack_risk" in vectors.columns:
            agg_spec["crack_risk"] = ["mean", "max", "std"]

        frame_features = vectors.groupby("frame").agg(agg_spec).reset_index()
        frame_features.columns = ["frame"] + [f"{col[0]}_{col[1]}" for col in frame_features.columns[1:]]
        
        # Add frame-level metrics if available
        if dataset.frame_metrics is not None and not dataset.frame_metrics.empty:
            frame_features = frame_features.merge(
                dataset.frame_metrics[["frame", "bend_angle_deg", "curvature_concentration", "est_max_strain"]],
                on="frame",
                how="left",
            )
        
        # Add advanced statistical features if requested
        if config.include_advanced_stats:
            # Compute advanced stats for each aggregated feature
            for base_col in ["acceleration", "curvature_like", "strain_surrogate"]:
                if f"{base_col}_mean" in frame_features.columns:
                    series = frame_features[f"{base_col}_mean"]
                    stats = _compute_advanced_stats(series)
                    for stat_name, stat_value in stats.items():
                        frame_features[f"{base_col}_mean_{stat_name}"] = stat_value
            
            # Add temporal features (frame-to-frame change rates)
            temporal_feature_cols = [
                c for c in frame_features.columns
                if c not in ["frame", "label", "dataset_path"] and c.endswith(("_mean", "_max"))
            ]
            frame_features = _compute_temporal_features(frame_features, temporal_feature_cols)
        
        # Add frequency-domain features if requested
        if config.include_frequency_domain:
            # Compute FFT features for key signals
            for base_col in ["acceleration", "curvature_like"]:
                if f"{base_col}_mean" in frame_features.columns:
                    series = frame_features[f"{base_col}_mean"]
                    freq_features = _compute_frequency_domain_features(series, dataset.fps)
                    for freq_name, freq_value in freq_features.items():
                        frame_features[f"{base_col}_mean_{freq_name}"] = freq_value
        
        frame_features["label"] = dataset.label
        frame_features["dataset_path"] = str(dataset.path)
        features_list.append(frame_features)

    if config.include_per_point:
        # Per-point features: select specific columns
        point_cols = config.feature_columns or [
            "frame", "index", "x", "y",
            "strain_surrogate", "stress_surrogate", "impact_surrogate",
            "curvature_like", "acceleration", "speed",
        ]
        if config.include_crack_risk_features:
            point_cols = list(point_cols) + ["crack_risk"]
        available_cols = [c for c in point_cols if c in vectors.columns]
        point_features = vectors[available_cols].copy()
        point_features["label"] = dataset.label
        point_features["dataset_path"] = str(dataset.path)
        features_list.append(point_features)

    if config.include_global_stats:
        # Global statistics: one row per dataset
        global_row: dict[str, float | int | str] = {
            "label": dataset.label,
            "dataset_path": str(dataset.path),
            "n_frames": vectors["frame"].nunique(),
            "n_points": vectors["index"].nunique(),
            "mean_strain": float(vectors["strain_surrogate"].mean()) if "strain_surrogate" in vectors.columns else 0.0,
            "max_strain": float(vectors["strain_surrogate"].max()) if "strain_surrogate" in vectors.columns else 0.0,
            "mean_acceleration": float(vectors["acceleration"].mean()),
            "max_acceleration": float(vectors["acceleration"].max()),
        }
        if config.include_crack_risk_features and "crack_risk" in vectors.columns:
            global_row["mean_crack_risk"] = float(vectors["crack_risk"].mean())
            global_row["max_crack_risk"] = float(vectors["crack_risk"].max())
            global_row["p95_crack_risk"] = float(vectors["crack_risk"].quantile(0.95))

        global_features = pd.DataFrame([global_row])
        features_list.append(global_features)

    if not features_list:
        raise ValueError("No features extracted: check FeatureExtractionConfig")

    # Combine all feature types
    result = pd.concat(features_list, ignore_index=True)
    return result


def prepare_training_data(
    normal_datasets: list[Path],
    crack_datasets: list[Path],
    fps: float | None = None,
    crack_params: CrackModelParams | None = None,
    feature_config: FeatureExtractionConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare training data from normal and crack datasets.

    Args:
        normal_datasets: List of normal dataset directories
        crack_datasets: List of crack dataset directories
        fps: FPS (if None, read from each dataset's fps.txt)
        crack_params: Crack model parameters
        feature_config: Feature extraction configuration

    Returns:
        (features_df, labels_array) where labels are 0 (normal) or 1 (crack)
    """
    if feature_config is None:
        feature_config = FeatureExtractionConfig()

    all_features = []
    all_labels = []

    # Load normal datasets
    for normal_path in normal_datasets:
        try:
            dataset = load_dataset(normal_path, label=0, fps=fps, crack_params=crack_params)
            features = extract_features(dataset, feature_config)
            all_features.append(features)
            all_labels.extend([0] * len(features))
        except Exception as exc:
            print(f"Warning: Failed to load normal dataset {normal_path}: {exc}")
            continue

    # Load crack datasets
    for crack_path in crack_datasets:
        try:
            dataset = load_dataset(crack_path, label=1, fps=fps, crack_params=crack_params)
            features = extract_features(dataset, feature_config)
            all_features.append(features)
            all_labels.extend([1] * len(features))
        except Exception as exc:
            print(f"Warning: Failed to load crack dataset {crack_path}: {exc}")
            continue

    if not all_features:
        raise ValueError("No datasets loaded successfully")

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.array(all_labels, dtype=int)

    return features_df, labels


def normalize_features(
    features_df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    *,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Normalize feature columns (z-score normalization).

    Args:
        features_df: Features DataFrame
        exclude_cols: Columns to exclude from normalization (e.g., "label", "frame", "index")
        fit_df: Optional DataFrame to compute normalization statistics from.
            If provided, mean/std are computed from fit_df and applied to features_df.
            Use this to avoid leakage (e.g., fit on normal-only, transform both normal+crack).

    Returns:
        Normalized DataFrame
    """
    if exclude_cols is None:
        exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]

    normalized = features_df.copy()
    fit_source = fit_df if fit_df is not None else features_df
    numeric_cols = [
        c
        for c in normalized.columns
        if c not in exclude_cols and normalized[c].dtype in [np.float64, np.int64, float, int]
    ]

    for col in numeric_cols:
        mean_val = float(fit_source[col].mean())
        std_val = float(fit_source[col].std())
        if std_val > 1e-9:
            normalized[col] = (normalized[col] - mean_val) / std_val
        else:
            normalized[col] = 0.0

    return normalized
