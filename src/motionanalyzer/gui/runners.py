"""
Model-mode runners for ML & Optimization tab.

Each mode (physics, dream, patchcore, grid_search, bayesian) is implemented
in a single function. The GUI calls run_training_or_optimization(mode, ...)
only; no model logic lives in the GUI. This keeps model code perfectly
separated and testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from motionanalyzer.paths import (
    get_default_dream_model_path,
    get_default_patchcore_model_path,
    get_default_temporal_model_path,
    get_user_models_dir,
)

# Mode identifiers; must match GUI radiobutton values
MODE_PHYSICS = "physics"
MODE_DREAM = "dream"
MODE_PATCHCORE = "patchcore"
MODE_ENSEMBLE = "ensemble"
MODE_TEMPORAL = "temporal"
MODE_GRID_SEARCH = "grid_search"
MODE_BAYESIAN = "bayesian"

ALL_MODES = [MODE_PHYSICS, MODE_DREAM, MODE_PATCHCORE, MODE_ENSEMBLE, MODE_TEMPORAL, MODE_GRID_SEARCH, MODE_BAYESIAN]


def run_training_or_optimization(
    mode: str,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[], None] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """
    Single entry point for ML training or parameter optimization.

    Args:
        mode: One of physics, dream, patchcore, grid_search, bayesian.
        features_df: Prepared (e.g. normalized) feature DataFrame.
        labels: 0 = normal, 1 = crack.
        log_callback: Optional callback for log lines (e.g. GUI text insert).
        progress_callback: Optional callback to update UI (e.g. self.update()).
        **options: Mode-specific options (e.g. epochs, batch_size for DREAM).

    Returns:
        Dict with at least: success (bool), message (str), and mode-specific keys
        (e.g. model_path, best_params, metrics).
    """
    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    def progress() -> None:
        if progress_callback:
            progress_callback()

    if mode == MODE_DREAM:
        return _run_dream(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_PATCHCORE:
        return _run_patchcore(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_ENSEMBLE:
        return _run_ensemble(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_TEMPORAL:
        return _run_temporal(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_GRID_SEARCH:
        return _run_grid_search(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_BAYESIAN:
        return _run_bayesian(features_df, labels, log=log, progress=progress, **options)
    if mode == MODE_PHYSICS:
        return _run_physics_placeholder(features_df, labels, log=log, **options)

    return {"success": False, "message": f"Unknown mode: {mode}"}


def _run_dream(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    epochs: int = 50,
    batch_size: int = 32,
    model_save_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train DREAM model (normal-only); evaluate on crack if present."""
    try:
        from motionanalyzer.ml_models.dream import DREAMAnomalyDetector
    except ImportError:
        return {
            "success": False,
            "message": "PyTorch not installed. Install with: pip install torch or pip install -e '.[ml]'",
        }

    exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]
    # Avoid Physics-derived crack_risk features for ML anomaly detection (leakage/circularity)
    feature_cols = [
        c for c in features_df.columns if c not in exclude_cols and "crack_risk" not in c.lower()
    ]
    normal_mask = labels == 0
    normal_data = features_df.loc[normal_mask, feature_cols]
    normal_array = normal_data.to_numpy(dtype=np.float32)

    log(f"Training DREAM on {len(normal_array)} normal samples (crack-like synthetic anomalies enabled)...")
    progress()

    model = DREAMAnomalyDetector(
        input_dim=len(feature_cols),
        hidden_dims=kwargs.get("hidden_dims", [64, 32, 16]),
        latent_dim=kwargs.get("latent_dim", 8),
        learning_rate=kwargs.get("learning_rate", 1e-3),
        batch_size=batch_size,
        use_discriminative=kwargs.get("use_discriminative", True),
        synthetic_noise_std=kwargs.get("synthetic_noise_std", 0.3),
        discriminator_weight=kwargs.get("discriminator_weight", 0.5),
        weight_decay=kwargs.get("weight_decay", 1e-5),
    )
    model.fit(normal_data, epochs=epochs, feature_names=feature_cols)

    # Threshold optimization: use optimize_threshold_for_precision_recall if crack data available
    crack_mask = ~normal_mask
    if crack_mask.any() and kwargs.get("optimize_threshold", True):
        crack_data = features_df.loc[crack_mask, feature_cols]
        try:
            thresh, metrics = model.optimize_threshold_for_precision_recall(
                normal_data,
                crack_data,
                target_metric=kwargs.get("threshold_metric", "balanced"),
            )
            log(f"Optimized threshold: {thresh:.4f} (Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f})")
        except Exception as e:
            log(f"Threshold optimization failed, using p95: {e}")
            model.set_threshold_from_normal(normal_data, percentile=kwargs.get("threshold_percentile", 95.0))
    else:
        model.set_threshold_from_normal(normal_data, percentile=kwargs.get("threshold_percentile", 95.0))

    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_default_dream_model_path() if model_save_dir is None else (save_dir / "dream_model.pt")
    model.save(model_path)
    log(f"Model saved to: {model_path}")

    result: dict[str, Any] = {"success": True, "message": "DREAM training complete", "model_path": model_path}

    if crack_mask.any():
        crack_data = features_df.loc[crack_mask, feature_cols]
        crack_array = crack_data.to_numpy(dtype=np.float32)
        crack_scores = model.predict(crack_array)
        crack_pred = model.predict_binary(crack_array)
        result["crack_anomaly_rate"] = float(crack_pred.mean())
        result["crack_mean_score"] = float(crack_scores.mean())
        log(f"Evaluation on {len(crack_array)} crack samples: anomaly rate = {result['crack_anomaly_rate']:.3f}")

    return result


def _run_patchcore(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    model_save_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train PatchCore (memory bank from normal data); evaluate on crack if present."""
    try:
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError:
        return {
            "success": False,
            "message": "PatchCore requires scikit-learn. Install with: pip install -e '.[ml]'",
        }

    exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]
    # Avoid Physics-derived crack_risk features for ML anomaly detection (leakage/circularity)
    feature_cols = [
        c for c in features_df.columns if c not in exclude_cols and "crack_risk" not in c.lower()
    ]
    normal_mask = labels == 0
    normal_df = features_df.loc[normal_mask, feature_cols]

    if len(normal_df) < 2:
        return {"success": False, "message": "PatchCore requires at least 2 normal samples."}

    feature_dim = len(feature_cols)
    coreset_size = int(kwargs.get("coreset_size", 1000))
    k_neighbors = int(kwargs.get("k_neighbors", 1))
    percentile = float(kwargs.get("threshold_percentile", 95.0))

    log(f"Training PatchCore on {len(normal_df)} normal samples (coreset_size={coreset_size}, k={k_neighbors})...")
    progress()

    model = PatchCoreScikitLearn(
        feature_dim=feature_dim,
        coreset_size=min(coreset_size, len(normal_df)),
        k_neighbors=min(k_neighbors, len(normal_df)),
    )
    model.fit(normal_df)
    model.set_threshold_from_normal(normal_df, percentile=percentile)

    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        get_default_patchcore_model_path()
        if model_save_dir is None
        else (save_dir / "patchcore_model.npz")
    )
    model.save(model_path)
    log(f"Model saved to: {model_path}")

    result: dict[str, Any] = {
        "success": True,
        "message": "PatchCore training complete",
        "model_path": model_path,
    }

    crack_mask = ~normal_mask
    if crack_mask.any():
        crack_df = features_df.loc[crack_mask, feature_cols]
        crack_scores = model.predict(crack_df)
        crack_pred = model.predict_binary(crack_df)
        result["crack_anomaly_rate"] = float(crack_pred.mean())
        result["crack_mean_score"] = float(crack_scores.mean())
        log(f"Evaluation on {len(crack_df)} crack samples: anomaly rate = {result['crack_anomaly_rate']:.3f}")

    return result


def _run_ensemble(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    dream_model_path: Path | str | None = None,
    patchcore_model_path: Path | str | None = None,
    strategy: str = "weighted_average",
    optimize_weights: bool = True,
    model_save_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Train ensemble model combining DREAM and PatchCore.

    Requires pre-trained DREAM and PatchCore models. Loads them and combines predictions.
    """
    try:
        from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy
        from motionanalyzer.ml_models.dream import DREAMPyTorch
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError as e:
        return {
            "success": False,
            "message": f"Ensemble requires ML dependencies: {e}. Install with: pip install -e '.[ml]'",
        }

    # Load base models
    dream_path = Path(dream_model_path) if dream_model_path else get_default_dream_model_path()
    patchcore_path = Path(patchcore_model_path) if patchcore_model_path else get_default_patchcore_model_path()

    if not dream_path.exists():
        return {
            "success": False,
            "message": f"DREAM model not found: {dream_path}\nTrain DREAM model first.",
        }
    if not patchcore_path.exists():
        return {
            "success": False,
            "message": f"PatchCore model not found: {patchcore_path}\nTrain PatchCore model first.",
        }

    log(f"Loading DREAM model from: {dream_path}")
    dream_model = DREAMPyTorch(input_dim=1)  # Will be set correctly after load
    dream_model.load(dream_path)

    log(f"Loading PatchCore model from: {patchcore_path}")
    patchcore_model = PatchCoreScikitLearn(feature_dim=1)  # Will be set correctly after load
    patchcore_model.load(patchcore_path)

    # Determine strategy
    try:
        ensemble_strategy = EnsembleStrategy(strategy)
    except ValueError:
        ensemble_strategy = EnsembleStrategy.WEIGHTED_AVERAGE
        log(f"Unknown strategy '{strategy}', using weighted_average")

    # Create ensemble
    ensemble = EnsembleAnomalyDetector(
        dream_model=dream_model,
        patchcore_model=patchcore_model,
        strategy=ensemble_strategy,
        dream_weight=0.5,
        patchcore_weight=0.5,
    )

    # Split data for evaluation
    normal_mask = labels == 0
    normal_df = features_df.loc[normal_mask]
    crack_mask = ~normal_mask
    crack_df = features_df.loc[crack_mask] if crack_mask.any() else pd.DataFrame()

    # Filter feature columns (exclude crack_risk)
    exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [c for c in features_df.columns if c not in exclude_cols and "crack_risk" not in c.lower()]
    if not feature_cols:
        return {"success": False, "message": "No valid features found (all excluded)"}

    normal_features = normal_df[feature_cols].fillna(0.0)
    crack_features = crack_df[feature_cols].fillna(0.0) if len(crack_df) > 0 else pd.DataFrame()

    # Optimize weights if requested and strategy is weighted_average
    if optimize_weights and ensemble_strategy == EnsembleStrategy.WEIGHTED_AVERAGE and len(crack_features) > 0:
        log("Optimizing ensemble weights...")
        dream_weight, patchcore_weight, best_metrics = ensemble.optimize_weights(
            normal_features, crack_features, target_metric="balanced"
        )
        log(f"Optimal weights: DREAM={dream_weight:.3f}, PatchCore={patchcore_weight:.3f}")
        log(f"Best metrics: {best_metrics}")

    # Set threshold
    if len(normal_features) > 0:
        ensemble.set_threshold_from_normal(normal_features, percentile=95.0)
        log(f"Ensemble threshold set to: {ensemble.ensemble_threshold:.4f}")

    # Evaluate on crack data if available
    result: dict[str, Any] = {
        "success": True,
        "message": f"Ensemble ({ensemble_strategy.value}) ready",
        "strategy": ensemble_strategy.value,
        "dream_weight": ensemble.dream_weight,
        "patchcore_weight": ensemble.patchcore_weight,
        "ensemble_threshold": ensemble.ensemble_threshold,
    }

    if len(crack_features) > 0:
        crack_scores = ensemble.predict(crack_features)
        crack_pred = ensemble.predict_binary(crack_features)
        result["crack_anomaly_rate"] = float(crack_pred.mean())
        result["crack_mean_score"] = float(crack_scores.mean())
        log(f"Evaluation on {len(crack_features)} crack samples: anomaly rate = {result['crack_anomaly_rate']:.3f}")

    # Save ensemble config
    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    ensemble_path = save_dir / "ensemble_config.json"
    ensemble.save(ensemble_path)
    result["model_path"] = ensemble_path
    log(f"Ensemble config saved to: {ensemble_path}")

    return result


def _run_grid_search(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    normal_dataset_paths: list[Path] | None = None,
    crack_dataset_paths: list[Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Grid search over CrackModelParams; maximize AUC-ROC on per-dataset max crack_risk."""
    normal_paths = normal_dataset_paths or []
    crack_paths = crack_dataset_paths or []
    if not normal_paths and not crack_paths:
        return {
            "success": False,
            "message": "Grid search requires dataset paths. Prepare data first, then run (paths are passed from ML tab).",
        }
    try:
        from motionanalyzer.optimizers.grid_search import run_grid_search
    except ImportError as e:
        return {"success": False, "message": f"Grid search module not available: {e}"}

    log("Grid search over CrackModelParams (AUC-ROC)...")
    result = run_grid_search(
        normal_paths,
        crack_paths,
        fps=kwargs.get("fps"),
        param_grid=kwargs.get("param_grid"),
        base_params=kwargs.get("base_params"),
        log=log,
        progress=progress,
    )
    if result.get("success") and result.get("best_params") is not None:
        from motionanalyzer.crack_model import save_params, get_user_params_path
        out_path = kwargs.get("params_save_path") or get_user_params_path()
        save_params(result["best_params"], Path(out_path))
        log(f"Best params saved to {out_path}")
    return result


def _run_bayesian(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    normal_dataset_paths: list[Path] | None = None,
    crack_dataset_paths: list[Path] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bayesian optimization of CrackModelParams (Optuna); maximize AUC-ROC."""
    normal_paths = normal_dataset_paths or []
    crack_paths = crack_dataset_paths or []
    if not normal_paths and not crack_paths:
        return {
            "success": False,
            "message": "Bayesian optimization requires dataset paths. Prepare data first, then run.",
        }
    try:
        from motionanalyzer.optimizers.bayesian import run_bayesian_optimization
    except ImportError as e:
        return {"success": False, "message": f"Bayesian module not available: {e}"}

    log("Bayesian optimization (Optuna) over CrackModelParams...")
    result = run_bayesian_optimization(
        normal_paths,
        crack_paths,
        fps=kwargs.get("fps"),
        n_trials=int(kwargs.get("n_trials", 20)),
        base_params=kwargs.get("base_params"),
        log=log,
        progress=progress,
    )
    if result.get("success") and result.get("best_params") is not None:
        from motionanalyzer.crack_model import save_params, get_user_params_path
        out_path = kwargs.get("params_save_path") or get_user_params_path()
        save_params(result["best_params"], Path(out_path))
        log(f"Best params saved to {out_path}")
    return result


def _run_temporal(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    sequence_length: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 2,
    cell_type: str = "LSTM",
    epochs: int = 50,
    batch_size: int = 32,
    model_save_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train temporal LSTM/GRU autoencoder model (normal-only); evaluate on crack if present."""
    try:
        from motionanalyzer.ml_models.dream_temporal import TemporalAnomalyDetector
    except ImportError:
        return {
            "success": False,
            "message": "PyTorch not installed. Install with: pip install torch or pip install -e '.[ml]'",
        }

    # Temporal model requires per-frame features with dataset_path and frame columns
    if "dataset_path" not in features_df.columns or "frame" not in features_df.columns:
        return {
            "success": False,
            "message": "Temporal model requires 'dataset_path' and 'frame' columns. Use per-frame features.",
        }

    exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]
    # Avoid Physics-derived crack_risk features for ML anomaly detection (leakage/circularity)
    feature_cols = [
        c for c in features_df.columns if c not in exclude_cols and "crack_risk" not in c.lower()
    ]
    if not feature_cols:
        return {"success": False, "message": "No valid features found (all excluded)"}

    normal_mask = labels == 0
    normal_df = features_df.loc[normal_mask].copy()

    if len(normal_df) == 0:
        return {"success": False, "message": "No normal samples found"}

    log(f"Training Temporal ({cell_type}) model on {len(normal_df)} normal frames...")
    log(f"  Sequence length: {sequence_length}, Hidden dim: {hidden_dim}, Layers: {num_layers}")
    progress()

    model = TemporalAnomalyDetector(
        feature_dim=len(feature_cols),
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        cell_type=cell_type,
        learning_rate=kwargs.get("learning_rate", 1e-3),
        batch_size=batch_size,
    )

    model.fit(normal_df, feature_cols, epochs=epochs)
    model.set_threshold_from_normal(normal_df, feature_cols, percentile=95.0)

    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        get_default_temporal_model_path()
        if model_save_dir is None
        else (save_dir / "temporal_model.pt")
    )
    model.save(model_path)
    log(f"Model saved to: {model_path}")

    result: dict[str, Any] = {
        "success": True,
        "message": "Temporal model training complete",
        "model_path": model_path,
        "sequence_length": sequence_length,
        "cell_type": cell_type,
    }

    crack_mask = ~normal_mask
    if crack_mask.any():
        crack_df = features_df.loc[crack_mask].copy()
        scores_df = model.predict(crack_df, feature_cols)
        if len(scores_df) > 0:
            result["crack_mean_score"] = float(scores_df["anomaly_score"].mean())
            result["crack_max_score"] = float(scores_df["anomaly_score"].max())
            log(f"Evaluation on {len(crack_df)} crack frames: mean score = {result['crack_mean_score']:.4f}")

    return result


def _run_physics_placeholder(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    **kwargs: Any,
) -> dict[str, Any]:
    """Physics model has no training; parameters are tuned in Crack Model Tuning tab."""
    log("Physics model uses parameters from Crack Model Tuning tab (no training step).")
    return {"success": True, "message": "Physics parameters are tuned in the Tuning tab"}
