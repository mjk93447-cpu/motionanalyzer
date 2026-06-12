"""
Model-mode runners for ML & Optimization tab.

Each mode (physics, draem, patchcore, grid_search, bayesian) is implemented
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
    get_default_draem_model_path,
    get_default_patchcore_model_path,
    get_default_temporal_model_path,
    get_user_models_dir,
)

# Mode identifiers; must match GUI radiobutton values
MODE_PHYSICS = "physics"
MODE_DRAEM = "draem"
MODE_PATCHCORE = "patchcore"
MODE_ENSEMBLE = "ensemble"
MODE_TEMPORAL = "temporal"
MODE_GRID_SEARCH = "grid_search"
MODE_BAYESIAN = "bayesian"

ALL_MODES = [MODE_PHYSICS, MODE_DRAEM, MODE_PATCHCORE, MODE_ENSEMBLE, MODE_TEMPORAL, MODE_GRID_SEARCH, MODE_BAYESIAN]


def _ml_feature_columns(features_df: pd.DataFrame) -> list[str]:
    exclude_cols = ["label", "dataset_path", "frame", "index", "x", "y"]
    return [
        c for c in features_df.columns if c not in exclude_cols and "crack_risk" not in c.lower()
    ]


def _normalize_ml_features(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Z-score fit on normal-only rows (same stats as bundle_manifest / inference)."""
    from motionanalyzer.auto_optimize import apply_norm_stats, compute_norm_stats

    normal_mask = labels == 0
    fit_df = features_df.loc[normal_mask]
    norm_stats = compute_norm_stats(fit_df)
    out = features_df.copy()
    transformed = apply_norm_stats(features_df, norm_stats, feature_cols)
    for col in feature_cols:
        out[col] = transformed[col]
    return out.fillna(0.0)


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
        mode: One of physics, draem, patchcore, grid_search, bayesian.
        features_df: Prepared (e.g. normalized) feature DataFrame.
        labels: 0 = normal, 1 = crack.
        log_callback: Optional callback for log lines (e.g. GUI text insert).
        progress_callback: Optional callback to update UI (e.g. self.update()).
        **options: Mode-specific options (e.g. epochs, batch_size for DRAEM).

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

    if mode == MODE_DRAEM:
        return _run_draem(features_df, labels, log=log, progress=progress, **options)
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


def _run_draem(
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
    """Train DRAEM model (normal-only); evaluate on crack if present."""
    try:
        from motionanalyzer.ml_models.draem import DRAEMAnomalyDetector
    except ImportError:
        return {
            "success": False,
            "message": "PyTorch not installed. Install with: pip install torch or pip install -e '.[ml]'",
        }

    feature_cols = _ml_feature_columns(features_df)
    normal_mask = labels == 0
    normed_df = _normalize_ml_features(features_df, labels, feature_cols)
    normal_array = normed_df.loc[normal_mask, feature_cols].to_numpy(dtype=np.float32)

    train_mode = str(kwargs.get("train_mode", "scratch")).lower()
    draem_model_path = kwargs.get("draem_model_path")
    learning_rate = float(kwargs.get("learning_rate", 1e-3))

    if train_mode == "refine" and draem_model_path:
        load_path = Path(draem_model_path)
        if not load_path.exists():
            return {
                "success": False,
                "message": f"Pretrained DRAEM not found: {load_path}\nTrain scratch first or set draem_model_path.",
            }
        log(f"Refining DRAEM from {load_path} with {len(normal_array)} normal samples for {epochs} epoch(s)...")
        progress()
        model = DRAEMAnomalyDetector(
            input_dim=len(feature_cols),
            hidden_dims=kwargs.get("hidden_dims", [64, 32, 16]),
            latent_dim=kwargs.get("latent_dim", 8),
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_discriminative=kwargs.get("use_discriminative", True),
            synthetic_noise_std=kwargs.get("synthetic_noise_std", 0.3),
            discriminator_weight=kwargs.get("discriminator_weight", 0.5),
            weight_decay=kwargs.get("weight_decay", 1e-5),
        )
        try:
            model.load(load_path)
        except Exception as exc:
            return {"success": False, "message": f"Failed to load DRAEM checkpoint: {exc}"}
        if int(getattr(model, "input_dim", len(feature_cols))) != len(feature_cols):
            return {
                "success": False,
                "message": (
                    f"Feature dimension mismatch: pretrained={model.input_dim}, data={len(feature_cols)}. "
                    "Use the same feature_cols as the source bundle."
                ),
            }
        model.fit(
            normal_array,
            epochs=epochs,
            feature_names=feature_cols,
            progress_callback=kwargs.get("epoch_progress_callback"),
            stop_callback=kwargs.get("stop_callback"),
        )
    else:
        log(f"Training DRAEM on {len(normal_array)} normal samples (crack-like synthetic anomalies enabled)...")
        progress()
        model = DRAEMAnomalyDetector(
            input_dim=len(feature_cols),
            hidden_dims=kwargs.get("hidden_dims", [64, 32, 16]),
            latent_dim=kwargs.get("latent_dim", 8),
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_discriminative=kwargs.get("use_discriminative", True),
            synthetic_noise_std=kwargs.get("synthetic_noise_std", 0.3),
            discriminator_weight=kwargs.get("discriminator_weight", 0.5),
            weight_decay=kwargs.get("weight_decay", 1e-5),
        )
        model.fit(
            normal_array,
            epochs=epochs,
            feature_names=feature_cols,
            progress_callback=kwargs.get("epoch_progress_callback"),
            stop_callback=kwargs.get("stop_callback"),
        )

    # Threshold optimization: use optimize_threshold_for_precision_recall if crack data available
    crack_mask = ~normal_mask
    if crack_mask.any() and kwargs.get("optimize_threshold", True):
        crack_data = normed_df.loc[crack_mask, feature_cols]
        try:
            thresh, metrics = model.optimize_threshold_for_precision_recall(
                normed_df.loc[normal_mask, feature_cols],
                crack_data,
                target_metric=kwargs.get("threshold_metric", "balanced"),
            )
            log(f"Optimized threshold: {thresh:.4f} (Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f})")
        except Exception as e:
            log(f"Threshold optimization failed, using p95: {e}")
            model.set_threshold_from_normal(
                normed_df.loc[normal_mask, feature_cols],
                percentile=kwargs.get("threshold_percentile", 95.0),
            )
    else:
        model.set_threshold_from_normal(
            normed_df.loc[normal_mask, feature_cols],
            percentile=kwargs.get("threshold_percentile", 95.0),
        )

    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = get_default_draem_model_path() if model_save_dir is None else (save_dir / "draem_model.pt")
    model.save(model_path)
    log(f"Model saved to: {model_path}")

    result: dict[str, Any] = {"success": True, "message": "DRAEM training complete", "model_path": model_path}

    if crack_mask.any():
        crack_array = normed_df.loc[crack_mask, feature_cols].to_numpy(dtype=np.float32)
        crack_scores = model.predict(crack_array)
        crack_pred = model.predict_binary(crack_array)
        result["crack_anomaly_rate"] = float(crack_pred.mean())
        result["crack_mean_score"] = float(crack_scores.mean())
        log(f"Evaluation on {len(crack_array)} crack samples: anomaly rate = {result['crack_anomaly_rate']:.3f}")

    try:
        from motionanalyzer.auto_optimize import PAPER_FEATURE_CONFIG
        from motionanalyzer.ml_bundle import save_model_bundle

        fc = kwargs.get("feature_config", PAPER_FEATURE_CONFIG)
        save_model_bundle(
            save_dir,
            feature_config=fc,
            features_df=features_df,
            labels=labels,
            draem_threshold=float(model.reconstruction_error_threshold or 0.0),
            ensemble_strategy="both_agree",
        )
        log(f"Updated bundle manifest in {save_dir}")
    except Exception as exc:
        log(f"Warning: could not save bundle manifest: {exc}")

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
    """Train or refine PatchCore on normal-only memory bank; evaluate crack if present."""
    try:
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError:
        return {
            "success": False,
            "message": "PatchCore requires scikit-learn. Install with: pip install -e '.[ml]'",
        }

    feature_cols = _ml_feature_columns(features_df)
    normal_mask = labels == 0
    normed_df = _normalize_ml_features(features_df, labels, feature_cols)
    normal_df = normed_df.loc[normal_mask, feature_cols]

    if len(normal_df) < 2:
        return {"success": False, "message": "PatchCore requires at least 2 normal samples."}

    feature_dim = len(feature_cols)
    coreset_size = int(kwargs.get("coreset_size", 1000))
    k_neighbors = int(kwargs.get("k_neighbors", 1))
    percentile = float(kwargs.get("threshold_percentile", 95.0))
    train_mode = str(kwargs.get("train_mode", "scratch")).lower()
    pretrained_path = kwargs.get("pretrained_path")

    save_dir = Path(model_save_dir) if model_save_dir is not None else get_user_models_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        get_default_patchcore_model_path()
        if model_save_dir is None
        else (save_dir / "patchcore_model.npz")
    )

    if train_mode == "refine":
        load_path = Path(pretrained_path) if pretrained_path else model_path
        if not load_path.exists():
            return {
                "success": False,
                "message": f"Pretrained PatchCore not found: {load_path}\nTrain scratch first or set pretrained path.",
            }
        log(f"Refining PatchCore from {load_path} with {len(normal_df)} new normal samples...")
        progress()
        model = PatchCoreScikitLearn(feature_dim=feature_dim, coreset_size=coreset_size, k_neighbors=k_neighbors)
        model.load(load_path)
        if model.feature_dim != feature_dim:
            return {
                "success": False,
                "message": (
                    f"Feature dimension mismatch: pretrained={model.feature_dim}, data={feature_dim}. "
                    "Use Paper/CLI preset and the same manifest feature_cols."
                ),
            }
        source_tag = str(kwargs.get("source_tag", "real_normal_refine"))
        model.fit_incremental(normal_df, source_tag=source_tag)
        model.refit_threshold(normal_df, percentile=percentile)
        msg = "PatchCore refine complete"
    else:
        log(f"Training PatchCore scratch on {len(normal_df)} normal samples (coreset={coreset_size}, k={k_neighbors})...")
        progress()
        model = PatchCoreScikitLearn(
            feature_dim=feature_dim,
            coreset_size=min(coreset_size, len(normal_df)),
            k_neighbors=min(k_neighbors, len(normal_df)),
        )
        model.fit(normal_df)
        model.set_threshold_from_normal(normal_df, percentile=percentile)
        msg = "PatchCore training complete"

    model.save(model_path)
    log(f"Model saved to: {model_path} (bank size={model.n_bank_samples})")

    result: dict[str, Any] = {
        "success": True,
        "message": msg,
        "model_path": model_path,
        "patchcore_n_bank_samples": model.n_bank_samples,
        "patchcore_training_sources": list(model.training_sources),
    }

    crack_mask = ~normal_mask
    if crack_mask.any():
        crack_df = normed_df.loc[crack_mask, feature_cols]
        crack_scores = model.predict(crack_df)
        crack_pred = model.predict_binary(crack_df)
        result["crack_anomaly_rate"] = float(crack_pred.mean())
        result["crack_mean_score"] = float(crack_scores.mean())
        log(f"Evaluation on {len(crack_df)} crack samples: anomaly rate = {result['crack_anomaly_rate']:.3f}")

    try:
        from motionanalyzer.auto_optimize import PAPER_FEATURE_CONFIG
        from motionanalyzer.ml_bundle import get_bundle_manifest_path, save_model_bundle
        import json

        fc = kwargs.get("feature_config", PAPER_FEATURE_CONFIG)
        manifest_path = get_bundle_manifest_path(save_dir)
        pc_thr = float(model.anomaly_threshold or 0.0)
        draem_thr = None
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            draem_thr = existing.get("draem_threshold")
        save_model_bundle(
            save_dir,
            feature_config=fc,
            features_df=features_df,
            labels=labels,
            draem_threshold=draem_thr,
            patchcore_threshold=pc_thr,
            ensemble_strategy="both_agree",
            training_dataset_id=str(kwargs.get("training_dataset_id", "")),
            patchcore_n_bank_samples=model.n_bank_samples,
            patchcore_training_sources=list(model.training_sources),
        )
        log(f"Updated bundle manifest in {save_dir}")
    except Exception as exc:
        log(f"Warning: could not save bundle manifest: {exc}")

    return result


def evaluate_patchcore_on_prepared(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    model_path: Path | str | None = None,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Confusion-style metrics for prepared feature matrix (GUI Evaluate)."""
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    feature_cols = _ml_feature_columns(features_df)
    normed_df = _normalize_ml_features(features_df, labels, feature_cols)
    path = Path(model_path or get_default_patchcore_model_path())
    model = PatchCoreScikitLearn(feature_dim=len(feature_cols))
    model.load(path)
    X = normed_df[feature_cols].fillna(0.0)
    scores = model.predict(X)
    t = threshold if threshold is not None else model.anomaly_threshold
    preds = model.predict_binary(X, threshold=t)
    y_true = np.asarray(labels, dtype=int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "crack_anomaly_rate": float(preds[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
    }


def _run_ensemble(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    draem_model_path: Path | str | None = None,
    patchcore_model_path: Path | str | None = None,
    strategy: str = "weighted_average",
    optimize_weights: bool = True,
    model_save_dir: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Train ensemble model combining DRAEM and PatchCore.

    Requires pre-trained DRAEM and PatchCore models. Loads them and combines predictions.
    """
    try:
        from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy
        from motionanalyzer.ml_models.draem import DRAEMPyTorch
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    except ImportError as e:
        return {
            "success": False,
            "message": f"Ensemble requires ML dependencies: {e}. Install with: pip install -e '.[ml]'",
        }

    # Load base models
    draem_path = Path(draem_model_path) if draem_model_path else get_default_draem_model_path()
    patchcore_path = Path(patchcore_model_path) if patchcore_model_path else get_default_patchcore_model_path()

    if not draem_path.exists():
        return {
            "success": False,
            "message": f"DRAEM model not found: {draem_path}\nTrain DRAEM model first.",
        }
    if not patchcore_path.exists():
        return {
            "success": False,
            "message": f"PatchCore model not found: {patchcore_path}\nTrain PatchCore model first.",
        }

    log(f"Loading DRAEM model from: {draem_path}")
    draem_model = DRAEMPyTorch(input_dim=1)  # Will be set correctly after load
    draem_model.load(draem_path)

    log(f"Loading PatchCore model from: {patchcore_path}")
    patchcore_model = PatchCoreScikitLearn(feature_dim=1)  # Will be set correctly after load
    patchcore_model.load(patchcore_path)

    # Determine strategy
    if strategy in ("both_agree", "paper"):
        ensemble_strategy = EnsembleStrategy.BOTH_AGREE
    else:
        try:
            ensemble_strategy = EnsembleStrategy(strategy)
        except ValueError:
            ensemble_strategy = EnsembleStrategy.BOTH_AGREE
            log(f"Unknown strategy '{strategy}', using both_agree")

    # Create ensemble
    ensemble = EnsembleAnomalyDetector(
        draem_model=draem_model,
        patchcore_model=patchcore_model,
        strategy=ensemble_strategy,
        draem_weight=0.5,
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
        draem_weight, patchcore_weight, best_metrics = ensemble.optimize_weights(
            normal_features, crack_features, target_metric="balanced"
        )
        log(f"Optimal weights: DRAEM={draem_weight:.3f}, PatchCore={patchcore_weight:.3f}")
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
        "draem_weight": ensemble.draem_weight,
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
        from motionanalyzer.ml_models.draem_temporal import TemporalAnomalyDetector
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
