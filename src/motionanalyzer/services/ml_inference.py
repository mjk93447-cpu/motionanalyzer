"""Shared ML inference: same features/normalization as analyze_crack_detection.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from motionanalyzer.auto_optimize import (
    PAPER_FEATURE_CONFIG,
    FeatureExtractionConfig,
    apply_norm_stats,
    load_dataset,
    extract_features,
    prepare_training_data,
    select_feature_columns,
)
from motionanalyzer.ml_bundle import load_model_bundle, transform_with_bundle
from motionanalyzer.paths import get_user_models_dir, resolve_draem_model_path

InferenceMode = Literal["draem", "patchcore", "ensemble", "temporal"]


def extract_features_for_bundle(
    bundle_dir: Path,
    config: FeatureExtractionConfig | None = None,
    *,
    label: int = 0,
    fps: float | None = None,
) -> pd.DataFrame:
    """Extract features for one bundle (same path as training pipeline)."""
    config = config or PAPER_FEATURE_CONFIG
    dataset = load_dataset(Path(bundle_dir), label=label, fps=fps)
    return extract_features(dataset, config)


def extract_features_from_vectors_output(
    bundle_dir: Path,
    vectors_csv: Path,
    config: FeatureExtractionConfig | None = None,
    *,
    label: int = 0,
) -> pd.DataFrame:
    """
    Prefer full pipeline from bundle dir (frame_*.txt).

    If only vectors.csv exists in output, re-run feature extraction from source bundle.
    """
    bundle_dir = Path(bundle_dir)
    if any(bundle_dir.glob("frame_*.txt")):
        return extract_features_for_bundle(bundle_dir, config, label=label)
    return extract_features_for_bundle(bundle_dir, config, label=label)


def fit_features_from_manifest(
    manifest_path: Path,
    config: FeatureExtractionConfig | None = None,
    *,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load ML manifest.json and extract all entries (for fitting norm stats)."""
    import json

    config = config or PAPER_FEATURE_CONFIG
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    base = Path(manifest_path).parent
    normal_paths: list[Path] = []
    crack_paths: list[Path] = []
    for entry in data.get("entries", []):
        p = base / entry["path"]
        if entry.get("label", 0) == 0:
            normal_paths.append(p)
        else:
            crack_paths.append(p)
    cache_key = data.get("dataset_id", base.name)
    features_df, labels = prepare_training_data(
        normal_datasets=normal_paths,
        crack_datasets=crack_paths,
        feature_config=config,
        cache_dir=cache_dir,
        cache_key=cache_key if cache_dir else None,
    )
    return features_df, labels, data


def predict_bundle(
    bundle_dir: Path,
    mode: InferenceMode,
    models_dir: Path | None = None,
    *,
    feature_config: FeatureExtractionConfig | None = None,
    dataset_level_max: bool = True,
) -> dict[str, Any]:
    """
    Run DRAEM / PatchCore / Ensemble on one bundle.

    Requires bundle_manifest.json from training.
    """
    models_dir = Path(models_dir or get_user_models_dir())
    manifest = load_model_bundle(models_dir)
    config = feature_config or feature_config_from_manifest(manifest)
    features_df = extract_features_for_bundle(bundle_dir, config, label=0)
    X_df = transform_with_bundle(features_df, manifest)
    feature_cols = manifest["feature_cols"]

    result: dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "n_rows": len(X_df),
        "feature_cols": feature_cols,
    }

    if mode == "draem":
        scores, preds, thresh = _predict_draem(X_df, manifest, models_dir)
        result.update(_pack_scores(features_df, scores, preds, dataset_level_max))
        result["draem_threshold"] = thresh
        return result

    if mode == "patchcore":
        scores, preds, thresh = _predict_patchcore(X_df, manifest, models_dir)
        result.update(_pack_scores(features_df, scores, preds, dataset_level_max))
        result["patchcore_threshold"] = thresh
        return result

    if mode == "ensemble":
        d_scores, d_pred, d_thr = _predict_draem(X_df, manifest, models_dir)
        p_scores, p_pred, p_thr = _predict_patchcore(X_df, manifest, models_dir)
        strategy = manifest.get("ensemble_strategy", "both_agree")
        if strategy == "both_agree":
            ens_pred = ((d_pred == 1) & (p_pred == 1)).astype(int)
            ens_scores = np.minimum(d_scores, p_scores)
        else:
            from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy

            ensemble = _load_ensemble(models_dir, manifest, strategy)
            ens_scores = ensemble.predict(X_df)
            ens_pred = ensemble.predict_binary(X_df)
        result.update(_pack_scores(features_df, ens_scores, ens_pred, dataset_level_max))
        result["draem_threshold"] = d_thr
        result["patchcore_threshold"] = p_thr
        result["ensemble_strategy"] = strategy
        return result

    raise ValueError(f"Unsupported inference mode: {mode}")


def predict_manifest(
    manifest_path: Path,
    mode: InferenceMode,
    models_dir: Path | None = None,
    *,
    split: str | None = "test",
    dataset_level_max: bool = True,
    max_bundles: int | None = None,
) -> dict[str, Any]:
    """Batch inference over manifest entries (CLI/GUI/benchmark)."""
    import json

    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    entries = data.get("entries", [])
    if split:
        entries = [e for e in entries if e.get("split") == split]

    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if max_bundles is not None and i >= max_bundles:
            break
        bundle_dir = base / entry["path"]
        if not bundle_dir.exists():
            continue
        y_true = int(entry.get("label", 0))
        try:
            inf = predict_bundle(
                bundle_dir,
                mode,
                models_dir=models_dir,
                dataset_level_max=dataset_level_max,
            )
            y_pred = int(inf.get("dataset_is_anomaly", 0))
        except Exception as exc:
            rows.append(
                {
                    "path": str(entry["path"]),
                    "label": y_true,
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "path": str(entry["path"]),
                "label": y_true,
                "pred": y_pred,
                "dataset_score": inf.get("dataset_score"),
                "correct": int(y_pred == y_true),
            }
        )

    y_true_arr = np.array([r["label"] for r in rows if "pred" in r], dtype=int)
    y_pred_arr = np.array([r["pred"] for r in rows if "pred" in r], dtype=int)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    if len(y_true_arr):
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        metrics = {
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "n": int(len(y_true_arr)),
        }
    else:
        metrics = {"precision": 0.0, "recall": 0.0, "tn": 0, "fp": 0, "fn": 0, "tp": 0, "n": 0}

    return {"mode": mode, "metrics": metrics, "rows": rows}


def evaluate_bundle_metrics(
    bundle_dir: Path,
    y_true: int,
    mode: InferenceMode,
    models_dir: Path | None = None,
    *,
    dataset_level_max: bool = True,
) -> dict[str, Any]:
    """Single-bundle prediction vs ground-truth label."""
    inf = predict_bundle(
        Path(bundle_dir),
        mode,
        models_dir=models_dir,
        dataset_level_max=dataset_level_max,
    )
    y_pred = int(inf.get("dataset_is_anomaly", 0))
    return {
        "bundle_dir": str(bundle_dir),
        "y_true": int(y_true),
        "y_pred": y_pred,
        "correct": int(y_pred == int(y_true)),
        "dataset_score": inf.get("dataset_score"),
    }


def feature_config_from_manifest(manifest: dict[str, Any]) -> FeatureExtractionConfig:
    from motionanalyzer.ml_bundle import feature_config_from_dict

    return feature_config_from_dict(manifest.get("feature_config", {}))


def _pack_scores(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    preds: np.ndarray,
    dataset_level_max: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scores": scores,
        "predictions": preds,
        "anomaly_rate": float(np.mean(preds)),
    }
    if "frame" in features_df.columns:
        frame_df = pd.DataFrame({"frame": features_df["frame"].values, "score": scores, "is_anomaly": preds})
        out["per_row"] = frame_df
        if dataset_level_max:
            agg_score = float(scores.max())
            agg_pred = int(preds.max())
            out["dataset_score"] = agg_score
            out["dataset_is_anomaly"] = agg_pred
    else:
        out["dataset_score"] = float(scores.max()) if len(scores) else 0.0
        out["dataset_is_anomaly"] = int(preds.max()) if len(preds) else 0
    return out


def _predict_draem(
    X_df: pd.DataFrame,
    manifest: dict[str, Any],
    models_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    from motionanalyzer.ml_models.draem import DRAEMPyTorch

    models_dir = Path(models_dir)
    draem_path = resolve_draem_model_path(
        models_dir / manifest.get("draem_model", "draem_model.pt")
    )
    model = DRAEMPyTorch(input_dim=len(manifest["feature_cols"]))
    model.load(draem_path)
    thresh = manifest.get("draem_threshold")
    if thresh is None:
        thresh = model.reconstruction_error_threshold
    if thresh is None:
        raise ValueError("DRAEM threshold not set in bundle manifest or model.")
    scores = model.predict(X_df)
    preds = (scores >= float(thresh)).astype(int)
    return np.asarray(scores), preds, float(thresh)


def _predict_patchcore(
    X_df: pd.DataFrame,
    manifest: dict[str, Any],
    models_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float]:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    models_dir = Path(models_dir)
    pc_path = models_dir / manifest.get("patchcore_model", "patchcore_model.npz")
    model = PatchCoreScikitLearn(feature_dim=len(manifest["feature_cols"]))
    model.load(pc_path)
    thresh = manifest.get("patchcore_threshold")
    if thresh is None:
        thresh = getattr(model, "anomaly_threshold", None)
    scores = model.predict(X_df)
    if thresh is None:
        preds = model.predict_binary(X_df)
        thresh = float(np.percentile(scores, 95))
    else:
        preds = (scores >= float(thresh)).astype(int)
    return np.asarray(scores), preds, float(thresh)


def _load_ensemble(models_dir: Path, manifest: dict[str, Any], strategy: str) -> Any:
    from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy
    from motionanalyzer.ml_models.draem import DRAEMPyTorch
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    draem = DRAEMPyTorch(input_dim=len(manifest["feature_cols"]))
    draem.load(resolve_draem_model_path(models_dir / manifest.get("draem_model", "draem_model.pt")))
    pc = PatchCoreScikitLearn(feature_dim=len(manifest["feature_cols"]))
    pc.load(models_dir / manifest.get("patchcore_model", "patchcore_model.npz"))
    try:
        strat = EnsembleStrategy(strategy)
    except ValueError:
        strat = EnsembleStrategy.WEIGHTED_AVERAGE
    return EnsembleAnomalyDetector(draem_model=draem, patchcore_model=pc, strategy=strat)
