"""Model bundle manifest: features, normalization, thresholds (CLI/GUI shared)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motionanalyzer.auto_optimize import (
    FEATURE_EXCLUDE_COLS,
    FeatureExtractionConfig,
    PAPER_FEATURE_CONFIG,
    apply_norm_stats,
    compute_norm_stats,
    select_feature_columns,
)
from motionanalyzer.paths import get_default_draem_model_path, get_default_patchcore_model_path, get_user_models_dir

BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"


def get_bundle_manifest_path(models_dir: Path | None = None) -> Path:
    return (models_dir or get_user_models_dir()) / BUNDLE_MANIFEST_FILENAME


def feature_config_to_dict(config: FeatureExtractionConfig) -> dict[str, Any]:
    return asdict(config)


def feature_config_from_dict(data: dict[str, Any]) -> FeatureExtractionConfig:
    return FeatureExtractionConfig(**{k: v for k, v in data.items() if k in FeatureExtractionConfig.__dataclass_fields__})


def save_model_bundle(
    models_dir: Path | None,
    *,
    feature_config: FeatureExtractionConfig,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    draem_threshold: float | None = None,
    patchcore_threshold: float | None = None,
    ensemble_strategy: str = "both_agree",
    training_dataset_id: str = "",
    patchcore_n_bank_samples: int | None = None,
    patchcore_training_sources: list[str] | None = None,
) -> Path:
    """
    Write bundle_manifest.json next to model weights.

    Normalization stats are fit on normal-only rows in features_df.
    """
    models_dir = Path(models_dir or get_user_models_dir())
    models_dir.mkdir(parents=True, exist_ok=True)
    normal_mask = np.asarray(labels, dtype=int) == 0
    fit_df = features_df.loc[normal_mask] if normal_mask.any() else features_df
    feature_cols = select_feature_columns(features_df)
    norm_stats = compute_norm_stats(fit_df)

    manifest: dict[str, Any] = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset_id": training_dataset_id,
        "feature_config": feature_config_to_dict(feature_config),
        "feature_cols": feature_cols,
        "norm_stats": norm_stats,
        "exclude_cols": FEATURE_EXCLUDE_COLS,
        "aggregation": "dataset_max",
        "draem_threshold": draem_threshold,
        "patchcore_threshold": patchcore_threshold,
        "ensemble_strategy": ensemble_strategy,
        "draem_model": get_default_draem_model_path().name,
        "patchcore_model": get_default_patchcore_model_path().name,
    }
    if patchcore_n_bank_samples is not None:
        manifest["patchcore_n_bank_samples"] = int(patchcore_n_bank_samples)
    if patchcore_training_sources:
        manifest["patchcore_training_sources"] = list(patchcore_training_sources)
    path = get_bundle_manifest_path(models_dir)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_model_bundle(models_dir: Path | None = None) -> dict[str, Any]:
    """Load and validate bundle_manifest.json."""
    models_dir = Path(models_dir or get_user_models_dir())
    path = get_bundle_manifest_path(models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle manifest not found: {path}\n"
            "Train models in ML tab or run analyze_crack_detection.py to create bundle_manifest.json."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    draem_path = models_dir / manifest.get("draem_model", "draem_model.pt")
    if not draem_path.exists():
        legacy = models_dir / "draem_model.pt"
        if legacy.exists():
            manifest["_legacy_draem_path"] = str(legacy)
        else:
            raise FileNotFoundError(f"DRAEM weights not found: {draem_path}")
    pc_path = models_dir / manifest.get("patchcore_model", "patchcore_model.npz")
    if not pc_path.exists():
        raise FileNotFoundError(f"PatchCore weights not found: {pc_path}")
    manifest["_models_dir"] = str(models_dir)
    manifest["_manifest_path"] = str(path)
    return manifest


def transform_with_bundle(
    features_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> pd.DataFrame:
    """Return normalized feature matrix columns in manifest order."""
    feature_cols: list[str] = manifest["feature_cols"]
    norm_stats: dict[str, dict[str, float]] = manifest["norm_stats"]
    if manifest.get("feature_enrichment_version") == "motion_geometry_v1":
        try:
            from bending_inspector.inspection_lab.motion_geometry_features import add_motion_geometry_features

            features_df = add_motion_geometry_features(features_df)
        except Exception:
            pass
    normed = apply_norm_stats(features_df, norm_stats, feature_cols)
    return normed[feature_cols].fillna(0.0)
