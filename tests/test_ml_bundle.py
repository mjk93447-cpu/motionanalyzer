"""Tests for model bundle manifest (feature cols, norm stats)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from motionanalyzer.auto_optimize import PAPER_FEATURE_CONFIG
from motionanalyzer.ml_bundle import (
    feature_config_from_dict,
    feature_config_to_dict,
    save_model_bundle,
    transform_with_bundle,
)


def test_feature_config_round_trip() -> None:
    d = feature_config_to_dict(PAPER_FEATURE_CONFIG)
    restored = feature_config_from_dict(d)
    assert restored.include_advanced_stats == PAPER_FEATURE_CONFIG.include_advanced_stats
    assert restored.include_frequency_domain == PAPER_FEATURE_CONFIG.include_frequency_domain


def test_save_load_manifest_and_transform(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "draem_model.pt").write_bytes(b"stub")
    (models_dir / "patchcore_model.npz").write_bytes(b"stub")

    n = 20
    features_df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "f_speed_mean": np.random.randn(n),
            "f_accel_std": np.random.randn(n),
            "label": [0] * 15 + [1] * 5,
        }
    )
    labels = features_df["label"].to_numpy()
    path = save_model_bundle(
        models_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        features_df=features_df,
        labels=labels,
        draem_threshold=0.5,
        patchcore_threshold=0.3,
        ensemble_strategy="both_agree",
        training_dataset_id="test_ds",
    )
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["ensemble_strategy"] == "both_agree"
    assert "feature_cols" in raw
    assert len(raw["norm_stats"]) == len(raw["feature_cols"])

    manifest = json.loads((models_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    normed = transform_with_bundle(features_df, manifest)
    assert list(normed.columns) == manifest["feature_cols"]
    assert len(normed) == n
