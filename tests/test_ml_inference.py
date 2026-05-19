"""Tests for shared ml_inference service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motionanalyzer.auto_optimize import PAPER_FEATURE_CONFIG
from motionanalyzer.services.ml_inference import extract_features_for_bundle
from motionanalyzer.ml_bundle import save_model_bundle


def _write_frame(path: Path, values: list[tuple[int, int, int]]) -> None:
    lines = ["# x,y,index"]
    for x, y, idx in values:
        lines.append(f"{x},{y},{idx}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_bundle(tmp_path: Path, n_frames: int = 8) -> Path:
    bundle = tmp_path / "normal_001"
    bundle.mkdir()
    (bundle / "fps.txt").write_text("30\n", encoding="utf-8")
    for i in range(n_frames):
        _write_frame(
            bundle / f"frame_{i:05d}.txt",
            [(0, i, 1), (5 + i, i, 2), (10, i + 1, 3)],
        )
    return bundle


def test_extract_features_for_bundle_has_rows(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path)
    df = extract_features_for_bundle(bundle, PAPER_FEATURE_CONFIG, label=0)
    assert len(df) >= 1
    assert "frame" in df.columns


def _has_ml_stack() -> bool:
    try:
        import sklearn.neighbors  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_ml_stack(), reason="ML stack not installed")
def test_predict_bundle_draem_and_ensemble(tmp_path: Path) -> None:
    from motionanalyzer.ml_models.draem import DRAEMPyTorch
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    from motionanalyzer.services.ml_inference import predict_bundle

    bundle = _make_bundle(tmp_path)
    feat_df = extract_features_for_bundle(bundle, PAPER_FEATURE_CONFIG, label=0)
    from motionanalyzer.auto_optimize import select_feature_columns

    feature_cols = select_feature_columns(feat_df)
    X = feat_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    dim = X.shape[1]
    assert dim >= 1

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    np.random.seed(0)
    train = np.random.randn(40, dim).astype(np.float32) * 0.2
    labels = np.array([0] * 30 + [1] * 10)

    draem = DRAEMPyTorch(input_dim=dim, hidden_dims=[16, 8], batch_size=8)
    draem.fit(train, epochs=3)
    draem.set_threshold_from_normal(train[:30], percentile=95.0)
    draem.save(models_dir / "draem_model.pt")

    pc = PatchCoreScikitLearn(feature_dim=dim)
    pc.fit(train[:30])
    pc.set_threshold_from_normal(train[:30], percentile=95.0)
    pc.save(models_dir / "patchcore_model.npz")

    train_df = pd.DataFrame(train, columns=feature_cols)
    train_df["label"] = labels
    train_df["frame"] = np.arange(len(train_df))
    save_model_bundle(
        models_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        features_df=train_df,
        labels=labels,
        draem_threshold=float(draem.reconstruction_error_threshold or 0.1),
        patchcore_threshold=float(pc.anomaly_threshold or 0.1),
        ensemble_strategy="both_agree",
    )

    out_d = predict_bundle(bundle, "draem", models_dir=models_dir)
    assert "dataset_score" in out_d
    assert out_d["feature_cols"] == feature_cols

    out_e = predict_bundle(bundle, "ensemble", models_dir=models_dir)
    assert out_e.get("ensemble_strategy") == "both_agree"
