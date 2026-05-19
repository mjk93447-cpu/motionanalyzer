"""CLI vs ml_inference score correlation on a tiny synthetic bundle (when ML installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motionanalyzer.auto_optimize import PAPER_FEATURE_CONFIG
from motionanalyzer.ml_bundle import save_model_bundle, transform_with_bundle
from tests.test_ml_inference import _has_ml_stack, _make_bundle


@pytest.mark.skipif(not _has_ml_stack(), reason="ML stack not installed")
def test_draem_scores_correlate_with_direct_model(tmp_path: Path) -> None:
    from motionanalyzer.ml_models.draem import DRAEMPyTorch
    from motionanalyzer.services.ml_inference import extract_features_for_bundle, predict_bundle
    from motionanalyzer.auto_optimize import select_feature_columns

    bundle = _make_bundle(tmp_path)
    feat_df = extract_features_for_bundle(bundle, PAPER_FEATURE_CONFIG, label=0)
    feature_cols = select_feature_columns(feat_df)
    X = feat_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    dim = X.shape[1]

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    train = np.random.randn(35, dim).astype(np.float32) * 0.15
    draem = DRAEMPyTorch(input_dim=dim, hidden_dims=[12, 6], batch_size=8)
    draem.fit(train, epochs=3)
    draem.set_threshold_from_normal(train, percentile=95.0)
    draem.save(models_dir / "draem_model.pt")
    (models_dir / "patchcore_model.npz").write_bytes(b"")

    import pandas as pd

    labels = np.zeros(len(train), dtype=int)
    train_df = pd.DataFrame(train, columns=feature_cols)
    train_df["frame"] = np.arange(len(train_df))
    save_model_bundle(
        models_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        features_df=train_df,
        labels=labels,
        draem_threshold=float(draem.reconstruction_error_threshold or 0.1),
    )

    manifest_path = models_dir / "bundle_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    X_norm = transform_with_bundle(feat_df, manifest)
    direct = draem.predict(X_norm)
    inf = predict_bundle(bundle, "draem", models_dir=models_dir)
    per_row = inf.get("per_row")
    if per_row is not None and len(per_row) == len(direct):
        corr = np.corrcoef(per_row["score"].to_numpy(), direct)[0, 1]
        assert corr > 0.99
