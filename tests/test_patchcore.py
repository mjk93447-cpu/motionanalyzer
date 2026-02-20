"""Tests for PatchCore anomaly detector (fit, predict, save/load)."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from motionanalyzer.ml_models.patchcore import (
    PatchCoreScikitLearn,
    _to_array,
)


def test_to_array_dataframe() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    arr = _to_array(df)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32


def test_to_array_ndarray() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr = _to_array(x)
    assert arr.shape == (2, 2)
    assert arr.dtype == np.float32


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_patchcore_fit_predict() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    rng = np.random.default_rng(42)
    normal = rng.standard_normal((50, 4)).astype(np.float32) * 0.5
    model = PatchCoreScikitLearn(feature_dim=4, coreset_size=20, k_neighbors=1)
    model.fit(normal)
    assert model.is_trained
    assert model.memory_bank is not None
    assert model.memory_bank.shape[0] <= 20

    scores = model.predict(normal)
    assert scores.shape == (50,)
    assert np.all(scores >= 0)
    # Normal samples should have relatively low scores
    assert scores.mean() < 5.0


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_patchcore_predict_binary_after_threshold() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    rng = np.random.default_rng(43)
    normal = rng.standard_normal((30, 3)).astype(np.float32) * 0.3
    model = PatchCoreScikitLearn(feature_dim=3, coreset_size=15, k_neighbors=2)
    model.fit(normal)
    model.set_threshold_from_normal(normal, percentile=90.0)
    assert model.anomaly_threshold is not None

    pred = model.predict_binary(normal)
    assert pred.shape == (30,)
    assert np.all((pred == 0) | (pred == 1))
    # Most normals should be classified normal at p90
    assert pred.sum() <= 5


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_patchcore_save_load() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    rng = np.random.default_rng(44)
    normal = rng.standard_normal((40, 5)).astype(np.float32) * 0.2
    model = PatchCoreScikitLearn(feature_dim=5, coreset_size=10, k_neighbors=1)
    model.fit(normal)
    model.set_threshold_from_normal(normal, percentile=95.0)
    scores_before = model.predict(normal[:3])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "patchcore.npz"
        model.save(path)
        assert path.exists()

        model2 = PatchCoreScikitLearn(feature_dim=5, coreset_size=10, k_neighbors=1)
        model2.load(path)
        assert model2.is_trained
        assert model2.memory_bank is not None
        assert model2.anomaly_threshold is not None

        scores_after = model2.predict(normal[:3])
        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_patchcore_fit_accepts_dataframe() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    df = pd.DataFrame({
        "f1": np.random.randn(20).astype(np.float32) * 0.1,
        "f2": np.random.randn(20).astype(np.float32) * 0.1,
    })
    model = PatchCoreScikitLearn(feature_dim=2, coreset_size=10, k_neighbors=1)
    model.fit(df)
    scores = model.predict(df)
    assert scores.shape == (20,)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_patchcore_predict_untrained_raises() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    model = PatchCoreScikitLearn(feature_dim=2, coreset_size=10, k_neighbors=1)
    with pytest.raises(ValueError, match="not trained"):
        model.predict(np.random.randn(5, 2).astype(np.float32))
