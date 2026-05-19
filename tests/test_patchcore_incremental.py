"""PatchCore incremental memory bank merge."""

from __future__ import annotations

import numpy as np
import pytest


def _has_sklearn() -> bool:
    try:
        import sklearn.neighbors  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_sklearn(), reason="scikit-learn not installed")
def test_fit_incremental_resamples_bank() -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    rng = np.random.default_rng(0)
    dim = 4
    base = rng.standard_normal((30, dim)).astype(np.float32)
    model = PatchCoreScikitLearn(feature_dim=dim, coreset_size=10, k_neighbors=1)
    model.fit(base)
    n0 = model.n_bank_samples
    extra = rng.standard_normal((40, dim)).astype(np.float32)
    model.fit_incremental(extra, source_tag="real")
    assert model.n_bank_samples == 10
    assert "real" in model.training_sources
    scores = model.predict(extra[:5])
    assert scores.shape == (5,)


@pytest.mark.skipif(not _has_sklearn(), reason="scikit-learn not installed")
def test_incremental_dim_mismatch_raises() -> None:
    import pytest
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    model = PatchCoreScikitLearn(feature_dim=3, coreset_size=5)
    model.fit(np.random.randn(10, 3).astype(np.float32))
    with pytest.raises(ValueError, match="mismatch"):
        model.fit_incremental(np.random.randn(5, 5).astype(np.float32))


@pytest.mark.skipif(not _has_sklearn(), reason="scikit-learn not installed")
def test_save_load_training_sources(tmp_path) -> None:
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    model = PatchCoreScikitLearn(feature_dim=2, coreset_size=5)
    model.fit(np.random.randn(8, 2).astype(np.float32))
    model.training_sources = ["scratch", "real"]
    path = tmp_path / "pc.npz"
    model.save(path)
    m2 = PatchCoreScikitLearn(feature_dim=2)
    m2.load(path)
    assert m2.n_bank_samples > 0
    assert "scratch" in m2.training_sources
