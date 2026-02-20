"""Tests for DREAM (DRAEM strategy): discriminative head and synthetic anomaly training."""
import numpy as np
import pytest


def _has_torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
def test_dream_with_discriminative_fit_predict() -> None:
    from motionanalyzer.ml_models.dream import DREAMPyTorch

    np.random.seed(42)
    X = np.random.randn(50, 6).astype(np.float32) * 0.2
    model = DREAMPyTorch(
        input_dim=6,
        hidden_dims=[16, 8],
        latent_dim=4,
        use_discriminative=True,
        synthetic_noise_std=0.25,
        batch_size=8,
    )
    model.fit(X, epochs=5)
    model.set_threshold_from_normal(X, percentile=95.0)
    scores = model.predict(X)
    assert scores.shape == (50,)
    # With discriminator, scores combine reconstruction error and discriminator confidence.
    # Scores are non-negative and are not constrained to [0, 1].
    assert np.all(scores >= 0)
    pred = model.predict_binary(X)
    assert pred.shape == (50,)
    assert set(np.unique(pred)).issubset({0, 1})


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
def test_dream_without_discriminative_reconstruction_score() -> None:
    from motionanalyzer.ml_models.dream import DREAMPyTorch

    np.random.seed(43)
    X = np.random.randn(30, 4).astype(np.float32) * 0.15
    model = DREAMPyTorch(
        input_dim=4,
        use_discriminative=False,
        batch_size=8,
    )
    model.fit(X, epochs=4)
    scores = model.predict(X)
    assert scores.shape == (30,)
    # Reconstruction error (MSE) is non-negative
    assert np.all(scores >= 0)


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
def test_dream_save_load_preserves_discriminative() -> None:
    import tempfile
    from pathlib import Path
    from motionanalyzer.ml_models.dream import DREAMPyTorch

    np.random.seed(44)
    X = np.random.randn(20, 5).astype(np.float32) * 0.1
    model = DREAMPyTorch(input_dim=5, use_discriminative=True, batch_size=4)
    model.fit(X, epochs=2)
    scores_before = model.predict(X[:3])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dream.pt"
        model.save(path)
        model2 = DREAMPyTorch(input_dim=5)
        model2.load(path)
        scores_after = model2.predict(X[:3])
        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-5)
