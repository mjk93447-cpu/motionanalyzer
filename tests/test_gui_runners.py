"""Tests for GUI model-mode runners (mode dispatch and separation)."""
import numpy as np
import pandas as pd

from motionanalyzer.gui.runners import (
    ALL_MODES,
    MODE_BAYESIAN,
    MODE_DREAM,
    MODE_ENSEMBLE,
    MODE_GRID_SEARCH,
    MODE_PATCHCORE,
    MODE_PHYSICS,
    MODE_TEMPORAL,
    run_training_or_optimization,
)


def test_all_modes_defined() -> None:
    assert set(ALL_MODES) == {
        MODE_PHYSICS,
        MODE_DREAM,
        MODE_PATCHCORE,
        MODE_ENSEMBLE,
        MODE_TEMPORAL,
        MODE_GRID_SEARCH,
        MODE_BAYESIAN,
    }


def test_unknown_mode_returns_failure() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "label": [0, 1]})
    labels = np.array([0, 1])
    result = run_training_or_optimization("unknown_mode", df, labels)
    assert result["success"] is False
    assert "Unknown mode" in result["message"]


def test_physics_mode_returns_success() -> None:
    df = pd.DataFrame({"f1": [0.1, 0.2], "label": [0, 1]})
    labels = np.array([0, 1])
    result = run_training_or_optimization(MODE_PHYSICS, df, labels)
    assert result["success"] is True


def test_grid_search_fails_without_dataset_paths() -> None:
    df = pd.DataFrame({"f1": [0.1, 0.2], "label": [0, 1]})
    labels = np.array([0, 1])
    result = run_training_or_optimization(MODE_GRID_SEARCH, df, labels)
    assert result["success"] is False
    assert "dataset path" in result["message"].lower()


def test_bayesian_fails_without_dataset_paths() -> None:
    df = pd.DataFrame({"f1": [0.1, 0.2], "label": [0, 1]})
    labels = np.array([0, 1])
    result = run_training_or_optimization(MODE_BAYESIAN, df, labels)
    assert result["success"] is False
    assert "dataset path" in result["message"].lower()


def test_patchcore_fails_with_insufficient_normal_samples() -> None:
    df = pd.DataFrame({"f1": [0.1], "f2": [0.2]})
    labels = np.array([0])
    result = run_training_or_optimization(MODE_PATCHCORE, df, labels)
    assert result["success"] is False
    assert "at least 2 normal" in result["message"].lower()


def test_patchcore_mode_with_sklearn_trains_and_returns_success() -> None:
    try:
        import sklearn.neighbors
    except ImportError:
        return  # skip if scikit-learn not installed
    np.random.seed(42)
    n_normal, n_crack = 25, 8
    dim = 4
    normal = np.random.randn(n_normal, dim).astype(np.float32) * 0.2
    crack = np.random.randn(n_crack, dim).astype(np.float32) * 0.5 + 0.8
    features = np.vstack([normal, crack])
    labels = np.array([0] * n_normal + [1] * n_crack)
    cols = [f"f{i}" for i in range(dim)]
    df = pd.DataFrame(features, columns=cols)

    result = run_training_or_optimization(MODE_PATCHCORE, df, labels, coreset_size=15, k_neighbors=1)
    assert result["success"] is True
    assert "model_path" in result
    assert str(result["model_path"]).endswith("patchcore_model.npz")
    if crack.size > 0:
        assert "crack_anomaly_rate" in result
        assert "crack_mean_score" in result


def test_dream_mode_with_torch_trains_and_returns_success() -> None:
    try:
        import torch
    except ImportError:
        return  # skip if PyTorch not installed
    np.random.seed(42)
    n_normal, n_crack = 30, 10
    dim = 5
    normal = np.random.randn(n_normal, dim).astype(np.float32) * 0.1
    crack = np.random.randn(n_crack, dim).astype(np.float32) * 0.5 + 1.0
    features = np.vstack([normal, crack])
    labels = np.array([0] * n_normal + [1] * n_crack)
    cols = [f"f{i}" for i in range(dim)] + ["label"]
    df = pd.DataFrame(np.hstack([features, labels.reshape(-1, 1)]), columns=cols)
    df = df.drop(columns=["label"])
    labels = np.array([0] * n_normal + [1] * n_crack)

    result = run_training_or_optimization(MODE_DREAM, df, labels, epochs=3)
    assert result["success"] is True
    assert "model_path" in result
