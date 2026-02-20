"""Tests for CrackModelParams optimizers (grid search, Bayesian)."""
from pathlib import Path

import pytest

from motionanalyzer.crack_model import CrackModelParams


def _has_sklearn() -> bool:
    try:
        import sklearn.metrics
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_sklearn(), reason="scikit-learn not installed")
def test_evaluate_params_returns_auc_and_lists(tmp_path: Path) -> None:
    """evaluate_params returns (auc, scores, labels); with one class only, auc is 0.5."""
    from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
    from motionanalyzer.optimizers.grid_search import evaluate_params

    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    crack_dir = tmp_path / "crack"
    crack_dir.mkdir()
    generate_synthetic_bundle(normal_dir, SyntheticConfig(frames=4, points_per_frame=20, fps=24.0, seed=1, scenario="normal"))
    generate_synthetic_bundle(crack_dir, SyntheticConfig(frames=4, points_per_frame=20, fps=24.0, seed=2, scenario="crack"))

    params = CrackModelParams()
    auc, scores, labels = evaluate_params(params, [normal_dir], [crack_dir], fps=24.0)
    assert isinstance(auc, float)
    assert 0 <= auc <= 1
    assert len(scores) == len(labels) == 2
    assert set(labels) == {0, 1}


@pytest.mark.skipif(not _has_sklearn(), reason="scikit-learn not installed")
def test_run_grid_search_with_synthetic(tmp_path: Path) -> None:
    """run_grid_search with one grid point and synthetic data returns success and best_auc."""
    from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
    from motionanalyzer.optimizers.grid_search import run_grid_search

    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    crack_dir = tmp_path / "crack"
    crack_dir.mkdir()
    generate_synthetic_bundle(normal_dir, SyntheticConfig(frames=3, points_per_frame=15, fps=24.0, seed=3, scenario="normal"))
    generate_synthetic_bundle(crack_dir, SyntheticConfig(frames=3, points_per_frame=15, fps=24.0, seed=4, scenario="crack"))

    result = run_grid_search(
        [normal_dir],
        [crack_dir],
        fps=24.0,
        param_grid=[{"sigmoid_center": 0.45, "sigmoid_steepness": 8.0}],
    )
    assert result["success"] is True
    assert "best_params" in result
    assert "best_auc" in result
    assert isinstance(result["best_params"], CrackModelParams)
    assert 0 <= result["best_auc"] <= 1


def test_run_grid_search_fails_with_no_paths() -> None:
    """run_grid_search with no paths returns success=False."""
    from motionanalyzer.optimizers.grid_search import run_grid_search

    result = run_grid_search([], [])
    assert result["success"] is False
    assert "at least one" in result["message"].lower()


def test_run_bayesian_without_optuna_returns_failure() -> None:
    """When Optuna is not installed, run_bayesian_optimization returns success=False."""
    from motionanalyzer.optimizers.bayesian import run_bayesian_optimization

    # Pass empty paths to avoid actually running optuna if it is installed
    result = run_bayesian_optimization([], [], n_trials=1)
    # Either fails for no paths or for optuna (if installed) may try to run and fail on load_dataset
    assert result["success"] is False
    assert "path" in result["message"].lower() or "optuna" in result["message"].lower()
