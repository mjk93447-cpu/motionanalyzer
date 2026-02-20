"""
Bayesian optimization of CrackModelParams using Optuna.

Maximizes AUC-ROC on per-dataset max crack_risk (same evaluation as grid search).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from motionanalyzer.crack_model import CrackModelParams


def run_bayesian_optimization(
    normal_paths: list[Path],
    crack_paths: list[Path],
    *,
    fps: float | None = None,
    n_trials: int = 20,
    base_params: CrackModelParams | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Run Optuna study to maximize AUC-ROC over CrackModelParams.

    Args:
        normal_paths: Paths to normal dataset directories.
        crack_paths: Paths to crack dataset directories.
        fps: FPS (if None, read from each dataset).
        n_trials: Number of Optuna trials.
        base_params: Base parameters; suggested values override. If None, use CrackModelParams().
        log: Optional log callback.
        progress: Optional progress callback.

    Returns:
        Dict with success, message, best_params, best_auc; or success=False if Optuna not installed.
    """
    try:
        import optuna
    except ImportError:
        return {
            "success": False,
            "message": "Bayesian optimization requires Optuna. Install with: pip install optuna",
        }

    from motionanalyzer.optimizers.grid_search import evaluate_params

    def no_log(_: str) -> None:
        pass

    log_fn = log or no_log

    if not normal_paths and not crack_paths:
        return {
            "success": False,
            "message": "At least one normal or crack dataset path is required.",
        }

    base = base_params or CrackModelParams()
    base_d = asdict(base)

    def objective(trial: optuna.Trial) -> float:
        override: dict[str, Any] = {}
        override["sigmoid_center"] = trial.suggest_float("sigmoid_center", 0.3, 0.7)
        override["sigmoid_steepness"] = trial.suggest_float("sigmoid_steepness", 4.0, 12.0)
        override["w_strain"] = trial.suggest_float("w_strain", 0.1, 0.4)
        override["w_stress"] = trial.suggest_float("w_stress", 0.1, 0.35)
        override["w_curvature_concentration"] = trial.suggest_float("w_curvature_concentration", 0.1, 0.35)
        override["w_bend_angle"] = trial.suggest_float("w_bend_angle", 0.05, 0.25)
        override["w_impact"] = trial.suggest_float("w_impact", 0.1, 0.35)

        d = dict(base_d)
        d.update(override)
        params = CrackModelParams(**d)
        auc, _, _ = evaluate_params(params, normal_paths, crack_paths, fps=fps)
        if progress:
            progress()
        return auc

    log_fn(f"Running Optuna study with n_trials={n_trials}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    override = dict(best_trial.params)
    d = dict(base_d)
    d.update(override)
    best_params = CrackModelParams(**d)

    log_fn(f"Best AUC-ROC: {study.best_value:.4f}")
    return {
        "success": True,
        "message": f"Bayesian optimization complete. Best AUC-ROC = {study.best_value:.4f}",
        "best_params": best_params,
        "best_auc": study.best_value,
        "best_trial_params": best_trial.params,
    }
