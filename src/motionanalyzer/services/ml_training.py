"""Public service APIs for ML training/evaluation orchestration.

Scripts should use these wrappers instead of importing private `_run_*` symbols.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from motionanalyzer.gui.runners import (
    _run_dream,
    _run_ensemble,
    _run_patchcore,
    _run_temporal,
)


def run_dream_training(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    **options: Any,
) -> dict[str, Any]:
    return _run_dream(features_df, labels, log=log, progress=progress, **options)


def run_patchcore_training(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    **options: Any,
) -> dict[str, Any]:
    return _run_patchcore(features_df, labels, log=log, progress=progress, **options)


def run_ensemble_training(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    **options: Any,
) -> dict[str, Any]:
    return _run_ensemble(features_df, labels, log=log, progress=progress, **options)


def run_temporal_training(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    log: Callable[[str], None],
    progress: Callable[[], None],
    **options: Any,
) -> dict[str, Any]:
    return _run_temporal(features_df, labels, log=log, progress=progress, **options)
