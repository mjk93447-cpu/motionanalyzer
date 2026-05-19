"""Ensemble BOTH_AGREE matches paper pipeline (logical AND)."""

from __future__ import annotations

import numpy as np


def test_both_agree_strategy() -> None:
    from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy

    class _Stub:
        def __init__(self, pred: np.ndarray) -> None:
            self._pred = pred

        def predict_binary(self, data: object) -> np.ndarray:
            return self._pred

        def predict(self, data: object) -> np.ndarray:
            return self._pred.astype(float)

    d_pred = np.array([1, 0, 1, 0])
    p_pred = np.array([1, 1, 0, 0])
    ens = EnsembleAnomalyDetector(
        draem_model=_Stub(d_pred),
        patchcore_model=_Stub(p_pred),
        strategy=EnsembleStrategy.BOTH_AGREE,
    )
    out = ens.predict_binary(np.zeros((4, 1)))
    np.testing.assert_array_equal(out, np.array([1, 0, 0, 0]))
