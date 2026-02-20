from __future__ import annotations

import numpy as np
import pandas as pd

from motionanalyzer.auto_optimize import normalize_features


def test_normalize_features_fit_on_normal_only_avoids_leakage() -> None:
    # Normal cluster near 0, anomaly cluster far away.
    df = pd.DataFrame(
        {
            "feature": [0.0, 0.0, 0.0, 100.0, 100.0],
            "label": [0, 0, 0, 1, 1],
            "dataset_path": ["a"] * 5,
        }
    )
    normal_df = df.loc[df["label"] == 0]

    norm = normalize_features(df, exclude_cols=["label", "dataset_path"], fit_df=normal_df)

    # If fit on normal-only and normal std is ~0, we collapse to 0.0 by design
    # (avoid exploding values due to near-zero std).
    assert np.allclose(norm.loc[df["label"] == 0, "feature"].to_numpy(), 0.0)


def test_normalize_features_fit_stats_applied_to_all_rows() -> None:
    df = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 10.0],
            "label": [0, 0, 0, 1],
            "dataset_path": ["a"] * 4,
        }
    )
    normal_df = df.loc[df["label"] == 0]

    norm = normalize_features(df, exclude_cols=["label", "dataset_path"], fit_df=normal_df)

    # Normal rows should have mean approx 0, std approx 1 (sample std)
    normal_vals = norm.loc[df["label"] == 0, "feature"].to_numpy(dtype=float)
    assert abs(float(normal_vals.mean())) < 1e-9
    assert abs(float(normal_vals.std(ddof=1)) - 1.0) < 1e-9

    # Anomaly row should be far on the normalized scale
    anomaly_val = float(norm.loc[df["label"] == 1, "feature"].iloc[0])
    assert anomaly_val > 5.0

