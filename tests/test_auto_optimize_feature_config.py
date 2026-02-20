from __future__ import annotations

from pathlib import Path

import numpy as np

from motionanalyzer.auto_optimize import FeatureExtractionConfig, prepare_training_data
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def test_feature_extraction_can_exclude_crack_risk(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    crack_dir = tmp_path / "crack"
    generate_synthetic_bundle(
        normal_dir,
        SyntheticConfig(frames=30, points_per_frame=80, fps=30.0, seed=1, scenario="normal"),
    )
    generate_synthetic_bundle(
        crack_dir,
        SyntheticConfig(frames=30, points_per_frame=80, fps=30.0, seed=2, scenario="crack"),
    )

    features_df, labels = prepare_training_data(
        normal_datasets=[normal_dir],
        crack_datasets=[crack_dir],
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,
        ),
    )

    assert len(features_df) == len(labels)
    assert isinstance(labels, np.ndarray)
    assert not any("crack_risk" in c for c in features_df.columns), features_df.columns.tolist()

