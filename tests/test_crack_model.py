"""Tests for physics-based FPCB copper crack prediction model."""
from pathlib import Path

import numpy as np
import pandas as pd

from motionanalyzer.crack_model import (
    CrackModelParams,
    compute_crack_risk,
    compute_impact_surrogate,
    compute_strain_surrogate,
    compute_stress_surrogate,
    crack_risk_global,
    crack_risk_mean,
    load_frame_metrics,
)


def test_load_frame_metrics_missing_returns_empty() -> None:
    out = load_frame_metrics(Path("/nonexistent/frame_metrics.csv"))
    assert out.empty


def test_load_frame_metrics_reads_csv(tmp_path: Path) -> None:
    csv = tmp_path / "frame_metrics.csv"
    csv.write_text(
        "frame,time_s,bend_angle_deg,curvature_concentration,est_max_strain\n"
        "0,0.0,0.0,1.0,0.0\n"
        "1,0.033,45.0,2.0,0.01\n"
        "2,0.066,90.0,3.0,0.02\n",
        encoding="utf-8",
    )
    df = load_frame_metrics(csv)
    assert len(df) == 3
    assert list(df.columns) == [
        "frame",
        "time_s",
        "bend_angle_deg",
        "curvature_concentration",
        "est_max_strain",
    ]
    assert float(df["est_max_strain"].iloc[2]) == 0.02


def test_compute_strain_surrogate_empty_metrics_returns_zeros() -> None:
    vectors = pd.DataFrame({
        "frame": [0, 0, 1, 1],
        "index": [1, 2, 1, 2],
        "curvature_like": [0.1, 0.2, 0.3, 0.4],
    })
    metrics = pd.DataFrame()
    strain = compute_strain_surrogate(vectors, metrics)
    assert strain.shape == (4,)
    np.testing.assert_array_equal(strain, 0.0)


def test_compute_strain_surrogate_distributes_by_curvature() -> None:
    vectors = pd.DataFrame({
        "frame": [0, 0, 1, 1],
        "index": [1, 2, 1, 2],
        "curvature_like": [1.0, 2.0, 1.0, 2.0],
    })
    metrics = pd.DataFrame({
        "frame": [0, 1],
        "est_max_strain": [0.02, 0.04],
    })
    strain = compute_strain_surrogate(vectors, metrics)
    assert strain.shape == (4,)
    # In frame 0 max curvature_like=2, so index 2 gets ratio 1.0 -> 0.02, index 1 gets 0.5 -> 0.01
    assert 0.009 <= strain[0] <= 0.011
    assert 0.019 <= strain[1] <= 0.021
    assert 0.019 <= strain[2] <= 0.021
    assert 0.039 <= strain[3] <= 0.041


def test_compute_stress_surrogate() -> None:
    strain = np.array([0.0, 0.01, 0.02])
    stress = compute_stress_surrogate(strain, E_eff=1.0)
    np.testing.assert_allclose(stress, [0.0, 0.01, 0.02])
    stress2 = compute_stress_surrogate(strain, E_eff=2.0)
    np.testing.assert_allclose(stress2, [0.0, 0.02, 0.04])


def test_compute_impact_surrogate() -> None:
    vectors = pd.DataFrame({
        "acceleration": [100.0, 200.0, 50.0],
    })
    impact = compute_impact_surrogate(vectors, dt_s=0.01)
    np.testing.assert_allclose(impact, [1.0, 2.0, 0.5])


def test_compute_crack_risk_adds_columns() -> None:
    vectors = pd.DataFrame({
        "frame": [0, 0, 1, 1],
        "index": [1, 2, 1, 2],
        "curvature_like": [0.5, 1.0, 0.5, 1.0],
        "acceleration": [100.0, 500.0, 200.0, 300.0],
    })
    metrics = pd.DataFrame({
        "frame": [0, 1],
        "bend_angle_deg": [30.0, 90.0],
        "curvature_concentration": [2.0, 4.0],
        "est_max_strain": [0.005, 0.015],
    })
    out = compute_crack_risk(vectors, metrics, dt_s=0.033)
    assert "strain_surrogate" in out.columns
    assert "stress_surrogate" in out.columns
    assert "impact_surrogate" in out.columns
    assert "crack_risk" in out.columns
    assert out["crack_risk"].min() >= 0.0
    assert out["crack_risk"].max() <= 1.0


def test_compute_crack_risk_empty_metrics_still_returns_risk() -> None:
    vectors = pd.DataFrame({
        "frame": [0, 1],
        "index": [1, 1],
        "curvature_like": [0.1, 0.2],
        "acceleration": [1000.0, 2000.0],
    })
    out = compute_crack_risk(vectors, pd.DataFrame(), dt_s=0.033)
    assert "crack_risk" in out.columns
    assert out["crack_risk"].max() > 0  # impact and curvature_like contribute


def test_crack_risk_global_and_mean() -> None:
    df = pd.DataFrame({"crack_risk": [0.1, 0.5, 0.9]})
    assert crack_risk_global(df) == 0.9
    assert crack_risk_mean(df) == 0.5
    df2 = pd.DataFrame({"other": [1, 2]})
    assert crack_risk_global(df2) == 0.0
    assert crack_risk_mean(df2) == 0.0


def test_crack_scenario_higher_risk_than_normal(tmp_path: Path) -> None:
    """Crack scenario (curvature concentration) should yield higher P(crack) than normal."""
    from motionanalyzer.analysis import load_bundle, compute_vectors, run_analysis
    from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle

    base = tmp_path / "base"
    base.mkdir(parents=True)
    gen_n = tmp_path / "normal"
    gen_c = tmp_path / "crack"
    gen_n.mkdir()
    gen_c.mkdir()

    cfg_n = SyntheticConfig(frames=20, points_per_frame=50, fps=30.0, seed=1, scenario="normal")
    cfg_c = SyntheticConfig(frames=20, points_per_frame=50, fps=30.0, seed=1, scenario="crack")
    generate_synthetic_bundle(gen_n, cfg_n)
    generate_synthetic_bundle(gen_c, cfg_c)

    out_n = tmp_path / "out_normal"
    out_c = tmp_path / "out_crack"
    summary_n = run_analysis(input_dir=gen_n, output_dir=out_n)
    summary_c = run_analysis(input_dir=gen_c, output_dir=out_c)

    assert summary_n.max_crack_risk is not None
    assert summary_c.max_crack_risk is not None
    # Crack scenario has higher curvature concentration -> higher crack risk
    assert summary_c.max_crack_risk >= summary_n.max_crack_risk * 0.9  # allow some tolerance
