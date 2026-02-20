"""
Physics-based FPCB copper crack prediction (2D side-view surrogate).

Models FPCB as layered organic film + copper composite. Combines:
- Bending geometry: bend angle, curvature, curvature concentration
- Strain surrogate: epsilon ~ t/(2R) from curvature (centerline tracking)
- Stress surrogate: sigma ~ E*epsilon (provisional E; no absolute truth without measured constants)
- Trajectory impact: impulse-like from acceleration (transient stress)
- Single risk score P(crack) in [0, 1] and per-point risk.

All thresholds and weights are PROVISIONAL until secure-site calibration.
Limitations: 2D surrogate; no full 3D stress state or material constants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Provisional thresholds (calibrate at secure site)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CrackModelParams:
    """Provisional parameters for crack risk aggregation. Calibrate with real crack data."""

    # Normalization caps (above these, contribution saturates at 1.0)
    strain_cap: float = 0.03  # typical FPCB strain limit order
    curvature_concentration_cap: float = 8.0
    bend_angle_cap_deg: float = 180.0
    impact_cap_px_s2: float = 5000.0  # acceleration surrogate cap (px/s²)

    # Weights for linear combination (before sigmoid). Sum need not be 1.
    w_strain: float = 0.25
    w_stress: float = 0.20  # stress surrogate ~ strain in normalized model
    w_curvature_concentration: float = 0.20
    w_bend_angle: float = 0.15
    w_impact: float = 0.20

    # Sigmoid steepness and center for P(crack)
    sigmoid_steepness: float = 8.0
    sigmoid_center: float = 0.45


def load_frame_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load frame_metrics.csv (frame, time_s, bend_angle_deg, curvature_concentration, est_max_strain)."""
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    for col in ("frame", "time_s", "bend_angle_deg", "curvature_concentration", "est_max_strain"):
        if col not in df.columns:
            return pd.DataFrame()
    return df


def _safe_divide(num: np.ndarray, den: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(num, fill, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def compute_strain_surrogate(
    vectors: pd.DataFrame,
    frame_metrics: pd.DataFrame,
) -> np.ndarray:
    """
    Per-point strain surrogate: epsilon ~ (t/2)*kappa.
    Uses frame-level est_max_strain and distributes by curvature_like ratio within frame.
    Units: dimensionless (strain).
    """
    if frame_metrics.empty or "curvature_like" not in vectors.columns:
        return np.zeros(len(vectors), dtype=float)

    work = vectors.copy()
    work = work.merge(
        frame_metrics[["frame", "est_max_strain"]],
        on="frame",
        how="left",
        suffixes=("", "_fm"),
    )
    if "est_max_strain" not in work.columns:
        return np.zeros(len(vectors), dtype=float)

    max_curv_per_frame = vectors.groupby("frame")["curvature_like"].transform("max")
    mean_curv_per_frame = vectors.groupby("frame")["curvature_like"].transform("mean")
    # Local curvature ratio: high curvature_like => high local strain
    curv_ratio = _safe_divide(
        vectors["curvature_like"].to_numpy(dtype=float),
        np.maximum(max_curv_per_frame.to_numpy(dtype=float), 1e-12),
        0.0,
    )
    est_max = work["est_max_strain"].to_numpy(dtype=float)
    strain_surrogate = curv_ratio * np.maximum(est_max, 0.0)
    return strain_surrogate


def compute_stress_surrogate(
    strain_surrogate: np.ndarray,
    E_eff: float = 1.0,
) -> np.ndarray:
    """
    Stress surrogate sigma ~ E_eff * epsilon (provisional; no absolute value without measured E).
    Returns normalized stress surrogate for risk aggregation.
    """
    return E_eff * np.maximum(strain_surrogate, 0.0)


def compute_impact_surrogate(
    vectors: pd.DataFrame,
    dt_s: float,
) -> np.ndarray:
    """
    Impulse-like impact from trajectory: |a|*dt per frame (transient stress proxy).
    Units: px/s (velocity change per step); we use magnitude only for normalization.
    """
    if "acceleration" not in vectors.columns:
        return np.zeros(len(vectors), dtype=float)
    return (vectors["acceleration"].to_numpy(dtype=float)) * dt_s


def compute_crack_risk(
    vectors: pd.DataFrame,
    frame_metrics: pd.DataFrame,
    dt_s: float,
    *,
    meters_per_pixel: Optional[float] = None,
    params: Optional[CrackModelParams] = None,
) -> pd.DataFrame:
    """
    Compute per-row crack risk P(crack) in [0, 1] from bending geometry, strain, stress surrogate, and impact.

    Adds columns: strain_surrogate, stress_surrogate, impact_surrogate, crack_risk.
    If frame_metrics is empty, crack_risk is based only on acceleration and curvature_like (fallback).
    """
    if params is None:
        params = CrackModelParams()

    work = vectors.copy()
    n = len(work)

    # --- Strain surrogate (per point) ---
    strain_surrogate = compute_strain_surrogate(work, frame_metrics)
    work["strain_surrogate"] = strain_surrogate

    # --- Stress surrogate (provisional E=1) ---
    stress_surrogate = compute_stress_surrogate(strain_surrogate, E_eff=1.0)
    work["stress_surrogate"] = stress_surrogate

    # --- Impact surrogate (impulse-like) ---
    impact_surrogate = compute_impact_surrogate(work, dt_s)
    work["impact_surrogate"] = impact_surrogate

    # --- Frame-level features (merge) ---
    if not frame_metrics.empty:
        work = work.merge(
            frame_metrics[["frame", "bend_angle_deg", "curvature_concentration", "est_max_strain"]],
            on="frame",
            how="left",
        )
        bend_deg = work["bend_angle_deg"].to_numpy(dtype=float)
        curv_conc = work["curvature_concentration"].to_numpy(dtype=float)
    else:
        bend_deg = np.zeros(n)
        curv_conc = np.ones(n)

    # --- Normalize each contribution to [0, 1] (cap and scale) ---
    strain_n = np.clip(strain_surrogate / max(params.strain_cap, 1e-9), 0.0, 1.0)
    stress_n = np.clip(stress_surrogate / max(params.strain_cap, 1e-9), 0.0, 1.0)  # same scale as strain
    curv_conc_n = np.clip(curv_conc / max(params.curvature_concentration_cap, 1e-9), 0.0, 1.0)
    bend_n = np.clip(bend_deg / max(params.bend_angle_cap_deg, 1e-9), 0.0, 1.0)

    accel = work["acceleration"].to_numpy(dtype=float) if "acceleration" in work.columns else np.zeros(n)
    impact_n = np.clip(accel / max(params.impact_cap_px_s2, 1e-9), 0.0, 1.0)

    # --- Weighted linear combination ---
    combined = (
        params.w_strain * strain_n
        + params.w_stress * stress_n
        + params.w_curvature_concentration * curv_conc_n
        + params.w_bend_angle * bend_n
        + params.w_impact * impact_n
    )
    # Normalize so max possible (if all components = 1) is 1.0
    w_sum = (
        params.w_strain + params.w_stress + params.w_curvature_concentration
        + params.w_bend_angle + params.w_impact
    )
    combined = combined / max(w_sum, 1e-9)
    # Sigmoid to get P(crack) in [0, 1]
    k = params.sigmoid_steepness
    x0 = params.sigmoid_center
    crack_risk = 1.0 / (1.0 + np.exp(-k * (combined - x0)))
    work["crack_risk"] = np.clip(crack_risk, 0.0, 1.0)

    return work


def crack_risk_global(vectors_with_risk: pd.DataFrame) -> float:
    """Single global P(crack) as max over all points (worst-case location)."""
    if "crack_risk" not in vectors_with_risk.columns:
        return 0.0
    return float(vectors_with_risk["crack_risk"].max())


def crack_risk_mean(vectors_with_risk: pd.DataFrame) -> float:
    """Mean P(crack) over all points."""
    if "crack_risk" not in vectors_with_risk.columns:
        return 0.0
    return float(vectors_with_risk["crack_risk"].mean())


def save_params(params: CrackModelParams, path: Path) -> None:
    """Save CrackModelParams to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")


def load_params(path: Path) -> CrackModelParams:
    """Load CrackModelParams from JSON file. Returns default if file not found."""
    if not path.exists():
        return CrackModelParams()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CrackModelParams(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid params file {path}: {exc}") from exc


def get_user_params_path() -> Path:
    """Get user-specific params file path (%APPDATA%/motionanalyzer/crack_model_params.json on Windows)."""
    import os
    if os.name == "nt":  # Windows
        appdata = os.getenv("APPDATA", os.path.expanduser("~"))
        return Path(appdata) / "motionanalyzer" / "crack_model_params.json"
    else:
        return Path.home() / ".config" / "motionanalyzer" / "crack_model_params.json"
