"""Shared helpers for 2D vector map visualization (metric-mm display)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Colormaps tuned for white backgrounds. Avoid very bright yellow/white highs.
CMAP_FRAME = "Blues"
CMAP_SPEED = "viridis"
CMAP_ACCEL = "Reds"

P98_PERCENTILE = 98.0
QUIVER_ALPHA_VELOCITY = 0.72
QUIVER_ALPHA_ACCEL = 0.92
QUIVER_ALPHA_VELOCITY_FAINT = 0.08
QUIVER_WIDTH_VELOCITY = 0.0020
QUIVER_WIDTH_ACCEL = 0.0033
QUIVER_WIDTH_VELOCITY_FAINT = 0.0007


def use_si_columns(df: pd.DataFrame, meters_per_pixel: float | None) -> bool:
    return (
        meters_per_pixel is not None
        and float(meters_per_pixel) > 0
        and "speed_si" in df.columns
        and "acceleration_si" in df.columns
    )


def motion_field_names(use_si: bool) -> dict[str, str | float]:
    if use_si:
        return {
            "speed": "speed_si",
            "ax": "ax_si",
            "ay": "ay_si",
            "acceleration": "acceleration_si",
            "position_unit": "mm",
            "speed_unit": "mm/s",
            "accel_unit": "mm/s^2",
            "display_scale": 1000.0,
        }
    return {
        "speed": "speed",
        "ax": "ax",
        "ay": "ay",
        "acceleration": "acceleration",
        "position_unit": "mm",
        "speed_unit": "mm/s",
        "accel_unit": "mm/s^2",
        "display_scale": 0.0,
    }


def p98_positive(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(arr), P98_PERCENTILE)), 1e-9)


def normalized_quiver_components(
    u: np.ndarray,
    v: np.ndarray,
    magnitude: np.ndarray,
    *,
    p98: float | None = None,
    target_fraction: float = 0.035,
    axis_span: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale vector components so typical arrows share a comparable display length."""
    mag = np.asarray(magnitude, dtype=float)
    cap = p98 if p98 is not None else p98_positive(mag)
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    hyp = np.hypot(u_arr, v_arr)
    safe_hyp = np.where(hyp > 1e-12, hyp, 1.0)
    unit_u = u_arr / safe_hyp
    unit_v = v_arr / safe_hyp
    if axis_span is not None and axis_span > 0:
        target_len = axis_span * target_fraction
    else:
        target_len = cap * target_fraction if cap > 0 else 1.0
    scale = np.clip(mag, 0.0, cap) / cap
    return unit_u * target_len * scale, unit_v * target_len * scale


def displacement_from_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    dt_s: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame-to-frame displacement for quiver (image-up y flip applied by caller on vy)."""
    dt = np.asarray(dt_s, dtype=float)
    if dt.ndim == 0:
        dt = float(dt)
    return vx * dt, vy * dt
