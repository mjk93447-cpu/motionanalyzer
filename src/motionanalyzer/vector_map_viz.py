"""Shared helpers for 2D vector map visualization (SI, normalization, colormaps)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Matplotlib colormap names tuned for white backgrounds (avoid bright yellow highs).
CMAP_FRAME = "cividis"
CMAP_SPEED = "cividis"
CMAP_ACCEL = "magma"

P98_PERCENTILE = 98.0
QUIVER_ALPHA_VELOCITY = 0.78
QUIVER_ALPHA_ACCEL = 0.82
QUIVER_ALPHA_VELOCITY_FAINT = 0.16
QUIVER_WIDTH_VELOCITY = 0.0022
QUIVER_WIDTH_ACCEL = 0.0024
QUIVER_WIDTH_VELOCITY_FAINT = 0.0009


def use_si_columns(df: pd.DataFrame, meters_per_pixel: float | None) -> bool:
    return (
        meters_per_pixel is not None
        and float(meters_per_pixel) > 0
        and "speed_si" in df.columns
        and "acceleration_si" in df.columns
    )


def motion_field_names(use_si: bool) -> dict[str, str]:
    if use_si:
        return {
            "speed": "speed_si",
            "ax": "ax_si",
            "ay": "ay_si",
            "acceleration": "acceleration_si",
            "speed_unit": "m/s",
            "accel_unit": "m/s²",
        }
    return {
        "speed": "speed",
        "ax": "ax",
        "ay": "ay",
        "acceleration": "acceleration",
        "speed_unit": "px/s",
        "accel_unit": "px/s²",
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
