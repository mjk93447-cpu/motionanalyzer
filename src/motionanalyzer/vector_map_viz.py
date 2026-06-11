"""Shared helpers for 2D vector map visualization (metric-mm display)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

# Dark gradients for white GUI backgrounds — avoid pale low stops that wash out.
CMAP_FRAME = LinearSegmentedColormap.from_list(
    "vector_frame",
    ["#123c69", "#4c1d95", "#9d174d", "#7f1d1d"],
)
CMAP_SPEED = LinearSegmentedColormap.from_list(
    "vector_speed",
    ["#075f5f", "#047857", "#1d4ed8", "#5b21b6"],
)
CMAP_ACCEL = LinearSegmentedColormap.from_list(
    "vector_accel",
    ["#9a3412", "#b91c1c", "#6d28d9", "#111827"],
)

P98_PERCENTILE = 98.0
PANEL_FIGSIZE = (9.2, 8.0)
PANEL_DPI = 150
QUIVER_ALPHA_VELOCITY = 0.88
QUIVER_ALPHA_ACCEL = 0.9
QUIVER_ALPHA_VELOCITY_FAINT = 0.22
QUIVER_WIDTH_VELOCITY = 0.0024
QUIVER_WIDTH_ACCEL = 0.0015
QUIVER_WIDTH_VELOCITY_FAINT = 0.0010
POINTS_PANEL_MARKER_SIZE = 1.75
VELOCITY_BACKGROUND_MARKER_SIZE = 0.75
VELOCITY_TAIL_MARKER_SIZE = 4.5
VELOCITY_CURRENT_MARKER_SIZE = 1.25
ACCEL_TAIL_MARKER_SIZE = 4.0
ACCEL_HIGHLIGHT_MARKER_SIZE = 18.0
ACCEL_QUIVER_HEADWIDTH = 2.4
ACCEL_QUIVER_HEADLENGTH = 3.2
ACCEL_QUIVER_HEADAXISLENGTH = 2.8

# Arrow length = (mm/s)·Δt (mm); color = speed (mm/s).
VELOCITY_LENGTH_LABEL = "length = v·Δt (mm),  v in mm/s,  Δt = frame interval (s)"
VELOCITY_COLOR_LABEL = "speed (mm/s)"
# Arrow length = (mm/s²)·Δt² (mm); color = |a| (mm/s²).
ACCEL_LENGTH_LABEL = "length = a·Δt² (mm),  a in mm/s²,  Δt = frame interval (s)"
ACCEL_COLOR_LABEL = "acceleration (mm/s²)"

_HUE_STEP = 0.618033988749895


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


def index_color_map(unique_indices: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    """Golden-ratio hue spacing so adjacent index numbers look distinct."""
    out: dict[int, tuple[float, float, float, float]] = {}
    for i, idx in enumerate(np.sort(unique_indices)):
        hue = (i * _HUE_STEP) % 1.0
        r, g, b = mcolors.hsv_to_rgb((hue, 0.82, 0.72))
        out[int(idx)] = (float(r), float(g), float(b), 1.0)
    return out


def velocity_step_components_mm(
    vx_mm_s: np.ndarray,
    vy_mm_s: np.ndarray,
    dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame displacement from velocity: (mm/s)·Δt → mm (chains tip-to-tail)."""
    dt = np.asarray(dt_s, dtype=float)
    vx = np.asarray(vx_mm_s, dtype=float)
    vy = np.asarray(vy_mm_s, dtype=float)
    return vx * dt, -vy * dt


def acceleration_step_components_mm(
    ax_mm_s2: np.ndarray,
    ay_mm_s2: np.ndarray,
    dt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame displacement from acceleration: (mm/s²)·Δt² → mm."""
    dt = np.asarray(dt_s, dtype=float)
    ax = np.asarray(ax_mm_s2, dtype=float)
    ay = np.asarray(ay_mm_s2, dtype=float)
    return ax * dt * dt, -ay * dt * dt


def quiver_tail_positions(
    end_x_mm: np.ndarray,
    end_y_mm_plot: np.ndarray,
    step_x_mm: np.ndarray,
    step_y_mm_plot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrow tail = end position minus per-frame step (previous frame)."""
    return (
        np.asarray(end_x_mm, dtype=float) - np.asarray(step_x_mm, dtype=float),
        np.asarray(end_y_mm_plot, dtype=float) - np.asarray(step_y_mm_plot, dtype=float),
    )


# Backward-compatible aliases
def displacement_quiver_components(dx_mm: np.ndarray, dy_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(dx_mm, dtype=float), -np.asarray(dy_mm, dtype=float)


def displacement_quiver_origins(
    x_mm: np.ndarray,
    y_mm_plot: np.ndarray,
    dx_mm: np.ndarray,
    dy_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    step_x, step_y = displacement_quiver_components(dx_mm, dy_mm)
    return quiver_tail_positions(x_mm, y_mm_plot, step_x, step_y)


def normalized_quiver_components(
    u: np.ndarray,
    v: np.ndarray,
    magnitude: np.ndarray,
    *,
    p98: float | None = None,
    target_fraction: float = 0.035,
    axis_span: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy display scaling (prefer velocity_step_components_mm for metric maps)."""
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
    dt = np.asarray(dt_s, dtype=float)
    if dt.ndim == 0:
        dt = float(dt)
    return vx * dt, vy * dt


def attach_colorbar(fig: Figure, ax: Any, mappable: Any, label: str) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.8%", pad=0.22)
    fig.colorbar(mappable, cax=cax, label=label)


def style_metric_axis(ax: Any, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.grid(True, alpha=0.22, linestyle="--", color="#94a3b8")
    ax.set_facecolor("#fafbfc")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def scatter_quiver_tails(
    ax: Any,
    tail_x: np.ndarray,
    tail_y: np.ndarray,
    indices: np.ndarray,
    index_colors: dict[int, tuple[float, float, float, float]],
    *,
    size: float = 16.0,
) -> None:
    """Mark each vector tail with an index-colored point."""
    colors = np.array([index_colors[int(i)] for i in indices], dtype=float)
    ax.scatter(
        tail_x,
        tail_y,
        c=colors,
        s=size,
        alpha=0.95,
        edgecolors="#1f2937",
        linewidths=0.35,
        zorder=4,
    )


def save_vector_map_panel(fig: Figure, path: Any) -> None:
    fig.subplots_adjust(left=0.11, right=0.74, top=0.90, bottom=0.10)
    fig.savefig(path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.28)


@dataclass(frozen=True)
class VectorMapRenderData:
    df: pd.DataFrame
    moving: pd.DataFrame
    index_colors: dict[int, tuple[float, float, float, float]]
    frame_norm: Normalize
    position_unit: str
    speed_unit: str
    accel_unit: str
    vel_tail_x: np.ndarray
    vel_tail_y: np.ndarray
    vel_qx: np.ndarray
    vel_qy: np.ndarray
    acc_qx: np.ndarray
    acc_qy: np.ndarray
    speed_mm_s: np.ndarray
    acc_mm_s2: np.ndarray
    speed_vmax: float
    acc_vmax: float
    max_index: int


def prepare_vector_map_render_data(
    df: pd.DataFrame,
    *,
    meters_per_pixel: float,
    display_scale: float,
    speed_col: str,
    acc_col: str,
    ax_col: str,
    ay_col: str,
    frame_min: int,
) -> VectorMapRenderData:
    work = df.copy()
    work["plot_y"] = -work["y"].astype(float)
    work["plot_x_mm"] = work["x"].astype(float) * float(meters_per_pixel) * display_scale
    work["plot_y_mm"] = work["plot_y"].astype(float) * float(meters_per_pixel) * display_scale
    work["speed_mm_s"] = work[speed_col].astype(float) * display_scale
    work["acc_mm_s2"] = work[acc_col].astype(float) * display_scale
    work["vx_mm_s"] = work["vx_si"].astype(float) * display_scale
    work["vy_mm_s"] = work["vy_si"].astype(float) * display_scale
    work["ax_mm_s2"] = work[ax_col].astype(float) * display_scale
    work["ay_mm_s2"] = work[ay_col].astype(float) * display_scale
    if "dt_s" not in work.columns:
        work["dt_s"] = 1.0 / 30.0
    work["dt_s"] = work["dt_s"].fillna(work["dt_s"].median()).astype(float)

    moving = work[work["frame"] > frame_min].copy()
    dt = moving["dt_s"].to_numpy(float)
    vel_qx, vel_qy = velocity_step_components_mm(
        moving["vx_mm_s"].to_numpy(float),
        moving["vy_mm_s"].to_numpy(float),
        dt,
    )
    acc_qx, acc_qy = acceleration_step_components_mm(
        moving["ax_mm_s2"].to_numpy(float),
        moving["ay_mm_s2"].to_numpy(float),
        dt,
    )
    vel_tail_x, vel_tail_y = quiver_tail_positions(
        moving["plot_x_mm"].to_numpy(float),
        moving["plot_y_mm"].to_numpy(float),
        vel_qx,
        vel_qy,
    )
    speed = moving["speed_mm_s"].to_numpy(float)
    acc = moving["acc_mm_s2"].to_numpy(float)
    return VectorMapRenderData(
        df=work,
        moving=moving,
        index_colors=index_color_map(work["index"].unique()),
        frame_norm=Normalize(float(work["frame"].min()), float(work["frame"].max())),
        position_unit="mm",
        speed_unit="mm/s",
        accel_unit="mm/s^2",
        vel_tail_x=vel_tail_x,
        vel_tail_y=vel_tail_y,
        vel_qx=vel_qx,
        vel_qy=vel_qy,
        acc_qx=acc_qx,
        acc_qy=acc_qy,
        speed_mm_s=speed,
        acc_mm_s2=acc,
        speed_vmax=p98_positive(speed),
        acc_vmax=p98_positive(acc),
        max_index=int(work["index"].max()),
    )


def render_points_panel(fig: Figure, ax: Any, data: VectorMapRenderData) -> None:
    for idx, group in data.df.groupby("index", sort=True):
        color = data.index_colors[int(idx)]
        if len(group) >= 2:
            ax.plot(
                group["plot_x_mm"],
                group["plot_y_mm"],
                color=color,
                alpha=0.42,
                linewidth=0.9,
                zorder=2,
            )
    scatter = ax.scatter(
        data.df["plot_x_mm"],
        data.df["plot_y_mm"],
        c=data.df["frame"],
        cmap=CMAP_FRAME,
        norm=data.frame_norm,
        s=POINTS_PANEL_MARKER_SIZE,
        alpha=0.9,
        edgecolors="#1f2937",
        linewidths=0.15,
        zorder=3,
    )
    attach_colorbar(fig, ax, scatter, "frame #")
    style_metric_axis(
        ax,
        xlabel=f"X ({data.position_unit})",
        ylabel=f"Y image-up ({data.position_unit})",
        title=f"All frame points | line color = index (1–{data.max_index}), point color = frame",
    )


def render_velocity_panel(fig: Figure, ax: Any, data: VectorMapRenderData) -> None:
    if data.moving.empty:
        style_metric_axis(ax, xlabel="X (mm)", ylabel="Y (mm)", title="Velocity (no motion rows)")
        return

    ax.scatter(
        data.df["plot_x_mm"],
        data.df["plot_y_mm"],
        c=data.df["frame"],
        cmap=CMAP_FRAME,
        norm=data.frame_norm,
        s=VELOCITY_BACKGROUND_MARKER_SIZE,
        alpha=0.18,
        edgecolors="none",
        zorder=1,
    )
    scatter_quiver_tails(
        ax,
        data.vel_tail_x,
        data.vel_tail_y,
        data.moving["index"].to_numpy(int),
        data.index_colors,
        size=VELOCITY_TAIL_MARKER_SIZE,
    )
    q = ax.quiver(
        data.vel_tail_x,
        data.vel_tail_y,
        data.vel_qx,
        data.vel_qy,
        np.clip(data.speed_mm_s, 0, data.speed_vmax),
        cmap=CMAP_SPEED,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=QUIVER_WIDTH_VELOCITY,
        alpha=QUIVER_ALPHA_VELOCITY,
        zorder=3,
    )
    ax.scatter(
        data.moving["plot_x_mm"],
        data.moving["plot_y_mm"],
        c=data.moving["index"].map(data.index_colors),
        s=VELOCITY_CURRENT_MARKER_SIZE,
        alpha=0.55,
        edgecolors="none",
        zorder=2,
    )
    attach_colorbar(fig, ax, q, VELOCITY_COLOR_LABEL)
    style_metric_axis(
        ax,
        xlabel=f"X ({data.position_unit})",
        ylabel=f"Y image-up ({data.position_unit})",
        title=f"Velocity | {VELOCITY_LENGTH_LABEL}\n"
        f"tail ● = index color, arrow color = {VELOCITY_COLOR_LABEL}",
    )


def render_acceleration_panel(fig: Figure, ax: Any, data: VectorMapRenderData) -> None:
    if data.moving.empty:
        style_metric_axis(ax, xlabel="X (mm)", ylabel="Y (mm)", title="Acceleration (no motion rows)")
        return

    from matplotlib.collections import LineCollection

    segments: list[np.ndarray] = []
    seg_colors: list[float] = []
    for idx, group in data.df.groupby("index", sort=True):
        group = group.sort_values("frame")
        points = group[["plot_x_mm", "plot_y_mm"]].to_numpy(float)
        if len(points) < 2:
            continue
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments.extend(segs)
        seg_colors.extend(group["acc_mm_s2"].iloc[1:].to_numpy(float).tolist())

    if segments:
        lc = LineCollection(
            segments,
            cmap=CMAP_ACCEL,
            norm=Normalize(0, data.acc_vmax),
            linewidths=1.0,
            alpha=0.45,
            zorder=1,
        )
        lc.set_array(np.clip(np.asarray(seg_colors), 0, data.acc_vmax))
        ax.add_collection(lc)

    scatter_quiver_tails(
        ax,
        data.vel_tail_x,
        data.vel_tail_y,
        data.moving["index"].to_numpy(int),
        data.index_colors,
        size=ACCEL_TAIL_MARKER_SIZE,
    )
    ax.quiver(
        data.vel_tail_x,
        data.vel_tail_y,
        data.vel_qx,
        data.vel_qy,
        np.clip(data.speed_mm_s, 0, data.speed_vmax),
        cmap=CMAP_SPEED,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=QUIVER_WIDTH_VELOCITY_FAINT,
        alpha=QUIVER_ALPHA_VELOCITY_FAINT,
        zorder=2,
    )
    aq = ax.quiver(
        data.vel_tail_x,
        data.vel_tail_y,
        data.acc_qx,
        data.acc_qy,
        np.clip(data.acc_mm_s2, 0, data.acc_vmax),
        cmap=CMAP_ACCEL,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=QUIVER_WIDTH_ACCEL,
        headwidth=ACCEL_QUIVER_HEADWIDTH,
        headlength=ACCEL_QUIVER_HEADLENGTH,
        headaxislength=ACCEL_QUIVER_HEADAXISLENGTH,
        alpha=QUIVER_ALPHA_ACCEL,
        zorder=4,
    )
    attach_colorbar(fig, ax, aq, ACCEL_COLOR_LABEL)

    hottest = data.moving.loc[data.moving["acc_mm_s2"].idxmax()]
    ax.scatter(
        [hottest["plot_x_mm"]],
        [hottest["plot_y_mm"]],
        s=ACCEL_HIGHLIGHT_MARKER_SIZE,
        c="#7f1d1d",
        edgecolors="#111111",
        linewidths=0.9,
        zorder=5,
    )
    axis_span = max(
        float(data.df["plot_x_mm"].max() - data.df["plot_x_mm"].min()) or 1.0,
        float(data.df["plot_y_mm"].max() - data.df["plot_y_mm"].min()) or 1.0,
        1.0,
    )
    ax.annotate(
        f"max |a|\nframe={int(hottest['frame'])}, idx={int(hottest['index'])}\n"
        f"{float(hottest['acc_mm_s2']):.3f} {data.accel_unit}",
        xy=(float(hottest["plot_x_mm"]), float(hottest["plot_y_mm"])),
        xytext=(
            float(hottest["plot_x_mm"]) + axis_span * 0.08,
            float(hottest["plot_y_mm"]) + axis_span * 0.08,
        ),
        fontsize=8,
        color="#4b0f16",
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#4b0f16"},
        zorder=6,
    )
    style_metric_axis(
        ax,
        xlabel=f"X ({data.position_unit})",
        ylabel=f"Y image-up ({data.position_unit})",
        title=f"Acceleration | {ACCEL_LENGTH_LABEL}\n"
        f"bold arrow = {ACCEL_COLOR_LABEL}, faint = velocity (mm/s), tail ● = index",
    )
