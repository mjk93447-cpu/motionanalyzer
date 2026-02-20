"""Vector map visualization with velocity, acceleration, and impact analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Golden-ratio step so adjacent indices get maximally different hues (≈0.618 apart)
_HUE_STEP = 0.618033988749895


def _index_color_map(unique_indices: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    """
    Return one unique RGBA color per index so adjacent indices are easy to tell apart.
    Uses golden-ratio hue spacing so consecutive index numbers get very different colors.
    """
    n = len(unique_indices)
    out = {}
    for i, idx in enumerate(unique_indices):
        hue = (i * _HUE_STEP) % 1.0
        r, g, b = mcolors.hsv_to_rgb((hue, 0.88, 0.96))
        out[int(idx)] = (r, g, b, 1.0)
    return out


def _complement_rgba(rgba: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Return complementary color (1-R, 1-G, 1-B) for distinction from base color."""
    r, g, b, a = rgba
    return (1.0 - r, 1.0 - g, 1.0 - b, a)


def _arrow_line_segments(
    x: float, y: float, u: float, v: float, scale: float, head_ratio: float = 0.2, head_angle: float = 0.4
) -> list[list[tuple[float, float]]]:
    """
    Build arrow as combination of lines: body + two head lines.
    Returns list of 3 segments [(x0,y0),(x1,y1)] for body and arrowhead.
    """
    tx = x + u * scale
    ty = y + v * scale
    body_len = np.hypot(u * scale, v * scale)
    head_len = body_len * head_ratio
    angle = np.arctan2(v, u)
    # Head lines point backward from tip
    al = angle + np.pi - head_angle
    ar = angle + np.pi + head_angle
    left = (tx + head_len * np.cos(al), ty + head_len * np.sin(al))
    right = (tx + head_len * np.cos(ar), ty + head_len * np.sin(ar))
    return [
        [(x, y), (tx, ty)],
        [(tx, ty), left],
        [(tx, ty), right],
    ]


def _crack_probability_heuristic(
    acceleration: float,
    max_acceleration: float,
    curvature_like: float,
    max_curvature: float,
) -> float:
    """
    Heuristic crack probability from acceleration and curvature surrogate.
    Returns value in [0, 1].
    """
    if max_acceleration <= 0 and max_curvature <= 0:
        return 0.0
    acc_norm = acceleration / max(max_acceleration, 1e-9)
    curv_norm = curvature_like / max(max_curvature, 1e-9)
    # Weighted combination; curvature often indicates sharp direction change
    combined = 0.5 * acc_norm + 0.5 * curv_norm
    return min(1.0, max(0.0, combined))


def plot_full_vector_map(
    vectors_csv: Path,
    output_image: Path,
    fps: float,
    *,
    meters_per_pixel: float | None = None,
    figsize: tuple[float, float] = (24, 18),
    dpi: int = 200,
    point_size: float = 3.5,
    point_alpha: float = 0.92,
    point_edge_width: float = 0.25,
    velocity_scale: float | None = None,
    velocity_linewidth: float = 0.55,
    acceleration_scale: float = 0.018,
    acceleration_linewidth: float = 0.24,
    velocity_alpha: float = 0.7,
    acceleration_alpha: float = 0.28,
) -> Path:
    """
    Plot velocity and acceleration vectors on a 2D graph with units and impact analysis.

    Visualization strategy so acceleration does not obscure velocity:
    - Draw order: points → acceleration (bottom) → velocity (top). Velocity is always on top.
    - Acceleration: shorter scale, thinner line, lower alpha so it stays in the background.
    - Alternative ideas: two-panel (velocity-only / acceleration-only), or color velocity by |a|.
    """
    df = pd.read_csv(vectors_csv)

    for col in ("x", "y", "index", "vx", "vy", "ax", "ay", "acceleration", "curvature_like"):
        if col not in df.columns:
            raise ValueError(f"vectors.csv must contain '{col}'")

    use_si = (
        meters_per_pixel is not None
        and meters_per_pixel > 0
        and "acceleration_si" in df.columns
    )
    vel_unit = "m/s" if use_si else "px/s"
    # Acceleration in km/s² when SI: shorter display length (values 1000× smaller) and clear unit
    acc_unit = "km/s²" if use_si else "px/s²"

    dt_s = 1.0 / fps
    # Velocity arrow length = actual displacement per frame (v*dt) so tip connects to same index in next frame
    if velocity_scale is None:
        velocity_scale = dt_s
    # When SI, draw acceleration in km/s² scale so arrows are shorter (1/1000 of m/s² scale)
    acc_scale_draw = acceleration_scale / 1000.0 if use_si else acceleration_scale

    m_per_px = float(meters_per_pixel) if use_si else 1.0
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    if use_si:
        x = x * m_per_px
        y = y * m_per_px
        velocity_scale = velocity_scale * m_per_px
        acc_scale_draw = acc_scale_draw * m_per_px

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")

    # --- 1) Points (x, y) colored by index (drawn first, clearly visible) ---
    unique_indices = np.sort(df["index"].unique())
    index_colors = _index_color_map(unique_indices)
    index_complement_colors = {i: _complement_rgba(index_colors[i]) for i in unique_indices}
    colors = np.array([index_colors[i] for i in df["index"]])
    ax.scatter(
        x,
        y,
        s=point_size,
        c=colors,
        alpha=point_alpha,
        edgecolors="white",
        linewidths=point_edge_width,
    )
    vx = df["vx"].to_numpy(np.float64)
    vy = df["vy"].to_numpy(np.float64)
    ax_arr = df["ax"].to_numpy(np.float64)
    ay_arr = df["ay"].to_numpy(np.float64)

    # --- 2) Acceleration first (bottom layer): short, faint, thin so velocity stays clear ---
    tax = x + ax_arr * acc_scale_draw
    tay = y + ay_arr * acc_scale_draw
    a_angle = np.arctan2(ay_arr, ax_arr)
    a_body_len = np.hypot(ax_arr * acc_scale_draw, ay_arr * acc_scale_draw)
    a_head_len = np.where(a_body_len > 1e-6, a_body_len * 0.22, 0.0)
    al_a, ar_a = a_angle + np.pi - 0.38, a_angle + np.pi + 0.38
    lax = tax + a_head_len * np.cos(al_a)
    lay = tay + a_head_len * np.sin(al_a)
    rax = tax + a_head_len * np.cos(ar_a)
    ray = tay + a_head_len * np.sin(ar_a)
    acc_segments = np.array([
        np.stack([np.stack([x, y], axis=1), np.stack([tax, tay], axis=1)], axis=0),
        np.stack([np.stack([tax, tay], axis=1), np.stack([lax, lay], axis=1)], axis=0),
        np.stack([np.stack([tax, tay], axis=1), np.stack([rax, ray], axis=1)], axis=0),
    ])
    acc_segments = np.transpose(acc_segments, (0, 2, 1, 3)).reshape(-1, 2, 2)
    acc_colors = np.array([index_complement_colors[i] for i in df["index"] for _ in range(3)])
    lc_acc = LineCollection(
        acc_segments,
        colors=acc_colors,
        linewidths=acceleration_linewidth,
        alpha=acceleration_alpha,
        capstyle="round",
    )
    ax.add_collection(lc_acc)

    # --- 3) Velocity on top (drawn last so it is never covered by acceleration) ---
    tx = x + vx * velocity_scale
    ty = y + vy * velocity_scale
    angle = np.arctan2(vy, vx)
    body_len = np.hypot(vx * velocity_scale, vy * velocity_scale)
    head_len = np.where(body_len > 1e-6, body_len * 0.22, 0.0)
    al, ar = angle + np.pi - 0.38, angle + np.pi + 0.38
    lx = tx + head_len * np.cos(al)
    ly = ty + head_len * np.sin(al)
    rx = tx + head_len * np.cos(ar)
    ry = ty + head_len * np.sin(ar)
    vel_segments = np.array([
        np.stack([np.stack([x, y], axis=1), np.stack([tx, ty], axis=1)], axis=0),
        np.stack([np.stack([tx, ty], axis=1), np.stack([lx, ly], axis=1)], axis=0),
        np.stack([np.stack([tx, ty], axis=1), np.stack([rx, ry], axis=1)], axis=0),
    ])
    n_pts = len(df)
    vel_segments = np.transpose(vel_segments, (0, 2, 1, 3)).reshape(-1, 2, 2)
    vel_colors = np.array([index_colors[i] for i in df["index"] for _ in range(3)])
    lc_vel = LineCollection(
        vel_segments,
        colors=vel_colors,
        linewidths=velocity_linewidth,
        alpha=velocity_alpha,
        capstyle="round",
    )
    ax.add_collection(lc_vel)

    # --- Crack risk: physics model if available, else heuristic at max acceleration ---
    if "crack_risk" in df.columns:
        risk_row = df.loc[df["crack_risk"].idxmax()]
        ann_x = float(risk_row["x"]) * m_per_px
        ann_y = float(risk_row["y"]) * m_per_px
        crack_prob = float(risk_row["crack_risk"])
        label_suffix = " (physics)"
    else:
        risk_row = df.loc[df["acceleration"].idxmax()]
        ann_x = float(risk_row["x"]) * m_per_px
        ann_y = float(risk_row["y"]) * m_per_px
        max_acc = float(risk_row["acceleration"])
        max_curv = float(df["curvature_like"].max())
        crack_prob = _crack_probability_heuristic(
            max_acc,
            float(df["acceleration"].max()),
            float(risk_row["curvature_like"]),
            max_curv,
        )
        label_suffix = ""

    accel_label = (
        f"{float(risk_row['acceleration_si']) / 1000.0:.6f} {acc_unit}"
        if use_si and "acceleration_si" in risk_row.index
        else f"{float(risk_row['acceleration']):.1f} {acc_unit}"
    )
    offset = 30 * m_per_px  # offset in data coords (px or m)
    ax.annotate(
        f"max crack risk point\naccel={accel_label}\nP(crack)={crack_prob:.3f}{label_suffix}",
        xy=(ann_x, ann_y),
        xytext=(ann_x + offset, ann_y + offset),
        fontsize=8,
        color="#8b0000",
        arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1),
    )
    ax.plot(ann_x, ann_y, "o", markersize=4, color="#8b0000", alpha=0.9)

    # --- Units and title ---
    ax.set_xlabel("X (m)" if use_si else "X (px)", fontsize=12)
    ax.set_ylabel("Y (m)" if use_si else "Y (px)", fontsize=12)
    n_frames = int(df["frame"].nunique())
    n_points = int(df["index"].nunique())
    scale_note = f" | scale={meters_per_pixel:.2e} m/px" if use_si and meters_per_pixel else ""
    ax.set_title(
        f"Vector Map: points & arrows by index — one color per index\n"
        f"velocity ({vel_unit}) & acceleration ({acc_unit}){scale_note} | "
        f"{n_frames} frames, {n_points} points | FPS={fps:.1f}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    # Legend: points by index, velocity (same color as index), acceleration (complement)
    from matplotlib.lines import Line2D

    sample_color = index_colors[unique_indices[0]]
    sample_complement = index_complement_colors[unique_indices[0]]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=sample_color,
            markeredgecolor="white",
            markersize=7,
            label=f"points (x,y) by index [{n_frames} frames, {n_points} points]",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=sample_color,
            markersize=8,
            label=f"velocity ({vel_unit}) — index color",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=sample_complement,
            markersize=8,
            label=f"acceleration ({acc_unit}) — complement",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=dpi, bbox_inches="tight")
    plt.close()
    return output_image


def create_full_vector_map_figure(
    vectors_csv: Path,
    fps: float,
    *,
    meters_per_pixel: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 100,
    point_size: float = 2.5,
    point_alpha: float = 0.92,
    point_edge_width: float = 0.2,
    velocity_scale: float | None = None,
    velocity_linewidth: float = 0.5,
    acceleration_scale: float = 0.018,
    acceleration_linewidth: float = 0.22,
    velocity_alpha: float = 0.7,
    acceleration_alpha: float = 0.28,
) -> Figure:
    """
    Create a matplotlib Figure with the same visualization as plot_full_vector_map.
    For GUI embedding: index-colored points, velocity & acceleration arrows, crack risk.
    Caller must NOT close the figure; GUI owns lifecycle.
    """
    from matplotlib.figure import Figure

    df = pd.read_csv(vectors_csv)
    for col in ("x", "y", "index", "vx", "vy", "ax", "ay", "acceleration", "curvature_like"):
        if col not in df.columns:
            raise ValueError(f"vectors.csv must contain '{col}'")

    use_si = (
        meters_per_pixel is not None
        and meters_per_pixel > 0
        and "acceleration_si" in df.columns
    )
    vel_unit = "m/s" if use_si else "px/s"
    acc_unit = "km/s²" if use_si else "px/s²"

    dt_s = 1.0 / fps
    if velocity_scale is None:
        velocity_scale = dt_s
    acc_scale_draw = acceleration_scale / 1000.0 if use_si else acceleration_scale

    m_per_px = float(meters_per_pixel) if use_si else 1.0
    x = df["x"].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    if use_si:
        x = x * m_per_px
        y = y * m_per_px
        velocity_scale = velocity_scale * m_per_px
        acc_scale_draw = acc_scale_draw * m_per_px

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    # --- 1) Points colored by index ---
    unique_indices = np.sort(df["index"].unique())
    index_colors = _index_color_map(unique_indices)
    index_complement_colors = {i: _complement_rgba(index_colors[i]) for i in unique_indices}
    colors = np.array([index_colors[i] for i in df["index"]])
    ax.scatter(
        x, y,
        s=point_size, c=colors, alpha=point_alpha,
        edgecolors="white", linewidths=point_edge_width,
    )

    vx = df["vx"].to_numpy(np.float64)
    vy = df["vy"].to_numpy(np.float64)
    ax_arr = df["ax"].to_numpy(np.float64)
    ay_arr = df["ay"].to_numpy(np.float64)

    # --- 2) Acceleration (bottom layer) ---
    tax = x + ax_arr * acc_scale_draw
    tay = y + ay_arr * acc_scale_draw
    a_angle = np.arctan2(ay_arr, ax_arr)
    a_body_len = np.hypot(ax_arr * acc_scale_draw, ay_arr * acc_scale_draw)
    a_head_len = np.where(a_body_len > 1e-6, a_body_len * 0.22, 0.0)
    al_a, ar_a = a_angle + np.pi - 0.38, a_angle + np.pi + 0.38
    lax = tax + a_head_len * np.cos(al_a)
    lay = tay + a_head_len * np.sin(al_a)
    rax = tax + a_head_len * np.cos(ar_a)
    ray = tay + a_head_len * np.sin(ar_a)
    acc_segments = np.array([
        np.stack([np.stack([x, y], axis=1), np.stack([tax, tay], axis=1)], axis=0),
        np.stack([np.stack([tax, tay], axis=1), np.stack([lax, lay], axis=1)], axis=0),
        np.stack([np.stack([tax, tay], axis=1), np.stack([rax, ray], axis=1)], axis=0),
    ])
    acc_segments = np.transpose(acc_segments, (0, 2, 1, 3)).reshape(-1, 2, 2)
    acc_colors = np.array([index_complement_colors[i] for i in df["index"] for _ in range(3)])
    lc_acc = LineCollection(
        acc_segments,
        colors=acc_colors,
        linewidths=acceleration_linewidth,
        alpha=acceleration_alpha,
        capstyle="round",
    )
    ax.add_collection(lc_acc)

    # --- 3) Velocity on top ---
    tx = x + vx * velocity_scale
    ty = y + vy * velocity_scale
    angle = np.arctan2(vy, vx)
    body_len = np.hypot(vx * velocity_scale, vy * velocity_scale)
    head_len = np.where(body_len > 1e-6, body_len * 0.22, 0.0)
    al, ar = angle + np.pi - 0.38, angle + np.pi + 0.38
    lx = tx + head_len * np.cos(al)
    ly = ty + head_len * np.sin(al)
    rx = tx + head_len * np.cos(ar)
    ry = ty + head_len * np.sin(ar)
    vel_segments = np.array([
        np.stack([np.stack([x, y], axis=1), np.stack([tx, ty], axis=1)], axis=0),
        np.stack([np.stack([tx, ty], axis=1), np.stack([lx, ly], axis=1)], axis=0),
        np.stack([np.stack([tx, ty], axis=1), np.stack([rx, ry], axis=1)], axis=0),
    ])
    vel_segments = np.transpose(vel_segments, (0, 2, 1, 3)).reshape(-1, 2, 2)
    vel_colors = np.array([index_colors[i] for i in df["index"] for _ in range(3)])
    lc_vel = LineCollection(
        vel_segments,
        colors=vel_colors,
        linewidths=velocity_linewidth,
        alpha=velocity_alpha,
        capstyle="round",
    )
    ax.add_collection(lc_vel)

    # --- Crack risk ---
    if "crack_risk" in df.columns:
        risk_row = df.loc[df["crack_risk"].idxmax()]
        ann_x = float(risk_row["x"]) * m_per_px
        ann_y = float(risk_row["y"]) * m_per_px
        crack_prob = float(risk_row["crack_risk"])
        label_suffix = " (physics)"
    else:
        risk_row = df.loc[df["acceleration"].idxmax()]
        ann_x = float(risk_row["x"]) * m_per_px
        ann_y = float(risk_row["y"]) * m_per_px
        crack_prob = _crack_probability_heuristic(
            float(risk_row["acceleration"]),
            float(df["acceleration"].max()),
            float(risk_row["curvature_like"]),
            float(df["curvature_like"].max()),
        )
        label_suffix = ""

    accel_label = (
        f"{float(risk_row['acceleration_si']) / 1000.0:.6f} {acc_unit}"
        if use_si and "acceleration_si" in risk_row.index
        else f"{float(risk_row['acceleration']):.1f} {acc_unit}"
    )
    offset = 30 * m_per_px
    ax.annotate(
        f"max crack risk point\naccel={accel_label}\nP(crack)={crack_prob:.3f}{label_suffix}",
        xy=(ann_x, ann_y),
        xytext=(ann_x + offset, ann_y + offset),
        fontsize=8,
        color="#8b0000",
        arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1),
    )
    ax.plot(ann_x, ann_y, "o", markersize=4, color="#8b0000", alpha=0.9)

    # --- Labels and legend ---
    ax.set_xlabel("X (m)" if use_si else "X (px)", fontsize=10)
    ax.set_ylabel("Y (m)" if use_si else "Y (px)", fontsize=10)
    n_frames = int(df["frame"].nunique())
    n_points = int(df["index"].nunique())
    scale_note = f" | scale={meters_per_pixel:.2e} m/px" if use_si and meters_per_pixel else ""
    ax.set_title(
        f"Vector Map: points & arrows by index — velocity ({vel_unit}) & acceleration ({acc_unit}){scale_note} | "
        f"{n_frames} frames, {n_points} points | FPS={fps:.1f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    from matplotlib.lines import Line2D
    sample_color = index_colors[unique_indices[0]]
    sample_complement = index_complement_colors[unique_indices[0]]
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sample_color, markeredgecolor="white",
               markersize=6, label=f"points by index [{n_frames} frames, {n_points} points]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sample_color, markersize=7,
               label=f"velocity ({vel_unit})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sample_complement, markersize=7,
               label=f"acceleration ({acc_unit})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def create_vector_map_figure(
    vectors_csv: Path,
    fps: float,
    *,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 100,
    point_size: float = 0.2,
    point_alpha: float = 0.4,
    velocity_scale: float = 0.08,
    velocity_width: float = 0.001,
    acceleration_scale: float = 0.05,
    acceleration_width: float = 0.0015,
    velocity_color: str = "#1f77b4",
    acceleration_color: str = "#d62728",
    velocity_alpha: float = 0.5,
    acceleration_alpha: float = 0.6,
) -> Figure:
    """
    Create a matplotlib Figure for embedding in GUI with zoom/pan toolbar.
    Caller must NOT close the figure; GUI owns lifecycle.
    """
    from matplotlib.figure import Figure

    df = pd.read_csv(vectors_csv)
    for col in ("x", "y", "vx", "vy", "ax", "ay", "acceleration", "curvature_like"):
        if col not in df.columns:
            raise ValueError(f"vectors.csv must contain '{col}'")

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    dt_s = 1.0 / fps

    ax.scatter(
        df["x"], df["y"],
        s=point_size, c="#333333", alpha=point_alpha, edgecolors="none",
    )
    ax.quiver(
        df["x"], df["y"],
        df["vx"] * velocity_scale, df["vy"] * velocity_scale,
        color=velocity_color, alpha=velocity_alpha,
        width=velocity_width, scale=None, angles="xy", scale_units="xy",
    )
    ax.quiver(
        df["x"], df["y"],
        df["ax"] * acceleration_scale, df["ay"] * acceleration_scale,
        color=acceleration_color, alpha=acceleration_alpha,
        width=acceleration_width, scale=None, angles="xy", scale_units="xy",
    )

    max_acc_row = df.loc[df["acceleration"].idxmax()]
    max_x = float(max_acc_row["x"])
    max_y = float(max_acc_row["y"])
    max_acc = float(max_acc_row["acceleration"])
    max_curv = float(df["curvature_like"].max())
    crack_prob = _crack_probability_heuristic(
        max_acc,
        float(df["acceleration"].max()),
        float(max_acc_row["curvature_like"]),
        max_curv,
    )
    ax.annotate(
        f"max impact\naccel={max_acc:.1f} px/s²\nP(crack)≈{crack_prob:.2f}",
        xy=(max_x, max_y),
        xytext=(max_x + 30, max_y + 30),
        fontsize=8,
        color="#8b0000",
        arrowprops=dict(arrowstyle="->", color="#8b0000", lw=1),
    )
    ax.plot(max_x, max_y, "o", markersize=4, color="#8b0000", alpha=0.9)

    ax.set_xlabel("X (px)", fontsize=10)
    ax.set_ylabel("Y (px)", fontsize=10)
    ax.set_title(
        f"Vector Map: velocity (px/s) & acceleration (px/s²) | dt={dt_s:.4f} s",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=velocity_color, markersize=8, label="velocity (px/s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=acceleration_color, markersize=8, label="acceleration (px/s²)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_vector_map(
    vectors_csv: Path,
    output_image: Path,
    color_by: str = "index",
    scale: float = 1.0,
    width: float = 0.002,
    alpha: float = 0.6,
    point_size: float = 0.5,
) -> None:
    """
    Legacy: Plot velocity vectors only (colored by index/frame).
    Prefer plot_full_vector_map for ver7.
    """
    df = pd.read_csv(vectors_csv)

    if "vx" not in df.columns or "vy" not in df.columns:
        raise ValueError("vectors.csv must contain vx and vy columns")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_aspect("equal", adjustable="box")

    if color_by == "index":
        unique_vals = sorted(df["index"].unique())
        color_map = plt.cm.tab20  # type: ignore[attr-defined]
        df = df.copy()
        df["color_val"] = df["index"].map({val: i for i, val in enumerate(unique_vals)})
    elif color_by == "frame":
        unique_vals = sorted(df["frame"].unique())
        color_map = plt.cm.viridis  # type: ignore[attr-defined]
        df = df.copy()
        df["color_val"] = df["frame"].map({val: i for i, val in enumerate(unique_vals)})
    else:
        raise ValueError(f"color_by must be 'index' or 'frame', got {color_by}")

    df["color_norm"] = df["color_val"] / max(df["color_val"].max(), 1)

    ax.scatter(
        df["x"],
        df["y"],
        c=df["color_norm"],
        cmap=color_map,
        s=point_size,
        alpha=alpha * 0.5,
        edgecolors="none",
    )

    vx_scaled = df["vx"] * scale
    vy_scaled = df["vy"] * scale

    ax.quiver(
        df["x"],
        df["y"],
        vx_scaled,
        vy_scaled,
        df["color_norm"],
        cmap=color_map,
        scale=None,
        width=width,
        alpha=alpha,
        angles="xy",
        scale_units="xy",
    )

    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.set_title(
        f"Velocity Vector Map (colored by {color_by})\n"
        f"Total frames: {df['frame'].nunique()}, Total points: {df['index'].nunique()}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    sm = plt.cm.ScalarMappable(
        cmap=color_map, norm=plt.Normalize(vmin=0, vmax=len(unique_vals) - 1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{color_by.capitalize()}", fontsize=11)

    plt.tight_layout()
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close()


def plot_vector_map_by_index(
    vectors_csv: Path, output_image: Path, **kwargs: Any
) -> None:
    """Convenience: plot velocity vectors colored by index."""
    plot_vector_map(vectors_csv, output_image, color_by="index", **kwargs)


def plot_vector_map_by_frame(
    vectors_csv: Path, output_image: Path, **kwargs: Any
) -> None:
    """Convenience: plot velocity vectors colored by frame."""
    plot_vector_map(vectors_csv, output_image, color_by="frame", **kwargs)


def plot_frame_metrics(
    frame_metrics_csv: Path,
    output_image: Path,
    *,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 150,
) -> Path:
    """
    Plot FPCB bending metrics from frame_metrics.csv: bend angle, curvature concentration, est_max_strain.

    Physical units: bend_angle_deg (deg), curvature_concentration (dimensionless), est_max_strain (surrogate).
    """
    df = pd.read_csv(frame_metrics_csv)
    for col in ("frame", "time_s", "bend_angle_deg", "curvature_concentration", "est_max_strain"):
        if col not in df.columns:
            raise ValueError(f"frame_metrics.csv must contain '{col}'")

    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    t = df["time_s"].to_numpy()

    axes[0].plot(t, df["bend_angle_deg"], color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Bend angle (°)", fontsize=10)
    axes[0].set_title("FPCB bending physics: angle, curvature concentration, strain surrogate")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(t.min(), t.max())

    axes[1].plot(t, df["curvature_concentration"], color="#2ca02c", linewidth=1.5)
    axes[1].set_ylabel("Curvature concentration", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, df["est_max_strain"], color="#d62728", linewidth=1.5)
    axes[2].set_ylabel("Est. max strain (surrogate)", fontsize=10)
    axes[2].set_xlabel("Time (s)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=dpi, bbox_inches="tight")
    plt.close()
    return output_image
