"""Vector map visualization with velocity, acceleration, and impact analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
    figsize: tuple[float, float] = (24, 18),
    dpi: int = 200,
    point_size: float = 0.15,
    point_alpha: float = 0.35,
    velocity_scale: float = 0.08,
    velocity_width: float = 0.0008,
    acceleration_scale: float = 0.05,
    acceleration_width: float = 0.0012,
    velocity_color: str = "#1f77b4",
    acceleration_color: str = "#d62728",
    velocity_alpha: float = 0.5,
    acceleration_alpha: float = 0.6,
) -> Path:
    """
    Plot velocity and acceleration vectors on a 2D graph with units and impact analysis.

    - Points: original coordinates (x, y)
    - Velocity arrows: vx, vy (px/s)
    - Acceleration arrows: ax, ay (px/s²), clearly distinguishable
    - Max impact region annotated
    - Crack probability heuristic displayed

    Args:
        vectors_csv: Path to vectors.csv from run_analysis
        output_image: Path to save PNG
        fps: Frames per second (for unit labels)
    """
    df = pd.read_csv(vectors_csv)

    for col in ("x", "y", "vx", "vy", "ax", "ay", "acceleration", "curvature_like"):
        if col not in df.columns:
            raise ValueError(f"vectors.csv must contain '{col}'")

    dt_s = 1.0 / fps

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")

    # --- Points (original coordinates) ---
    ax.scatter(
        df["x"],
        df["y"],
        s=point_size,
        c="#333333",
        alpha=point_alpha,
        edgecolors="none",
    )

    # --- Velocity vectors (px/s) ---
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()
    ax.quiver(
        df["x"],
        df["y"],
        vx * velocity_scale,
        vy * velocity_scale,
        color=velocity_color,
        alpha=velocity_alpha,
        width=velocity_width,
        scale=None,
        angles="xy",
        scale_units="xy",
    )

    # --- Acceleration vectors (px/s²) - dashed style via thinner/offset for distinction ---
    ax.quiver(
        df["x"],
        df["y"],
        df["ax"] * acceleration_scale,
        df["ay"] * acceleration_scale,
        color=acceleration_color,
        alpha=acceleration_alpha,
        width=acceleration_width,
        scale=None,
        angles="xy",
        scale_units="xy",
    )

    # --- Max impact (highest acceleration point) ---
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

    # --- Units and title ---
    ax.set_xlabel("X (px)", fontsize=12)
    ax.set_ylabel("Y (px)", fontsize=12)
    ax.set_title(
        f"Vector Map: velocity (px/s) & acceleration (px/s²)\n"
        f"dt={dt_s:.4f} s/frame, FPS={fps:.1f} | "
        f"Frames={df['frame'].nunique()}, Points={df['index'].nunique()}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=velocity_color,
            markersize=8,
            label="velocity (px/s)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=acceleration_color,
            markersize=8,
            label="acceleration (px/s²)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=dpi, bbox_inches="tight")
    plt.close()
    return output_image


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
