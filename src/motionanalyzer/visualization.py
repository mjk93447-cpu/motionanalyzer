from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


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
    Plot all velocity vectors as arrows on a 2D graph.

    Args:
        vectors_csv: Path to vectors.csv from run_analysis output
        output_image: Path to save the output image (PNG)
        color_by: "index" or "frame" - how to color-code vectors
        scale: Arrow scale factor (smaller = shorter arrows)
        width: Arrow width
        alpha: Transparency (0-1)
        point_size: Size of starting point markers
    """
    df = pd.read_csv(vectors_csv)

    if "vx" not in df.columns or "vy" not in df.columns:
        raise ValueError("vectors.csv must contain vx and vy columns")

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_aspect("equal", adjustable="box")

    # Get unique values for color mapping
    if color_by == "index":
        unique_vals = sorted(df["index"].unique())
        color_map = plt.cm.tab20  # type: ignore[attr-defined]
        df["color_val"] = df["index"].map({val: i for i, val in enumerate(unique_vals)})
    elif color_by == "frame":
        unique_vals = sorted(df["frame"].unique())
        color_map = plt.cm.viridis  # type: ignore[attr-defined]
        df["color_val"] = df["frame"].map({val: i for i, val in enumerate(unique_vals)})
    else:
        raise ValueError(f"color_by must be 'index' or 'frame', got {color_by}")

    # Normalize color values to [0, 1]
    df["color_norm"] = df["color_val"] / max(df["color_val"].max(), 1)

    # Plot starting points (small dots)
    ax.scatter(
        df["x"],
        df["y"],
        c=df["color_norm"],
        cmap=color_map,
        s=point_size,
        alpha=alpha * 0.5,
        edgecolors="none",
    )

    # Plot velocity vectors as arrows
    # Use quiver for arrows: (x, y, u, v) where u,v are direction components
    # Scale down arrows to avoid overlap
    vx_scaled = df["vx"] * scale
    vy_scaled = df["vy"] * scale

    ax.quiver(
        df["x"],
        df["y"],
        vx_scaled,
        vy_scaled,
        df["color_norm"],
        cmap=color_map,
        scale=None,  # Auto-scale based on data
        width=width,
        alpha=alpha,
        angles="xy",
        scale_units="xy",
    )

    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.set_title(
        f"Velocity Vector Map (colored by {color_by})\n"
        f"Total frames: {df['frame'].nunique()}, "
        f"Total points: {df['index'].nunique()}",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=color_map, norm=plt.Normalize(vmin=0, vmax=len(unique_vals) - 1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{color_by.capitalize()}", fontsize=11)

    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close()


def plot_vector_map_by_index(vectors_csv: Path, output_image: Path, **kwargs: Any) -> None:
    """Convenience wrapper: plot vectors colored by index."""
    plot_vector_map(vectors_csv, output_image, color_by="index", **kwargs)


def plot_vector_map_by_frame(vectors_csv: Path, output_image: Path, **kwargs: Any) -> None:
    """Convenience wrapper: plot vectors colored by frame."""
    plot_vector_map(vectors_csv, output_image, color_by="frame", **kwargs)
