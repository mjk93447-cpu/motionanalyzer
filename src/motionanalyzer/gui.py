from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import to_hex
import streamlit as st

from motionanalyzer.analysis import compare_summaries, load_summary, run_analysis

DEFAULT_INPUT_DIR = "data/synthetic/normal_case"
DEFAULT_OUTPUT_DIR = "exports/vectors/normal_case"
DEFAULT_BASE_SUMMARY = "exports/vectors/normal_case/summary.json"
DEFAULT_CANDIDATE_SUMMARY = "exports/vectors/crack_case/summary.json"


def _apply_full_hd_layout() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1920px;
            min-height: 1080px;
            padding-top: 1.25rem;
            padding-right: 1.5rem;
            padding-left: 1.5rem;
            padding-bottom: 1rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_summary(summary_data: dict[str, Any], pixel_per_meter: float = 1.0) -> None:
    """Render summary metrics with optional SI unit conversion."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", int(summary_data.get("frame_count", 0)))
    
    # Convert to SI units if pixel_per_meter is provided
    if pixel_per_meter > 0:
        mean_speed_ms = float(summary_data.get("mean_speed", 0.0)) * pixel_per_meter
        mean_accel_ms2 = float(summary_data.get("mean_acceleration", 0.0)) * pixel_per_meter
        c2.metric("Mean Speed", f'{mean_speed_ms:.4f} m/s')
        c3.metric("Mean Accel", f'{mean_accel_ms2:.4f} m/s²')
    else:
        c2.metric("Mean Speed", f'{float(summary_data.get("mean_speed", 0.0)):.3f} px/s')
        c3.metric("Mean Accel", f'{float(summary_data.get("mean_acceleration", 0.0)):.3f} px/s²')
    
    unique_count = summary_data.get("unique_index_count")
    if unique_count is not None:
        c4.metric("Unique Points", int(unique_count))
        st.caption(f"Tracked indices: {int(unique_count)}")


def _generate_color_map(unique_indices: np.ndarray) -> dict[int, str]:
    """Generate unique colors for each index using a colormap."""
    n = len(unique_indices)
    # Use a colormap that provides good color distinction
    cmap = cm.get_cmap('tab20', max(20, n))
    color_map = {}
    for i, idx in enumerate(unique_indices):
        # Cycle through colormap if more indices than colors
        color = cmap(i % max(20, n))
        color_map[int(idx)] = to_hex(color)
    return color_map


def _plot_vectors_enhanced(
    vectors_path: Path,
    pixel_per_meter: float = 1.0,
    frame_selection: int | None = None,
    arrow_scale: float = 0.1,
    max_points: int = 5000,
) -> None:
    """Enhanced visualization with points, arrows, and SI unit conversion."""
    df = pd.read_csv(vectors_path)
    
    # Filter by frame if specified
    if frame_selection is not None:
        df = df[df["frame"] == frame_selection].copy()
    
    # Performance optimization: sample data if too many points
    if len(df) > max_points:
        # Sample evenly across indices
        unique_indices = df["index"].unique()
        sample_size = max_points // len(unique_indices) if len(unique_indices) > 0 else max_points
        df_sampled = df.groupby("index", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size))
        ).reset_index(drop=True)
        df = df_sampled
    
    # Convert to SI units
    if pixel_per_meter > 0:
        df["vx_si"] = df["vx"] * pixel_per_meter
        df["vy_si"] = df["vy"] * pixel_per_meter
        df["ax_si"] = df["ax"] * pixel_per_meter
        df["ay_si"] = df["ay"] * pixel_per_meter
        df["speed_si"] = df["speed"] * pixel_per_meter
        df["acceleration_si"] = df["acceleration"] * pixel_per_meter
        vx_col, vy_col = "vx_si", "vy_si"
        ax_col, ay_col = "ax_si", "ay_si"
        speed_col = "speed_si"
        accel_col = "acceleration_si"
        speed_unit = "m/s"
        accel_unit = "m/s²"
    else:
        vx_col, vy_col = "vx", "vy"
        ax_col, ay_col = "ax", "ay"
        speed_col = "speed"
        accel_col = "acceleration"
        speed_unit = "px/s"
        accel_unit = "px/s²"
    
    # Get unique indices for color mapping
    unique_indices = sorted(df["index"].unique())
    color_map = _generate_color_map(np.array(unique_indices))
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Plot points with index-based colors
    for idx in unique_indices:
        idx_df = df[df["index"] == idx]
        fig.add_trace(go.Scatter(
            x=idx_df["x"],
            y=-idx_df["y"],  # Invert y-axis
            mode='markers',
            marker=dict(
                size=4,
                color=color_map[idx],
                line=dict(width=0.5, color='black')
            ),
            name=f'Point {idx}',
            showlegend=True,
            hovertemplate=f'Index: {idx}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
        ))
    
    # Add velocity arrows
    arrow_length_scale = arrow_scale
    for idx in unique_indices[:min(100, len(unique_indices))]:  # Limit arrows for performance
        idx_df = df[df["index"] == idx].head(50)  # Further limit per index
        for _, row in idx_df.iterrows():
            if abs(row[vx_col]) > 1e-6 or abs(row[vy_col]) > 1e-6:
                x0, y0 = row["x"], -row["y"]  # Invert y
                dx = row[vx_col] * arrow_length_scale
                dy = -row[vy_col] * arrow_length_scale  # Invert y component
                
                # Create arrow annotation
                fig.add_annotation(
                    x=x0 + dx,
                    y=y0 + dy,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,  # Larger arrowhead for clarity
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=color_map[idx],
                    opacity=0.7,
                )
    
    # Update layout
    fig.update_layout(
        title=f'Point Coordinates and Velocity Vectors (Frame: {frame_selection if frame_selection is not None else "All"})',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)',
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        height=700,
        hovermode='closest',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot acceleration arrows separately
    fig_acc = go.Figure()
    
    for idx in unique_indices:
        idx_df = df[df["index"] == idx]
        fig_acc.add_trace(go.Scatter(
            x=idx_df["x"],
            y=-idx_df["y"],  # Invert y-axis
            mode='markers',
            marker=dict(
                size=4,
                color=color_map[idx],
                line=dict(width=0.5, color='black')
            ),
            name=f'Point {idx}',
            showlegend=True,
            hovertemplate=f'Index: {idx}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
        ))
    
    # Add acceleration arrows
    for idx in unique_indices[:min(100, len(unique_indices))]:  # Limit arrows for performance
        idx_df = df[df["index"] == idx].head(50)  # Further limit per index
        for _, row in idx_df.iterrows():
            if abs(row[ax_col]) > 1e-6 or abs(row[ay_col]) > 1e-6:
                x0, y0 = row["x"], -row["y"]  # Invert y
                dx = row[ax_col] * arrow_length_scale
                dy = -row[ay_col] * arrow_length_scale  # Invert y component
                
                # Create arrow annotation
                fig_acc.add_annotation(
                    x=x0 + dx,
                    y=y0 + dy,
                    ax=x0,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,  # Larger arrowhead for clarity
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor=color_map[idx],
                    opacity=0.7,
                )
    
    fig_acc.update_layout(
        title=f'Point Coordinates and Acceleration Vectors (Frame: {frame_selection if frame_selection is not None else "All"})',
        xaxis_title='X (pixels)',
        yaxis_title='Y (pixels)',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=700,
        hovermode='closest',
    )
    
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Time series plots with SI units
    speed_df = df.groupby("frame", as_index=False)[speed_col].mean()
    acc_df = df.groupby("frame", as_index=False)[accel_col].mean()
    
    fig_speed = px.line(
        speed_df,
        x="frame",
        y=speed_col,
        title=f"Mean Speed by Frame ({speed_unit})"
    )
    fig_acc = px.line(
        acc_df,
        x="frame",
        y=accel_col,
        title=f"Mean Acceleration by Frame ({accel_unit})"
    )
    st.plotly_chart(fig_speed, use_container_width=True)
    st.plotly_chart(fig_acc, use_container_width=True)


def _plot_vectors(vectors_path: Path) -> None:
    """Legacy function for backward compatibility."""
    df = pd.read_csv(vectors_path)
    speed_df = df.groupby("frame", as_index=False)["speed"].mean()
    acc_df = df.groupby("frame", as_index=False)["acceleration"].mean()

    fig_speed = px.line(speed_df, x="frame", y="speed", title="Mean Speed by Frame")
    fig_acc = px.line(acc_df, x="frame", y="acceleration", title="Mean Acceleration by Frame")
    st.plotly_chart(fig_speed, use_container_width=True)
    st.plotly_chart(fig_acc, use_container_width=True)


def _read_json_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render_analyze_tab() -> None:
    st.subheader("Run analysis from frame bundle")
    input_col, output_col = st.columns(2)

    with input_col:
        input_dir = st.text_input("Input bundle path", DEFAULT_INPUT_DIR, key="input_dir")
    with output_col:
        output_dir = st.text_input("Output analysis path", DEFAULT_OUTPUT_DIR, key="output_dir")

    run_col, info_col = st.columns([1, 2])
    with run_col:
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)
    with info_col:
        st.caption("Target UI profile: 1920x1080 Full HD (offline local execution).")

    if not run_clicked:
        return

    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()

    if not input_path.exists():
        st.error(f"Input path not found: {input_path}")
        return

    summary = run_analysis(input_path, output_path)
    st.success("Analysis complete")
    pixel_per_meter = st.session_state.get("pixel_per_meter", 1.0)
    _render_summary(summary.__dict__, pixel_per_meter=pixel_per_meter)

    summary_json = output_path / "summary.json"
    vectors_csv = output_path / "vectors.csv"
    if summary_json.exists() and vectors_csv.exists():
        _plot_vectors(vectors_csv)

        preview_df = pd.read_csv(vectors_csv).head(20)
        st.dataframe(preview_df, use_container_width=True, height=340)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                label="Download summary.json",
                data=_read_json_text(summary_json),
                file_name="summary.json",
                mime="application/json",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                label="Download vectors.csv",
                data=vectors_csv.read_bytes(),
                file_name="vectors.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.warning(f"Expected result files were not found under: {output_path}")


def _render_visualize_tab() -> None:
    st.subheader("Enhanced Visualization")
    
    # Input for vectors CSV path
    vectors_path = st.text_input(
        "Vectors CSV path", "exports/vectors/normal_case/vectors.csv", key="vectors_path"
    )
    
    # Pixel per meter input for SI unit conversion
    col1, col2 = st.columns(2)
    with col1:
        pixel_per_meter = st.number_input(
            "Pixel per meter (m/pixel)",
            min_value=0.0,
            value=1.0,
            step=0.001,
            format="%.6f",
            help="Enter the conversion factor: meters per pixel. For example, if 100 pixels = 1 meter, enter 0.01",
            key="pixel_per_meter_input"
        )
        st.session_state["pixel_per_meter"] = pixel_per_meter
    
    with col2:
        arrow_scale = st.slider(
            "Arrow scale factor",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Scale factor for arrow length. Smaller values = shorter arrows",
            key="arrow_scale"
        )
    
    # Frame selection
    frame_selection = st.number_input(
        "Frame number (leave empty for all frames)",
        min_value=0,
        value=None,
        step=1,
        help="Select a specific frame to visualize, or leave empty to show all frames",
        key="frame_selection"
    )
    
    # Performance settings
    with st.expander("Performance Settings"):
        max_points = st.number_input(
            "Maximum points to display",
            min_value=100,
            max_value=50000,
            value=5000,
            step=500,
            help="Reduce this value if visualization is slow",
            key="max_points"
        )
    
    if st.button("Visualize", type="primary", use_container_width=True):
        vectors_file = Path(vectors_path).expanduser()
        if not vectors_file.exists():
            st.error(f"Vectors file not found: {vectors_path}")
        else:
            frame_val = int(frame_selection) if frame_selection is not None else None
            _plot_vectors_enhanced(
                vectors_file,
                pixel_per_meter=pixel_per_meter,
                frame_selection=frame_val,
                arrow_scale=arrow_scale,
                max_points=max_points,
            )


def _render_compare_tab() -> None:
    st.subheader("Compare two analysis results")
    base_col, cand_col = st.columns(2)
    with base_col:
        base_summary_path = st.text_input(
            "Base summary.json", DEFAULT_BASE_SUMMARY, key="base_summary"
        )
    with cand_col:
        cand_summary_path = st.text_input(
            "Candidate summary.json",
            DEFAULT_CANDIDATE_SUMMARY,
            key="cand_summary",
        )

    if not st.button("Compare Results", use_container_width=True):
        return

    base_path = Path(base_summary_path).expanduser()
    cand_path = Path(cand_summary_path).expanduser()
    missing_paths = [str(p) for p in (base_path, cand_path) if not p.exists()]
    if missing_paths:
        st.error("Missing file(s): " + ", ".join(missing_paths))
        return

    base = load_summary(base_path)
    cand = load_summary(cand_path)
    delta = compare_summaries(base, cand)
    st.write("Delta (candidate - base)")
    st.json(delta)
    delta_df = pd.DataFrame([delta]).T.reset_index()
    delta_df.columns = ["metric", "delta"]
    st.dataframe(delta_df, use_container_width=True, height=260)


def main() -> None:
    st.set_page_config(
        page_title="motionanalyzer GUI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_full_hd_layout()
    st.title("motionanalyzer - FPCB bending analysis")
    st.caption("Offline-capable local GUI for vector analysis and summary comparison.")

    with st.sidebar:
        st.header("Display Profile")
        st.write("Resolution target: 1920x1080")
        st.write("Mode: local offline")
        st.divider()
        st.write("Quick paths")
        st.code(DEFAULT_INPUT_DIR)
        st.code(DEFAULT_OUTPUT_DIR)

    tab_analyze, tab_visualize, tab_compare = st.tabs(["Analyze", "Visualize", "Compare"])
    with tab_analyze:
        _render_analyze_tab()

    with tab_visualize:
        _render_visualize_tab()

    with tab_compare:
        _render_compare_tab()


if __name__ == "__main__":
    main()
