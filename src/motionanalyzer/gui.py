from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
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


def _render_summary(summary_data: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames", int(summary_data.get("frame_count", 0)))
    c2.metric("Mean Speed", f'{float(summary_data.get("mean_speed", 0.0)):.3f}')
    c3.metric("Mean Accel", f'{float(summary_data.get("mean_acceleration", 0.0)):.3f}')

    unique_count = summary_data.get("unique_index_count")
    if unique_count is not None:
        st.caption(f"Tracked indices: {int(unique_count)}")


def _plot_vectors(vectors_path: Path) -> None:
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
    _render_summary(summary.__dict__)

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

    tab_analyze, tab_compare = st.tabs(["Analyze", "Compare"])
    with tab_analyze:
        _render_analyze_tab()

    with tab_compare:
        _render_compare_tab()


if __name__ == "__main__":
    main()
