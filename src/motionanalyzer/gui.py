from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from motionanalyzer.analysis import compare_summaries, load_summary, run_analysis


def _render_summary(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames", int(summary["frame_count"]))
    c2.metric("Mean Speed", f'{summary["mean_speed"]:.2f}')
    c3.metric("Mean Accel", f'{summary["mean_acceleration"]:.2f}')


def _plot_vectors(vectors_path: Path) -> None:
    df = pd.read_csv(vectors_path)
    speed_df = df.groupby("frame", as_index=False)["speed"].mean()
    acc_df = df.groupby("frame", as_index=False)["acceleration"].mean()

    fig_speed = px.line(speed_df, x="frame", y="speed", title="Mean Speed by Frame")
    fig_acc = px.line(acc_df, x="frame", y="acceleration", title="Mean Acceleration by Frame")
    st.plotly_chart(fig_speed, use_container_width=True)
    st.plotly_chart(fig_acc, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="motionanalyzer GUI", layout="wide")
    st.title("motionanalyzer - FPCB bending analysis")

    tab_analyze, tab_compare = st.tabs(["Analyze", "Compare"])

    with tab_analyze:
        st.subheader("Run analysis from frame bundle")
        input_dir = st.text_input("Input bundle path", "data/synthetic/normal_case")
        output_dir = st.text_input("Output analysis path", "exports/vectors/normal_case")
        if st.button("Run Analysis", type="primary"):
            summary = run_analysis(Path(input_dir), Path(output_dir))
            st.success("Analysis complete")
            st.json(summary.__dict__)
            _render_summary(Path(output_dir) / "summary.json")
            _plot_vectors(Path(output_dir) / "vectors.csv")

    with tab_compare:
        st.subheader("Compare two analysis results")
        base_summary_path = st.text_input(
            "Base summary.json", "exports/vectors/normal_case/summary.json"
        )
        cand_summary_path = st.text_input(
            "Candidate summary.json", "exports/vectors/crack_case/summary.json"
        )
        if st.button("Compare Results"):
            base = load_summary(Path(base_summary_path))
            cand = load_summary(Path(cand_summary_path))
            delta = compare_summaries(base, cand)
            st.write("Delta (candidate - base)")
            st.json(delta)


if __name__ == "__main__":
    main()
