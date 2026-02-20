from pathlib import Path

import pandas as pd

from motionanalyzer.analysis import (
    compare_summaries,
    compute_vectors,
    run_analysis,
    summarize,
)


def _write_frame(path: Path, values: list[tuple[int, int, int]]) -> None:
    lines = ["# x,y,index"]
    for x, y, idx in values:
        lines.append(f"{x},{y},{idx}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_bundle_and_run_analysis(tmp_path: Path) -> None:
    input_dir = tmp_path / "bundle"
    input_dir.mkdir()
    (input_dir / "fps.txt").write_text("10\n", encoding="utf-8")
    _write_frame(input_dir / "frame_00000.txt", [(0, 0, 1), (5, 0, 2)])
    _write_frame(input_dir / "frame_00001.txt", [(1, 0, 1), (7, 0, 2)])
    _write_frame(input_dir / "frame_00002.txt", [(3, 0, 1), (9, 0, 2)])

    output_dir = tmp_path / "out"
    summary = run_analysis(input_dir=input_dir, output_dir=output_dir)
    assert summary.frame_count == 3
    assert summary.unique_index_count == 2
    assert (output_dir / "vectors.csv").exists()
    assert (output_dir / "vectors.txt").exists()
    assert (output_dir / "summary.json").exists()


def test_compare_summaries_returns_expected_delta() -> None:
    df_a = pd.DataFrame(
        [
            {"frame": 0, "index": 1, "x": 0, "y": 0},
            {"frame": 1, "index": 1, "x": 1, "y": 0},
            {"frame": 2, "index": 1, "x": 2, "y": 0},
        ]
    )
    df_b = pd.DataFrame(
        [
            {"frame": 0, "index": 1, "x": 0, "y": 0},
            {"frame": 1, "index": 1, "x": 2, "y": 0},
            {"frame": 2, "index": 1, "x": 5, "y": 0},
        ]
    )
    a_summary = summarize(compute_vectors(df_a, fps=10), fps=10)
    b_summary = summarize(compute_vectors(df_b, fps=10), fps=10)
    delta = compare_summaries(a_summary, b_summary)
    assert delta["delta_mean_speed"] > 0
