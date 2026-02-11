from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisSummary:
    fps: float
    frame_count: int
    point_count_per_frame_min: int
    point_count_per_frame_max: int
    unique_index_count: int
    mean_speed: float
    max_speed: float
    mean_acceleration: float
    max_acceleration: float
    mean_curvature_like: float
    p95_curvature_like: float
    max_curvature_like: float


def _read_single_frame(path: Path, frame_idx: int) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty frame file: {path}")
    if lines[0].strip() != "# x,y,index":
        raise ValueError(f"Invalid header in frame file: {path.name}")

    records: list[dict[str, int]] = []
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 3:
            raise ValueError(f"Malformed row {i} in {path.name}")
        try:
            x = int(cols[0])
            y = int(cols[1])
            idx = int(cols[2])
        except ValueError as exc:
            raise ValueError(f"Non-integer row {i} in {path.name}") from exc
        records.append({"frame": frame_idx, "index": idx, "x": x, "y": y})
    return pd.DataFrame.from_records(records)


def load_bundle(input_dir: Path) -> tuple[pd.DataFrame, float]:
    fps_file = input_dir / "fps.txt"
    if not fps_file.exists():
        raise FileNotFoundError(f"fps.txt not found in {input_dir}")
    fps = float(fps_file.read_text(encoding="utf-8").strip())
    if fps <= 0:
        raise ValueError("fps must be > 0")

    frame_files = sorted(input_dir.glob("frame_*.txt"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.txt found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for frame_file in frame_files:
        stem_digits = "".join(ch for ch in frame_file.stem if ch.isdigit())
        frame_idx = int(stem_digits) if stem_digits else len(frames)
        frames.append(_read_single_frame(frame_file, frame_idx=frame_idx))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["index", "frame"]).reset_index(drop=True)
    return df, fps


def compute_vectors(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    dt = 1.0 / fps
    work = df.copy()
    grouped = work.groupby("index", sort=False)

    work["dx"] = grouped["x"].diff().fillna(0.0)
    work["dy"] = grouped["y"].diff().fillna(0.0)
    work["vx"] = work["dx"] / dt
    work["vy"] = work["dy"] / dt
    work["speed"] = np.hypot(work["vx"], work["vy"])

    work["dvx"] = grouped["vx"].diff().fillna(0.0)
    work["dvy"] = grouped["vy"].diff().fillna(0.0)
    work["ax"] = work["dvx"] / dt
    work["ay"] = work["dvy"] / dt
    work["acceleration"] = np.hypot(work["ax"], work["ay"])

    # Curvature-like surrogate from directional velocity change.
    eps = 1e-6
    work["curvature_like"] = np.hypot(work["dvx"], work["dvy"]) / (work["speed"] + eps)
    return work


def summarize(vectors: pd.DataFrame, fps: float) -> AnalysisSummary:
    frame_counts = vectors.groupby("frame")["index"].count()
    return AnalysisSummary(
        fps=float(fps),
        frame_count=int(vectors["frame"].nunique()),
        point_count_per_frame_min=int(frame_counts.min()),
        point_count_per_frame_max=int(frame_counts.max()),
        unique_index_count=int(vectors["index"].nunique()),
        mean_speed=float(vectors["speed"].mean()),
        max_speed=float(vectors["speed"].max()),
        mean_acceleration=float(vectors["acceleration"].mean()),
        max_acceleration=float(vectors["acceleration"].max()),
        mean_curvature_like=float(vectors["curvature_like"].mean()),
        p95_curvature_like=float(vectors["curvature_like"].quantile(0.95)),
        max_curvature_like=float(vectors["curvature_like"].max()),
    )


def export_analysis(vectors: pd.DataFrame, summary: AnalysisSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "vectors.csv"
    vectors.to_csv(csv_path, index=False)

    # Standardized txt output for process integration.
    txt_path = output_dir / "vectors.txt"
    cols = [
        "frame",
        "index",
        "x",
        "y",
        "vx",
        "vy",
        "speed",
        "ax",
        "ay",
        "acceleration",
        "curvature_like",
    ]
    txt_lines = [",".join(cols)]
    for row in vectors[cols].itertuples(index=False):
        txt_lines.append(
            ",".join(f"{float(v):.6f}" if isinstance(v, float) else str(v) for v in row)
        )
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=True, indent=2), encoding="utf-8"
    )

    summary_txt_path = output_dir / "summary.txt"
    txt_lines = [
        "motionanalyzer summary",
        f"fps={summary.fps}",
        f"frame_count={summary.frame_count}",
        f"point_count_per_frame_min={summary.point_count_per_frame_min}",
        f"point_count_per_frame_max={summary.point_count_per_frame_max}",
        f"unique_index_count={summary.unique_index_count}",
        f"mean_speed={summary.mean_speed:.6f}",
        f"max_speed={summary.max_speed:.6f}",
        f"mean_acceleration={summary.mean_acceleration:.6f}",
        f"max_acceleration={summary.max_acceleration:.6f}",
        f"mean_curvature_like={summary.mean_curvature_like:.6f}",
        f"p95_curvature_like={summary.p95_curvature_like:.6f}",
        f"max_curvature_like={summary.max_curvature_like:.6f}",
    ]
    summary_txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


def run_analysis(input_dir: Path, output_dir: Path) -> AnalysisSummary:
    df, fps = load_bundle(input_dir)
    vectors = compute_vectors(df=df, fps=fps)
    summary = summarize(vectors=vectors, fps=fps)
    export_analysis(vectors=vectors, summary=summary, output_dir=output_dir)
    return summary


def compare_summaries(
    base_summary: AnalysisSummary, candidate_summary: AnalysisSummary
) -> dict[str, float]:
    return {
        "delta_mean_speed": candidate_summary.mean_speed - base_summary.mean_speed,
        "delta_max_speed": candidate_summary.max_speed - base_summary.max_speed,
        "delta_mean_acceleration": candidate_summary.mean_acceleration
        - base_summary.mean_acceleration,
        "delta_max_acceleration": candidate_summary.max_acceleration
        - base_summary.max_acceleration,
        "delta_mean_curvature_like": candidate_summary.mean_curvature_like
        - base_summary.mean_curvature_like,
        "delta_p95_curvature_like": candidate_summary.p95_curvature_like
        - base_summary.p95_curvature_like,
        "delta_max_curvature_like": candidate_summary.max_curvature_like
        - base_summary.max_curvature_like,
    }


def load_summary(summary_path: Path) -> AnalysisSummary:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return AnalysisSummary(**payload)
