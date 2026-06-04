from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_meters_per_pixel(input_dir: Path) -> float | None:
    """Read SI scale (m/px) from bundle metadata if present."""
    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        scale = data.get("meters_per_pixel")
        if scale is not None:
            return float(scale)
        cfg = data.get("config") or {}
        scale = cfg.get("meters_per_pixel")
        if scale is not None:
            return float(scale)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


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
    meters_per_pixel: float | None = None
    mean_speed_m_s: float | None = None
    max_speed_m_s: float | None = None
    mean_acceleration_m_s2: float | None = None
    max_acceleration_m_s2: float | None = None
    mean_crack_risk: float | None = None
    max_crack_risk: float | None = None
    effective_fps: float | None = None
    median_dt_s: float | None = None
    timing_source: str | None = None


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


def _extract_frame_index(path: Path) -> int | None:
    """
    Extract frame index from filename.

    Preferred rule (for real data):
      - Find a number that starts with "000" and has 5~6 digits total,
        e.g. 00001, 000001, 000123.

    Fallback (for legacy/test data):
      - Use any digits from the stem (e.g. frame_00000.txt -> 0).
    """
    match = re.search(r"000\d{2,3}", path.name)
    if match is not None:
        return int(match.group(0))

    # Legacy fallback: digits from stem
    stem_digits = "".join(ch for ch in path.stem if ch.isdigit())
    if stem_digits:
        return int(stem_digits)
    return None


def load_bundle(input_dir: Path, fps: float | None = None) -> tuple[pd.DataFrame, float, float | None]:
    """
    Load bundle of frame_*.txt-like files from input_dir.

    fps:
        - When provided, use this value directly (no fps.txt required).
        - When None, fall back to legacy behavior that reads fps.txt.

    Returns:
        (df, fps_val, meters_per_pixel). meters_per_pixel is from metadata.json if present, else None.
    """
    if fps is None:
        fps_file = input_dir / "fps.txt"
        if not fps_file.exists():
            raise FileNotFoundError(f"fps.txt not found in {input_dir}")
        fps_val = float(fps_file.read_text(encoding="utf-8").strip())
    else:
        fps_val = fps
    if fps_val <= 0:
        raise ValueError("fps must be > 0")

    meters_per_pixel = _read_meters_per_pixel(input_dir)

    # Collect all frame files based on 6-digit index pattern (000xxx) in filename.
    candidates = [p for p in input_dir.glob("*.txt") if p.name.lower() != "fps.txt"]
    indexed_files: list[tuple[int, Path]] = []
    for path in candidates:
        idx = _extract_frame_index(path)
        if idx is not None:
            indexed_files.append((idx, path))

    if not indexed_files:
        raise FileNotFoundError(f"No frame files with 6-digit index (000xxx) found in {input_dir}")

    indexed_files.sort(key=lambda item: item[0])

    frames: list[pd.DataFrame] = []
    for frame_idx, frame_file in indexed_files:
        frames.append(_read_single_frame(frame_file, frame_idx=frame_idx))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["index", "frame"]).reset_index(drop=True)
    return df, fps_val, meters_per_pixel


def load_frame_timing(input_dir: Path) -> pd.DataFrame | None:
    """Load per-bundle-frame timestamps from frame_timing.json if present."""
    path = input_dir / "frame_timing.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frames = payload.get("frames") or []
    if not frames:
        return None
    records = [
        {"frame": int(f["bundle_order"]), "time_s": float(f["time_s"])}
        for f in frames
    ]
    return pd.DataFrame.from_records(records).sort_values("frame").reset_index(drop=True)


def _frame_dt_table(frame_times: pd.DataFrame) -> pd.DataFrame:
    ft = frame_times.sort_values("frame").copy()
    ft["dt_s"] = ft["time_s"].diff()
    positive = ft["dt_s"][ft["dt_s"] > 0]
    median_dt = float(positive.median()) if len(positive) else None
    if median_dt is None or median_dt <= 0:
        median_dt = 1.0 / 30.0
    ft["dt_s"] = ft["dt_s"].fillna(median_dt)
    ft.loc[ft["dt_s"] <= 0, "dt_s"] = median_dt
    return ft


def compute_vectors(
    df: pd.DataFrame,
    fps: float,
    meters_per_pixel: float | None = None,
    frame_times: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute velocity and acceleration from position time series.
    Raw columns are retained for compatibility; official display units are mm, mm/s, and mm/s^2 when meters_per_pixel is set.
    When frame_times is provided, uses per-frame dt from recorded extraction timestamps.
    """
    work = df.copy()
    grouped = work.groupby("index", sort=False)

    work["dx"] = grouped["x"].diff().fillna(0.0)
    work["dy"] = grouped["y"].diff().fillna(0.0)

    if frame_times is not None and not frame_times.empty:
        ft = _frame_dt_table(frame_times)
        work = work.merge(ft[["frame", "time_s", "dt_s"]], on="frame", how="left")
        work["dt_s"] = work["dt_s"].fillna(1.0 / fps)
        work["vx"] = work["dx"] / work["dt_s"]
        work["vy"] = work["dy"] / work["dt_s"]
    else:
        dt = 1.0 / fps
        work["dt_s"] = dt
        work["vx"] = work["dx"] / dt
        work["vy"] = work["dy"] / dt

    work["speed"] = np.hypot(work["vx"], work["vy"])

    grouped_v = work.groupby("index", sort=False)
    work["dvx"] = grouped_v["vx"].diff().fillna(0.0)
    work["dvy"] = grouped_v["vy"].diff().fillna(0.0)
    if frame_times is not None and not frame_times.empty:
        work["ax"] = work["dvx"] / work["dt_s"]
        work["ay"] = work["dvy"] / work["dt_s"]
    else:
        dt = 1.0 / fps
        work["ax"] = work["dvx"] / dt
        work["ay"] = work["dvy"] / dt
    work["acceleration"] = np.hypot(work["ax"], work["ay"])

    # Curvature-like surrogate from directional velocity change.
    eps = 1e-6
    work["curvature_like"] = np.hypot(work["dvx"], work["dvy"]) / (work["speed"] + eps)

    if meters_per_pixel is not None and meters_per_pixel > 0:
        scale = float(meters_per_pixel)
        work["vx_si"] = work["vx"] * scale
        work["vy_si"] = work["vy"] * scale
        work["speed_si"] = work["speed"] * scale
        work["ax_si"] = work["ax"] * scale
        work["ay_si"] = work["ay"] * scale
        work["acceleration_si"] = work["acceleration"] * scale
    return work


def summarize(
    vectors: pd.DataFrame,
    fps: float,
    meters_per_pixel: float | None = None,
    *,
    effective_fps: float | None = None,
    median_dt_s: float | None = None,
    timing_source: str | None = None,
) -> AnalysisSummary:
    frame_counts = vectors.groupby("frame")["index"].count()
    mean_speed_m_s = None
    max_speed_m_s = None
    mean_acceleration_m_s2 = None
    max_acceleration_m_s2 = None
    mean_crack_risk = None
    max_crack_risk = None
    if meters_per_pixel is not None and "speed_si" in vectors.columns:
        mean_speed_m_s = float(vectors["speed_si"].mean())
        max_speed_m_s = float(vectors["speed_si"].max())
    if meters_per_pixel is not None and "acceleration_si" in vectors.columns:
        mean_acceleration_m_s2 = float(vectors["acceleration_si"].mean())
        max_acceleration_m_s2 = float(vectors["acceleration_si"].max())
    if "crack_risk" in vectors.columns:
        mean_crack_risk = float(vectors["crack_risk"].mean())
        max_crack_risk = float(vectors["crack_risk"].max())
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
        meters_per_pixel=meters_per_pixel,
        mean_speed_m_s=mean_speed_m_s,
        max_speed_m_s=max_speed_m_s,
        mean_acceleration_m_s2=mean_acceleration_m_s2,
        max_acceleration_m_s2=max_acceleration_m_s2,
        mean_crack_risk=mean_crack_risk,
        max_crack_risk=max_crack_risk,
        effective_fps=effective_fps,
        median_dt_s=median_dt_s,
        timing_source=timing_source,
    )


def export_analysis(vectors: pd.DataFrame, summary: AnalysisSummary, output_dir: Path) -> None:
    from motionanalyzer.visualization import plot_full_vector_map

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "vectors.csv"
    vectors.to_csv(csv_path, index=False)

    # Vector map image (velocity m/s, acceleration km/s² when meters_per_pixel set)
    image_path = output_dir / "vector_map.png"
    plot_full_vector_map(
        csv_path, image_path, fps=summary.fps, meters_per_pixel=summary.meters_per_pixel
    )

    # Standardized txt output for process integration (include SI columns when present).
    txt_path = output_dir / "vectors.txt"
    base_cols = [
        "frame", "index", "x", "y",
        "vx", "vy", "speed", "ax", "ay", "acceleration", "curvature_like",
    ]
    optional = [
        c
        for c in (
            "time_s",
            "dt_s",
            "vx_si",
            "vy_si",
            "speed_si",
            "ax_si",
            "ay_si",
            "acceleration_si",
            "strain_surrogate",
            "stress_surrogate",
            "impact_surrogate",
            "crack_risk",
        )
        if c in vectors.columns
    ]
    cols = base_cols + optional
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

    # Summary txt keeps raw diagnostics and adds official metric values when calibrated.
    summary_txt_path = output_dir / "summary.txt"
    txt_lines = [
        "motionanalyzer summary (official display units: mm, mm/s, mm/s^2 when meters_per_pixel is set)",
        f"fps={summary.fps}",
        f"dt_s={summary.median_dt_s if summary.median_dt_s is not None else 1.0/summary.fps:.6f}",
        f"effective_fps={summary.effective_fps if summary.effective_fps is not None else ''}",
        f"timing_source={summary.timing_source if summary.timing_source else ''}",
        f"meters_per_pixel={summary.meters_per_pixel if summary.meters_per_pixel is not None else ''}",
        f"frame_count={summary.frame_count}",
        f"point_count_per_frame_min={summary.point_count_per_frame_min}",
        f"point_count_per_frame_max={summary.point_count_per_frame_max}",
        f"unique_index_count={summary.unique_index_count}",
        f"mean_speed_px_s={summary.mean_speed:.6f}",
        f"max_speed_px_s={summary.max_speed:.6f}",
        f"mean_acceleration_px_s2={summary.mean_acceleration:.6f}",
        f"max_acceleration_px_s2={summary.max_acceleration:.6f}",
    ]
    if summary.mean_speed_m_s is not None:
        txt_lines.append(f"mean_speed_m_s={summary.mean_speed_m_s:.6f}")
        txt_lines.append(f"max_speed_m_s={summary.max_speed_m_s:.6f}")
    if summary.mean_acceleration_m_s2 is not None:
        txt_lines.append(f"mean_acceleration_m_s2={summary.mean_acceleration_m_s2:.6f}")
        txt_lines.append(f"max_acceleration_m_s2={summary.max_acceleration_m_s2:.6f}")
    txt_lines.extend([
        f"mean_curvature_like={summary.mean_curvature_like:.6f}",
        f"p95_curvature_like={summary.p95_curvature_like:.6f}",
        f"max_curvature_like={summary.max_curvature_like:.6f}",
    ])
    if summary.mean_crack_risk is not None and summary.max_crack_risk is not None:
        txt_lines.append(f"mean_crack_risk={summary.mean_crack_risk:.6f}")
        txt_lines.append(f"max_crack_risk={summary.max_crack_risk:.6f}")
    summary_txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    fps: float | None = None,
    crack_params: Optional[CrackModelParams] = None,
    meters_per_pixel_override: Optional[float] = None,
) -> AnalysisSummary:
    """
    Run full analysis pipeline. When meters_per_pixel_override is set (>0), it is used
    as the length scale (m/px) for SI units; otherwise metadata.json is used if present.
    """
    from motionanalyzer.crack_model import compute_crack_risk, load_frame_metrics, get_user_params_path, load_params, CrackModelParams

    df, fps_val, m_from_meta = load_bundle(input_dir=input_dir, fps=fps)
    meters_per_pixel = (
        float(meters_per_pixel_override)
        if meters_per_pixel_override is not None and meters_per_pixel_override > 0
        else m_from_meta
    )
    frame_times = load_frame_timing(input_dir)
    timing_payload_path = input_dir / "frame_timing.json"
    timing_source = None
    median_dt_s = None
    effective_fps = None
    if timing_payload_path.exists():
        try:
            tp = json.loads(timing_payload_path.read_text(encoding="utf-8"))
            timing_source = tp.get("timing_source")
            median_dt_s = tp.get("median_dt_s")
            if median_dt_s is not None:
                median_dt_s = float(median_dt_s)
            effective_fps = tp.get("effective_fps")
            if effective_fps is not None:
                effective_fps = float(effective_fps)
                fps_val = effective_fps
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    vectors = compute_vectors(
        df=df,
        fps=fps_val,
        meters_per_pixel=meters_per_pixel,
        frame_times=frame_times,
    )
    dt_s = median_dt_s if median_dt_s and median_dt_s > 0 else (1.0 / fps_val)
    frame_metrics = load_frame_metrics(input_dir / "frame_metrics.csv")
    
    # Load crack model params: explicit > user config > default
    if crack_params is None:
        try:
            crack_params = load_params(get_user_params_path())
        except (ValueError, FileNotFoundError):
            crack_params = CrackModelParams()
    
    vectors = compute_crack_risk(
        vectors,
        frame_metrics,
        dt_s,
        meters_per_pixel=meters_per_pixel,
        params=crack_params,
    )
    summary = summarize(
        vectors=vectors,
        fps=fps_val,
        meters_per_pixel=meters_per_pixel,
        effective_fps=effective_fps,
        median_dt_s=median_dt_s,
        timing_source=timing_source,
    )
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
    field_names = {f.name for f in dataclasses.fields(AnalysisSummary)}
    kwargs = {k: payload[k] for k in payload if k in field_names}
    return AnalysisSummary(**kwargs)

