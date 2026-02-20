from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightConfig:
    min_points_per_frame: int = 150
    max_points_per_frame: int = 500
    max_allowed_missing_frames: int = 0


@dataclass(frozen=True)
class PreflightSummary:
    fps: float
    frame_count: int
    min_points: int
    max_points: int
    unique_index_count: int
    missing_frame_count: int
    passed: bool


def _frame_number(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _read_points(path: Path) -> tuple[int, set[int], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    if not lines:
        return 0, set(), [f"{path.name}: empty file"]
    if lines[0].strip() != "# x,y,index":
        errors.append(f"{path.name}: invalid header")

    indices: set[int] = set()
    point_count = 0
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 3:
            errors.append(f"{path.name}:{i} malformed row")
            continue
        try:
            int(cols[0])
            int(cols[1])
            idx = int(cols[2])
        except ValueError:
            errors.append(f"{path.name}:{i} non-integer value")
            continue
        indices.add(idx)
        point_count += 1
    return point_count, indices, errors


def preflight_realdata_bundle(
    input_dir: Path, config: PreflightConfig
) -> tuple[PreflightSummary, list[str]]:
    errors: list[str] = []
    fps_file = input_dir / "fps.txt"
    if not fps_file.exists():
        errors.append("fps.txt not found")
        fps = 0.0
    else:
        try:
            fps = float(fps_file.read_text(encoding="utf-8").strip())
        except ValueError:
            fps = 0.0
            errors.append("fps.txt is not a valid number")

    frame_files = sorted(input_dir.glob("frame_*.txt"))
    if not frame_files:
        errors.append("No frame_*.txt files found")

    frame_numbers = [_frame_number(p) for p in frame_files]
    missing = 0
    if frame_numbers and min(frame_numbers) >= 0:
        expected = set(range(min(frame_numbers), max(frame_numbers) + 1))
        actual = set(frame_numbers)
        missing = len(expected - actual)
        if missing > config.max_allowed_missing_frames:
            errors.append(f"Missing frame files detected: {missing}")

    min_points = 10**9
    max_points = 0
    all_indices: set[int] = set()
    per_frame_index_sets: list[set[int]] = []
    for frame in frame_files:
        point_count, indices, frame_errors = _read_points(frame)
        errors.extend(frame_errors)
        min_points = min(min_points, point_count)
        max_points = max(max_points, point_count)
        all_indices.update(indices)
        per_frame_index_sets.append(indices)

    if frame_files:
        if min_points < config.min_points_per_frame:
            errors.append(
                f"Point count too low in at least one frame: min={min_points}, "
                f"required>={config.min_points_per_frame}"
            )
        if max_points > config.max_points_per_frame:
            errors.append(
                f"Point count too high in at least one frame: max={max_points}, "
                f"required<={config.max_points_per_frame}"
            )

    if per_frame_index_sets:
        base = per_frame_index_sets[0]
        for i, index_set in enumerate(per_frame_index_sets[1:], start=1):
            if index_set != base:
                errors.append(f"Frame index set mismatch at frame offset {i}")
                break

    summary = PreflightSummary(
        fps=fps,
        frame_count=len(frame_files),
        min_points=0 if min_points == 10**9 else min_points,
        max_points=max_points,
        unique_index_count=len(all_indices),
        missing_frame_count=missing,
        passed=not errors,
    )
    return summary, errors


def write_preflight_report(
    report_path: Path,
    summary: PreflightSummary,
    errors: list[str],
    config: PreflightConfig,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "summary": asdict(summary),
        "errors": errors,
        "preflight_config": asdict(config),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_log_template(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "build_version",
                "scenario_tag",
                "frame_count",
                "fps",
                "elapsed_ms",
                "validation_passed",
                "failure_signature",
                "notes_redacted",
            ]
        )
