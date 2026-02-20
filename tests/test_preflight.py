from pathlib import Path

from motionanalyzer.preflight import PreflightConfig, preflight_realdata_bundle


def _write_frame(path: Path, rows: list[tuple[int, int, int]]) -> None:
    lines = ["# x,y,index"]
    for x, y, idx in rows:
        lines.append(f"{x},{y},{idx}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_preflight_passes_on_valid_bundle(tmp_path: Path) -> None:
    input_dir = tmp_path / "real"
    input_dir.mkdir()
    (input_dir / "fps.txt").write_text("30\n", encoding="utf-8")

    rows = [(100 + i, 200 + i, i + 1) for i in range(160)]
    _write_frame(input_dir / "frame_00000.txt", rows)
    _write_frame(input_dir / "frame_00001.txt", rows)

    summary, errors = preflight_realdata_bundle(
        input_dir=input_dir,
        config=PreflightConfig(min_points_per_frame=150, max_points_per_frame=500),
    )
    assert summary.passed
    assert errors == []
    assert summary.frame_count == 2


def test_preflight_fails_on_missing_frames(tmp_path: Path) -> None:
    input_dir = tmp_path / "real_missing"
    input_dir.mkdir()
    (input_dir / "fps.txt").write_text("30\n", encoding="utf-8")

    rows = [(100 + i, 200 + i, i + 1) for i in range(160)]
    _write_frame(input_dir / "frame_00000.txt", rows)
    _write_frame(input_dir / "frame_00002.txt", rows)

    summary, errors = preflight_realdata_bundle(
        input_dir=input_dir,
        config=PreflightConfig(min_points_per_frame=150, max_points_per_frame=500),
    )
    assert not summary.passed
    assert any("Missing frame files detected" in e for e in errors)
