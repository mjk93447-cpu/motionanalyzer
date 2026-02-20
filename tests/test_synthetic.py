from pathlib import Path

from motionanalyzer.synthetic import (
    SyntheticConfig,
    generate_synthetic_bundle,
    validate_synthetic_bundle,
)


def test_generate_synthetic_bundle_creates_expected_files(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_case"
    config = SyntheticConfig(frames=8, points_per_frame=14, fps=24.0, seed=7, scenario="normal")
    generate_synthetic_bundle(output_dir=out, config=config)

    frame_files = sorted(out.glob("frame_*.txt"))
    assert len(frame_files) == 8
    assert (out / "fps.txt").exists()
    assert (out / "frame_metrics.csv").exists()
    assert (out / "metadata.json").exists()

    sample = frame_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert sample[0] == "# x,y,index"
    assert len(sample) == 1 + 14
    assert sample[1].endswith(",1")


def test_validate_synthetic_bundle_with_matching_scenario(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_crack"
    config = SyntheticConfig(frames=60, points_per_frame=180, fps=30.0, seed=11, scenario="crack")
    generate_synthetic_bundle(output_dir=out, config=config)
    ok, errors = validate_synthetic_bundle(output_dir=out, scenario="crack")
    assert ok, f"Expected crack validation to pass, got: {errors}"
