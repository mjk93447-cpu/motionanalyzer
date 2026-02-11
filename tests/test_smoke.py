from pathlib import Path

from typer.testing import CliRunner

from motionanalyzer.cli import app


def test_doctor_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Environment looks ready" in result.stdout


def test_validate_synthetic_command(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "synthetic_normal"
    gen = runner.invoke(
        app,
        [
            "gen-synthetic",
            "--output-dir",
            str(out),
            "--frames",
            "60",
            "--points-per-frame",
            "180",
            "--scenario",
            "normal",
        ],
    )
    assert gen.exit_code == 0

    result = runner.invoke(
        app,
        ["validate-synthetic", "--input-dir", str(out), "--scenario", "normal"],
    )
    assert result.exit_code == 0
    assert "validation passed" in result.stdout


def test_prepare_internal_command_creates_template(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["prepare-internal", "--base-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert (tmp_path / "internal_eval" / "logs" / "result_template.csv").exists()


def test_internal_realdata_run_creates_txt_log(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    (input_dir / "fps.txt").write_text("30\n", encoding="utf-8")
    frame_data = "# x,y,index\n" + "\n".join([f"{100+i},{200+i},{i+1}" for i in range(180)]) + "\n"
    (input_dir / "frame_00000.txt").write_text(frame_data, encoding="utf-8")
    (input_dir / "frame_00001.txt").write_text(frame_data, encoding="utf-8")

    run_log = tmp_path / "internal_run.txt"
    result = runner.invoke(
        app,
        [
            "internal-realdata-run",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--baseline-summary",
            "",
            "--run-log-txt",
            str(run_log),
        ],
    )
    assert result.exit_code == 0
    assert run_log.exists()
    assert "internal-realdata-run:complete" in run_log.read_text(encoding="utf-8")
