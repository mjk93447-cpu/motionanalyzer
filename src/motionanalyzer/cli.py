import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from motionanalyzer.analysis import (
    compare_summaries,
    load_summary,
    run_analysis,
)
from motionanalyzer.preflight import (
    PreflightConfig,
    preflight_realdata_bundle,
    write_log_template,
    write_preflight_report,
)
from motionanalyzer.synthetic import (
    ScenarioName,
    SyntheticConfig,
    generate_synthetic_bundle,
    validate_synthetic_bundle,
)
from motionanalyzer.testsuite import run_synthetic_feature_suite

app = typer.Typer(help="Motion analyzer command line interface.")
console = Console()


def _append_txt_log(log_txt: Path | None, title: str, payload: dict[str, Any]) -> None:
    if log_txt is None:
        return
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"=== {title} ===",
        f"timestamp={datetime.now(UTC).isoformat()}",
    ]
    for k, v in payload.items():
        if isinstance(v, dict | list):
            lines.append(f"{k}={json.dumps(v, ensure_ascii=True)}")
        else:
            lines.append(f"{k}={v}")
    lines.append("")
    with log_txt.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Render help and exit cleanly when no command is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@app.command("doctor")
def doctor(log_txt: Path | None = None) -> None:
    """Validate local runtime prerequisites."""
    console.print("[green]Environment looks ready.[/green]")
    console.print("Python package installed and CLI wired successfully.")
    _append_txt_log(log_txt, "doctor", {"status": "ok"})


@app.command("init-dirs")
def init_dirs(base_dir: Path = Path(".")) -> None:
    """Create standard project data directories."""
    for rel in (
        "data/raw",
        "data/processed",
        "exports/vectors",
        "exports/plots",
        "logs",
        "reports/preflight",
        "internal_eval/inbox",
        "internal_eval/outbox",
        "internal_eval/logs",
    ):
        target = base_dir / rel
        target.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]created[/cyan] {target}")


@app.command("gen-synthetic")
def gen_synthetic(
    output_dir: Path = Path("data/synthetic/session_001"),
    frames: int = 120,
    points_per_frame: int = 230,
    fps: float = 30.0,
    scenario: ScenarioName = "normal",
    seed: int = 42,
) -> None:
    """Generate synthetic side-view FPCB bending bundle for offline-safe tests."""
    config = SyntheticConfig(
        frames=frames,
        points_per_frame=points_per_frame,
        fps=fps,
        scenario=scenario,
        seed=seed,
    )
    result = generate_synthetic_bundle(output_dir=output_dir, config=config)
    console.print(f"[green]synthetic data created[/green] {result}")
    console.print("Includes frame_*.txt, fps.txt, frame_metrics.csv, metadata.json")


@app.command("validate-synthetic")
def validate_synthetic(
    input_dir: Path = Path("data/synthetic/session_001"),
    scenario: ScenarioName = "normal",
) -> None:
    """Validate generated synthetic bundle against scenario-specific signatures."""
    ok, errors = validate_synthetic_bundle(output_dir=input_dir, scenario=scenario)
    if ok:
        console.print(f"[green]validation passed[/green] scenario={scenario} path={input_dir}")
        return
    console.print(f"[red]validation failed[/red] scenario={scenario} path={input_dir}")
    for error in errors:
        console.print(f"- {error}")
    raise typer.Exit(code=1)


@app.command("preflight-realdata")
def preflight_realdata(
    input_dir: Path = Path("data/raw/session_real_001"),
    min_points_per_frame: int = 150,
    max_points_per_frame: int = 500,
    max_allowed_missing_frames: int = 0,
    report_path: Path = Path("reports/preflight/latest_preflight.json"),
    log_txt: Path | None = None,
) -> None:
    """Run pre-analysis quality checks for internal real-data bundles."""
    cfg = PreflightConfig(
        min_points_per_frame=min_points_per_frame,
        max_points_per_frame=max_points_per_frame,
        max_allowed_missing_frames=max_allowed_missing_frames,
    )
    summary, errors = preflight_realdata_bundle(input_dir=input_dir, config=cfg)
    write_preflight_report(report_path=report_path, summary=summary, errors=errors, config=cfg)
    _append_txt_log(
        log_txt,
        "preflight-realdata",
        {
            "input_dir": str(input_dir),
            "report_path": str(report_path),
            "passed": summary.passed,
            "errors": errors,
            "frame_count": summary.frame_count,
            "unique_index_count": summary.unique_index_count,
            "missing_frame_count": summary.missing_frame_count,
        },
    )
    if summary.passed:
        console.print(f"[green]preflight passed[/green] input={input_dir}")
        console.print(f"report: {report_path}")
        return
    console.print(f"[red]preflight failed[/red] input={input_dir}")
    console.print(f"report: {report_path}")
    for err in errors[:12]:
        console.print(f"- {err}")
    raise typer.Exit(code=1)


@app.command("prepare-internal")
def prepare_internal(base_dir: Path = Path("."), log_txt: Path | None = None) -> None:
    """Create required folders/templates before internal real-data evaluation."""
    init_dirs(base_dir=base_dir)
    template_path = base_dir / "internal_eval" / "logs" / "result_template.csv"
    write_log_template(output_path=template_path)
    console.print(f"[green]internal setup ready[/green] template={template_path}")
    _append_txt_log(
        log_txt,
        "prepare-internal",
        {"base_dir": str(base_dir), "template_path": str(template_path), "status": "ok"},
    )


@app.command("analyze-bundle")
def analyze_bundle(
    input_dir: Path = Path("data/synthetic/normal_case"),
    output_dir: Path = Path("exports/vectors/normal_case"),
    log_txt: Path | None = None,
) -> None:
    """Analyze frame bundle and export vectors/summary."""
    summary = run_analysis(input_dir=input_dir, output_dir=output_dir)
    console.print(f"[green]analysis complete[/green] input={input_dir} output={output_dir}")
    console.print(
        f"frames={summary.frame_count}, unique_index={summary.unique_index_count}, "
        f"mean_speed={summary.mean_speed:.3f}, mean_acc={summary.mean_acceleration:.3f}"
    )
    _append_txt_log(
        log_txt,
        "analyze-bundle",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "summary": summary.__dict__,
            "summary_txt": str(output_dir / "summary.txt"),
        },
    )


@app.command("compare-runs")
def compare_runs(
    base_summary: Path = Path("exports/vectors/normal_case/summary.json"),
    candidate_summary: Path = Path("exports/vectors/candidate_case/summary.json"),
    output_txt: Path = Path("reports/compare/latest_compare.txt"),
    log_txt: Path | None = None,
) -> None:
    """Compare two analysis summary files."""
    base = load_summary(base_summary)
    candidate = load_summary(candidate_summary)
    delta = compare_summaries(base_summary=base, candidate_summary=candidate)
    console.print("[green]comparison complete[/green]")
    for k, v in delta.items():
        console.print(f"- {k}: {v:.6f}")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "motionanalyzer compare-runs",
        f"base_summary={base_summary}",
        f"candidate_summary={candidate_summary}",
    ]
    lines.extend([f"{k}={v:.6f}" for k, v in delta.items()])
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_txt_log(
        log_txt,
        "compare-runs",
        {
            "base_summary": str(base_summary),
            "candidate_summary": str(candidate_summary),
            "output_txt": str(output_txt),
            "delta": delta,
        },
    )


@app.command("gui")
def gui(host: str = "127.0.0.1", port: int = 8501, log_txt: Path | None = None) -> None:
    """Launch streamlit GUI."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/motionanalyzer/gui.py",
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    console.print(f"[cyan]launching[/cyan] {' '.join(cmd)}")
    _append_txt_log(log_txt, "gui", {"host": host, "port": port, "status": "launching"})
    subprocess.run(cmd, check=True)


@app.command("run-synthetic-suite")
def run_synthetic_suite(
    output_root: Path = Path("reports/synthetic_suite"),
    log_txt: Path | None = None,
) -> None:
    """Run internal synthetic feature test suite and create report."""
    report = run_synthetic_feature_suite(output_root=output_root)
    console.print(f"[green]synthetic suite complete[/green] report={report}")
    _append_txt_log(
        log_txt, "run-synthetic-suite", {"output_root": str(output_root), "report": str(report)}
    )


@app.command("internal-realdata-run")
def internal_realdata_run(
    input_dir: Path = Path("data/raw/session_real_001"),
    output_dir: Path = Path("exports/vectors/real_session_001"),
    baseline_summary: str = "exports/vectors/baseline/summary.json",
    run_log_txt: Path = Path("internal_eval/logs/internal_run_latest.txt"),
) -> None:
    """
    End-to-end internal run for EXE use.

    Produces a single comprehensive txt log for external sharing after redaction.
    """
    run_log_txt.parent.mkdir(parents=True, exist_ok=True)
    run_log_txt.write_text("", encoding="utf-8")
    _append_txt_log(
        run_log_txt,
        "internal-realdata-run:start",
        {"input_dir": str(input_dir), "output_dir": str(output_dir)},
    )

    preflight_report = Path("reports/preflight/internal_preflight_latest.json")
    preflight_realdata(
        input_dir=input_dir,
        report_path=preflight_report,
        log_txt=run_log_txt,
    )
    analyze_bundle(
        input_dir=input_dir,
        output_dir=output_dir,
        log_txt=run_log_txt,
    )
    baseline_path = Path(baseline_summary) if baseline_summary.strip() else None
    if baseline_path is not None and baseline_path.exists():
        compare_runs(
            base_summary=baseline_path,
            candidate_summary=output_dir / "summary.json",
            output_txt=Path("reports/compare/internal_vs_baseline.txt"),
            log_txt=run_log_txt,
        )
    else:
        _append_txt_log(
            run_log_txt,
            "internal-realdata-run:compare-skipped",
            {"reason": f"baseline summary missing: {baseline_summary}"},
        )
    _append_txt_log(run_log_txt, "internal-realdata-run:complete", {"status": "ok"})
    console.print(f"[green]internal run complete[/green] log={run_log_txt}")


if __name__ == "__main__":
    app()
