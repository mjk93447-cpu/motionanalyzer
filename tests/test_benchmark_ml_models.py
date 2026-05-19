"""Benchmark script gate (skipped without trained bundle)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_benchmark_import() -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "scripts" / "benchmark_ml_models.py"
    assert script.exists()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "release" / "models" / "bundle_manifest.json").exists()
    and not (__import__("os").environ.get("APPDATA") and (
        Path(__import__("os").environ["APPDATA"]) / "motionanalyzer" / "models" / "bundle_manifest.json"
    ).exists()),
    reason="No pretrained bundle on disk",
)
def test_benchmark_runs_when_bundle_present() -> None:
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    manifest = repo / "data" / "synthetic" / "ml_pretrain_balanced_3k_60f" / "manifest.json"
    if not manifest.exists():
        pytest.skip("ml_pretrain_balanced_3k_60f manifest not on disk")
    cmd = [
        sys.executable,
        str(repo / "scripts" / "benchmark_ml_models.py"),
        "--models-dir",
        str(repo / "release" / "models"),
        "--manifest",
        str(manifest),
    ]
    r = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr + r.stdout
