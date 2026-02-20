from pathlib import Path

from motionanalyzer.testsuite import run_synthetic_feature_suite


def test_run_synthetic_feature_suite_creates_report(tmp_path: Path) -> None:
    report = run_synthetic_feature_suite(output_root=tmp_path / "suite")
    assert report.exists()
    assert report.name == "synthetic_feature_suite_report.json"
