from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from motionanalyzer.analysis import AnalysisSummary, compare_summaries, run_analysis
from motionanalyzer.synthetic import (
    ScenarioName,
    SyntheticConfig,
    generate_synthetic_bundle,
    validate_synthetic_bundle,
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    synthetic_validation_passed: bool
    synthetic_validation_errors: list[str]
    summary: AnalysisSummary
    delta_vs_normal: dict[str, float]


def run_synthetic_feature_suite(output_root: Path, fps: float = 30.0) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    scenarios = ["normal", "crack", "pre_damage", "thick_panel", "uv_overcured"]
    summaries: dict[str, AnalysisSummary] = {}
    results: list[ScenarioResult] = []

    for scenario in scenarios:
        scenario_name = cast(ScenarioName, scenario)
        in_dir = output_root / "synthetic" / scenario
        out_dir = output_root / "analysis" / scenario
        cfg = SyntheticConfig(frames=120, points_per_frame=230, fps=fps, scenario=scenario_name)
        generate_synthetic_bundle(output_dir=in_dir, config=cfg)
        ok, errors = validate_synthetic_bundle(output_dir=in_dir, scenario=scenario_name)
        summary = run_analysis(input_dir=in_dir, output_dir=out_dir)
        summaries[scenario] = summary
        results.append(
            ScenarioResult(
                scenario=scenario,
                synthetic_validation_passed=ok,
                synthetic_validation_errors=errors,
                summary=summary,
                delta_vs_normal={},
            )
        )

    base = summaries["normal"]
    finalized: list[ScenarioResult] = []
    for item in results:
        delta = compare_summaries(base, item.summary)
        finalized.append(
            ScenarioResult(
                scenario=item.scenario,
                synthetic_validation_passed=item.synthetic_validation_passed,
                synthetic_validation_errors=item.synthetic_validation_errors,
                summary=item.summary,
                delta_vs_normal=delta,
            )
        )

    payload = {
        "suite": "synthetic_feature_suite",
        "scenario_count": len(finalized),
        "results": [
            {
                "scenario": r.scenario,
                "synthetic_validation_passed": r.synthetic_validation_passed,
                "synthetic_validation_errors": r.synthetic_validation_errors,
                "summary": asdict(r.summary),
                "delta_vs_normal": r.delta_vs_normal,
            }
            for r in finalized
        ],
    }
    report_path = output_root / "synthetic_feature_suite_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return report_path
