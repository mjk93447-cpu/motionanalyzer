"""
Evaluate roadmap scenario readiness (0-100 per scenario).
Target: top 4% = 96+ overall.

Scenarios: S01-S12 from ROADMAP_SCENARIOS_AND_READINESS.md
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent.parent
TARGET = 96


SCENARIOS = {
    "S01": {
        "name": "Temporal 모델 개선",
        "scripts": ["validate_temporal_synthetic.py", "benchmark_phase_b_comprehensive.py"],
        "docs": ["PHASE_B_INSIGHTS.md"],
    },
    "S02": {
        "name": "CPD 정확도 향상",
        "scripts": ["validate_cpd_optimization.py", "evaluate_goal1_cpd.py"],
        "docs": ["CHANGEPOINT_DETECTION.md"],
    },
    "S03": {
        "name": "충격파·진동 감지 강화",
        "scripts": ["validate_enhanced_synthetic.py"],
        "docs": ["SYNTHETIC_DATA_SPEC.md"],
    },
    "S04": {
        "name": "고급 특징 과적합 관리",
        "scripts": ["analyze_advanced_features_overfitting.py", "validate_advanced_features.py"],
        "docs": ["PHASE_B_INSIGHTS.md"],
    },
    "S05": {
        "name": "전체 파이프라인",
        "scripts": ["run_full_pipeline.ps1", "generate_ml_dataset.py", "analyze_crack_detection.py"],
        "docs": ["PIPELINE_SETUP_COMPLETE.md"],
    },
    "S06": {
        "name": "논문/리포트 작성",
        "scripts": ["generate_final_report_docx.py", "generate_final_report_ppt.py"],
        "docs": ["RELEASE_NOTES_TEMPLATE.md"],
    },
    "S07": {
        "name": "EXE 빌드·배포",
        "scripts": ["build_exe.ps1", "test_build_exe.ps1"],
        "docs": ["EXE_LOCAL_TEST_GUIDE.md"],
    },
    "S08": {
        "name": "QA 게이트 검증",
        "scripts": ["evaluate_synthetic_dataset_quality.py", "validate_enhanced_dream.py"],
        "docs": [],
    },
    "S09": {
        "name": "앙상블 가중치 최적화",
        "scripts": ["analyze_crack_detection.py", "benchmark_phase_b_comprehensive.py"],
        "docs": ["PHASE_B_INSIGHTS.md"],
    },
    "S10": {
        "name": "Goal1/Goal2 ML 평가",
        "scripts": ["evaluate_goal1_ml.py", "evaluate_goal2_ml.py", "evaluate_goals_summary.py"],
        "docs": ["PROJECT_GOALS.md"],
    },
    "S11": {
        "name": "배치 분석 (Phase D)",
        "scripts": ["analyze_crack_detection.py"],
        "docs": ["ANALYSIS_SCENARIOS_AND_OUTPUT_EVALUATION.md"],
    },
    "S12": {
        "name": "Phase C 준비",
        "scripts": [],
        "docs": [],
    },
    "S13": {
        "name": "GUI 테스트 시나리오",
        "scripts": ["run_gui_test_scenarios.py"],
        "docs": ["USER_GUIDE.md"],
    },
    "S14": {
        "name": "데이터셋 인벤토리",
        "scripts": ["analyze_bending_datasets.py"],
        "docs": ["BENDING_DATASETS_INVENTORY.md"],
    },
    "S15": {
        "name": "논문 재현성 검증",
        "scripts": ["run_full_pipeline.ps1", "benchmark_phase_b_comprehensive.py"],
        "docs": ["PHASE_B_INSIGHTS.md", "RELEASE_NOTES_TEMPLATE.md"],
    },
}


def _script_exists(name: str) -> bool:
    if name.endswith(".ps1"):
        return (REPO / "scripts" / name).exists()
    return (REPO / "scripts" / name).exists()


def _doc_exists(name: str) -> bool:
    for base in ["docs", "reports", ""]:
        if (REPO / base / name).exists():
            return True
    return False


def _agent_tool_exists(sid: str) -> bool:
    at = REPO / "scripts" / "agent_tools"
    runner = at / f"run_scenario_{sid}.ps1"
    return runner.exists() and runner.stat().st_size > 50


def _scenario_index_exists() -> bool:
    idx = REPO / "docs" / "ROADMAP_SCENARIOS_AND_READINESS.md"
    return idx.exists() and "시나리오별 필수 도구" in idx.read_text(encoding="utf-8", errors="replace")


def score_scenario(sid: str, cfg: dict) -> dict:
    scripts = cfg.get("scripts", [])
    docs = cfg.get("docs", [])
    script_ok = sum(1 for s in scripts if _script_exists(s))
    script_pct = (script_ok / len(scripts) * 40) if scripts else 40
    doc_ok = sum(1 for d in docs if _doc_exists(d))
    doc_pct = (doc_ok / len(docs) * 20) if docs else 20
    run_pct = 30 if (REPO / ".venv-gpu").exists() or (REPO / ".venv").exists() else 0
    oneclick = 10 if _agent_tool_exists(sid) else 0
    total = min(100, script_pct + doc_pct + run_pct + oneclick)
    return {
        "id": sid,
        "name": cfg["name"],
        "score": round(total, 1),
        "breakdown": {"scripts": script_pct, "docs": doc_pct, "env": run_pct, "oneclick": oneclick},
        "gaps": [],
    }


def run_evaluation() -> dict:
    results = []
    for sid, cfg in SCENARIOS.items():
        r = score_scenario(sid, cfg)
        if r["score"] < 100:
            if r["breakdown"]["scripts"] < 40:
                r["gaps"].append("스크립트 누락")
            if r["breakdown"]["docs"] < 20 and cfg.get("docs"):
                r["gaps"].append("문서 누락")
            if r["breakdown"]["oneclick"] == 0:
                r["gaps"].append("원클릭 러너 없음")
        results.append(r)
    # Bonus: scenario index doc (-5 if missing)
    index_penalty = 0 if _scenario_index_exists() else 5
    avg = max(0, (sum(r["score"] for r in results) / len(results) if results else 0) - index_penalty)
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall_score": round(avg, 1),
        "target": TARGET,
        "in_top_4": avg >= TARGET,
        "scenarios": results,
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="")
    args = ap.parse_args()
    out = run_evaluation()
    print("\n=== Scenario Readiness ===\n")
    print(f"Overall: {out['overall_score']}/100 (target >= {TARGET})")
    print(f"Result: {'PASS' if out['in_top_4'] else 'FAIL'}\n")
    for s in out["scenarios"]:
        icon = "OK" if s["score"] >= 90 else ("~" if s["score"] >= 70 else "FAIL")
        print(f"  [{icon}] {s['id']} {s['name']}: {s['score']}/100")
        if s["gaps"]:
            print(f"       Gaps: {', '.join(s['gaps'])}")
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nJSON: {args.output}")
    return 0 if out["in_top_4"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
