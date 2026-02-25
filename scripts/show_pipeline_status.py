"""
파이프라인 진행 상태 대시보드: 한눈에 진행도와 필요한 조치를 확인.

Usage:
  python scripts/show_pipeline_status.py          # stdout + reports/progress/pipeline_status.txt
  python scripts/show_pipeline_status.py --stdout # stdout only
  python scripts/show_pipeline_status.py --json   # JSON 출력
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

# fp_focused targets
TARGETS = {
    "normal": 16_000,
    "crack_in_bending": 1_050,
    "pre_damaged": 150,
    "thick_panel": 2_800,
}
TOTAL_TARGET = 20_000


def _bar(current: int, total: int, width: int = 20, fill: str = "#", empty: str = "-") -> str:
    if total <= 0:
        return empty * width
    pct = min(1.0, current / total)
    n = int(pct * width)
    return fill * n + empty * (width - n)


def _pct(current: int, total: int) -> float:
    return 100.0 * current / total if total > 0 else 0.0


def get_counts(base: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for folder in TARGETS:
        p = base / folder
        if not p.exists():
            counts[folder] = 0
        else:
            try:
                counts[folder] = sum(1 for x in p.iterdir() if x.is_dir())
            except OSError:
                counts[folder] = 0
    counts["total"] = sum(counts.get(k, 0) for k in TARGETS)
    return counts


def render_dashboard(base_fp: Path, out: Path) -> str:
    counts = get_counts(base_fp)
    manifest_path = base_fp / "manifest.json"
    manifest_exists = manifest_path.exists()
    analysis_path = repo_root / "reports" / "crack_detection_analysis" / "analysis.json"
    analysis_exists = analysis_path.exists()
    report_path = repo_root / "reports" / "deliverables" / "FPCB_Crack_Detection_Final_Report.docx"
    report_exists = report_path.exists()

    total = counts.get("total", 0)
    gen_pct = _pct(total, TOTAL_TARGET)
    gen_bar = _bar(total, TOTAL_TARGET)

    lines = [
        "",
        "========================================================================",
        "  FPCB Pipeline Status  |  " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "----------------------------------------------------------------------",
        f"  [1] Dataset (fp_focused)  {gen_bar} {gen_pct:5.1f}% ({total:,}/{TOTAL_TARGET:,})",
        "----------------------------------------------------------------------",
    ]

    for k, target in TARGETS.items():
        c = counts.get(k, 0)
        b = _bar(c, target, width=12)
        st = "[OK]" if c >= target else " [ ]"
        lines.append(f"      {k:20} {b} {c:5}/{target:<5} {st}")
    lines.append("----------------------------------------------------------------------")

    steps = [
        ("manifest.json", manifest_exists, "Dataset meta"),
        ("analysis.json", analysis_exists, "Analysis"),
        ("Final_Report.docx", report_exists, "Report"),
    ]
    for name, done, desc in steps:
        st = "[OK]" if done else "[ ]"
        lines.append(f"  [2] {desc:12} {name:22} {st}")
    lines.append("----------------------------------------------------------------------")

    # 필요한 조치
    actions: list[str] = []
    if not manifest_exists and total < TOTAL_TARGET:
        actions.append("  > python scripts/generate_ml_dataset.py --scale fp_focused --out data/synthetic/ml_dataset_fp_focused --resume")
    elif manifest_exists and not analysis_exists:
        actions.append("  > python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --zero-fp-priority --min-precision 0.99")
    elif analysis_exists and not report_exists:
        actions.append("  > python scripts/generate_final_report_docx.py")
    elif manifest_exists and analysis_exists and report_exists:
        actions.append("  > Pipeline complete. To close terminals: Terminal panel > right-click > Kill Terminal")
    else:
        actions.append("  > Resume generation with --resume option")

    lines.append("  [3] Next Actions")
    for a in actions:
        lines.append(a)
    lines.append("")
    lines.append("  Terminals: To close extra terminals in Cursor: Terminal panel >")
    lines.append("    right-click tab > Kill Terminal, or Terminal > Kill All.")
    lines.append("========================================================================")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="파이프라인 진행 상태 대시보드")
    ap.add_argument("--stdout", action="store_true", help="stdout만 출력 (파일 저장 안함)")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    ap.add_argument("--base-dir", default=None, help="fp_focused 데이터셋 경로")
    args = ap.parse_args()

    base = Path(args.base_dir or str(repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"))
    out_path = repo_root / "reports" / "progress" / "pipeline_status.txt"

    if args.json:
        counts = get_counts(base)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_dir": str(base),
            "counts": counts,
            "total_target": TOTAL_TARGET,
            "manifest_exists": (base / "manifest.json").exists(),
            "analysis_exists": (repo_root / "reports" / "crack_detection_analysis" / "analysis.json").exists(),
            "report_exists": (repo_root / "reports" / "deliverables" / "FPCB_Crack_Detection_Final_Report.docx").exists(),
        }
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    txt = render_dashboard(base, out_path)
    print(txt)

    if not args.stdout:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
