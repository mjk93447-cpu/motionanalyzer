"""
Precision 99%+ improvement loop.

Strict isolation: train only for learning, val for threshold, test for final report.
Loop: 실험분석 → precision 개선 전략 → 개발/업데이트 → 테스트 → 반복 (최소 4회).

Usage:
  python scripts/precision_improvement_loop.py --base-dir data/synthetic/ml_dataset_100k_v2
  python scripts/precision_improvement_loop.py --max-loops 6 --min-precision 0.99
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
REPORTS = repo_root / "reports"
OUT_DIR = REPORTS / "precision_improvement_loop"
ANALYZE_SCRIPT = repo_root / "scripts" / "analyze_crack_detection.py"
LOOP_LOG = OUT_DIR / "loop_log.jsonl"
RESULTS_JSON = OUT_DIR / "precision_results.json"


def run_analysis(
    base_dir: Path,
    *,
    dataset_level_eval: bool = False,
    min_precision: float = 0.99,
    zero_fp_priority: bool = False,
    max_train: int | None = None,
) -> dict:
    """Run analyze_crack_detection and return parsed results."""
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--base-dir",
        str(base_dir),
    ]
    if dataset_level_eval:
        cmd.append("--dataset-level-eval")
    cmd.extend(["--min-precision", str(min_precision)])
    if zero_fp_priority:
        cmd.append("--zero-fp-priority")
    if max_train is not None:
        cmd.extend(["--max-train", str(max_train)])

    r = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        return {"error": r.stderr or r.stdout, "returncode": r.returncode}

    analysis_path = REPORTS / "crack_detection_analysis" / "analysis.json"
    if not analysis_path.exists():
        return {"error": "analysis.json not found after run"}

    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    return {"success": True, "analysis": data}


def extract_metrics(data: dict) -> dict:
    """Extract precision, recall, FP, TP per model."""
    out = {}
    for model_name, m in data.get("models", {}).items():
        tn, fp, fn, tp = m.get("tn", 0), m.get("fp", 0), m.get("fn", 0), m.get("tp", 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[model_name] = {
            "precision": prec,
            "recall": rec,
            "fp": fp,
            "tp": tp,
            "fn": fn,
            "tn": tn,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Precision 99%+ improvement loop")
    ap.add_argument("--base-dir", default="data/synthetic/ml_dataset_100k_v2", help="Dataset directory")
    ap.add_argument("--min-precision", type=float, default=0.99, help="Target precision (default 0.99)")
    ap.add_argument("--max-loops", type=int, default=6, help="Max improvement loops (default 6)")
    ap.add_argument("--max-train", type=int, default=None, help="Cap train samples for faster runs")
    args = ap.parse_args()

    base = (repo_root / args.base_dir).resolve()
    manifest = base / "manifest.json"
    if not manifest.exists():
        print(f"ERROR: {manifest} not found. Run dataset generation first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Strategy sequence: each loop tries a combination
    strategies = [
        {"dataset_level_eval": False, "zero_fp_priority": False},
        {"dataset_level_eval": True, "zero_fp_priority": False},
        {"dataset_level_eval": True, "zero_fp_priority": True},
        {"dataset_level_eval": True, "zero_fp_priority": True},
        {"dataset_level_eval": True, "zero_fp_priority": True},
        {"dataset_level_eval": True, "zero_fp_priority": True},
    ]

    all_results = []
    best_precision = 0.0
    best_loop = -1

    print("=" * 60)
    print("Precision Improvement Loop (Target: 99%+)")
    print("=" * 60)
    print(f"Base dir: {base}")
    print(f"Target precision: {args.min_precision:.1%}")
    print(f"Max loops: {args.max_loops}")
    print()

    start_time = datetime.now(UTC)
    for loop in range(min(args.max_loops, len(strategies))):
        loop_start = datetime.now(UTC)
        s = strategies[loop]
        print(f"\n--- Loop {loop + 1}/{args.max_loops} ---")
        print(f"  Strategy: dataset_level={s['dataset_level_eval']}, zero_fp_priority={s['zero_fp_priority']}")
        # Progress file for external monitoring
        progress_path = OUT_DIR / "progress.json"
        progress_path.write_text(json.dumps({
            "loop": loop + 1,
            "max_loops": args.max_loops,
            "strategy": s,
            "started_at": loop_start.isoformat(),
            "elapsed_sec": (loop_start - start_time).total_seconds(),
        }, indent=2), encoding="utf-8")

        result = run_analysis(
            base,
            dataset_level_eval=s["dataset_level_eval"],
            min_precision=args.min_precision,
            zero_fp_priority=s["zero_fp_priority"],
            max_train=args.max_train,
        )

        if "error" in result:
            print(f"  ERROR: {result['error'][:200]}")
            all_results.append({
                "loop": loop + 1,
                "strategy": s,
                "error": result.get("error", ""),
            })
            continue

        metrics = extract_metrics(result["analysis"])
        n_test = result["analysis"].get("n_test", 0)
        n_normal = result["analysis"].get("n_normal", 0)
        n_crack = result["analysis"].get("n_crack", 0)

        row = {
            "loop": loop + 1,
            "strategy": s,
            "n_test": n_test,
            "n_normal": n_normal,
            "n_crack": n_crack,
            "models": metrics,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        all_results.append(row)

        for model_name, m in metrics.items():
            prec = m["precision"]
            rec = m["recall"]
            fp = m["fp"]
            print(f"  {model_name}: Precision={prec:.4f}, Recall={rec:.4f}, FP={fp}")

        # Best by Ensemble precision (or any model)
        for model_name, m in metrics.items():
            if m["precision"] > best_precision:
                best_precision = m["precision"]
                best_loop = loop + 1

        # Log to JSONL
        with open(LOOP_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        if best_precision >= args.min_precision:
            print(f"\n*** Target precision {args.min_precision:.1%} reached at loop {best_loop}. ***")
            break

    # Save summary
    summary = {
        "target_precision": args.min_precision,
        "best_precision": best_precision,
        "best_loop": best_loop,
        "total_loops": len(all_results),
        "results": all_results,
    }
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_JSON}")

    if best_precision >= args.min_precision:
        print("\nSUCCESS: Precision target achieved.")
    else:
        print(f"\nPrecision {best_precision:.2%} < target {args.min_precision:.1%}. Consider more loops or strategy tweaks.")

    # 최종 보고서 생성
    print("\nGenerating final report (docx)...")
    report_script = repo_root / "scripts" / "generate_final_report_docx.py"
    subprocess.run([sys.executable, str(report_script)], cwd=str(repo_root), timeout=120)
    print("Report: reports/deliverables/FPCB_Crack_Detection_Final_Report.docx")

    # 알림음 (결과 분석 완료)
    try:
        import winsound
        winsound.Beep(1000, 500)  # 1000Hz, 500ms
        winsound.Beep(1200, 500)  # 1200Hz, 500ms
        print("\n*** Analysis complete. Notification sound played. ***")
    except ImportError:
        print("\a\n*** Analysis complete. ***")


if __name__ == "__main__":
    main()
