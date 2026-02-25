"""
Monitor precision loop progress and run follow-up when results are ready.

Polls progress.json, loop_log.jsonl, precision_results.json.
When precision_results.json appears: generate report, play notification.

Usage:
  python scripts/monitor_precision_and_continue.py
  python scripts/monitor_precision_and_continue.py --poll-sec 120 --timeout-min 480
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
OUT_DIR = repo_root / "reports" / "precision_improvement_loop"
RESULTS_JSON = OUT_DIR / "precision_results.json"
PROGRESS_JSON = OUT_DIR / "progress.json"
LOOP_LOG = OUT_DIR / "loop_log.jsonl"
ANALYZE_SCRIPT = repo_root / "scripts" / "analyze_crack_detection.py"
REPORT_SCRIPT = repo_root / "scripts" / "generate_final_report_docx.py"


def get_progress() -> dict:
    """Current progress: loops done, current loop, ETA."""
    out = {"loops_done": 0, "current_loop": 0, "max_loops": 4, "eta_min": None, "status": "idle"}
    if not OUT_DIR.exists():
        return out

    # Completed loops from loop_log
    if LOOP_LOG.exists():
        lines = LOOP_LOG.read_text(encoding="utf-8").strip().splitlines()
        out["loops_done"] = len([l for l in lines if l.strip()])
        try:
            last = json.loads(lines[-1]) if lines else {}
            out["max_loops"] = last.get("loop", 4)
        except Exception:
            pass

    # Current loop from progress.json
    if PROGRESS_JSON.exists():
        try:
            p = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
            out["current_loop"] = p.get("loop", 0)
            out["max_loops"] = p.get("max_loops", 4)
            out["status"] = "running"
            elapsed = p.get("elapsed_sec", 0)
            if out["loops_done"] > 0 and elapsed > 0:
                # ETA: (remaining loops) * (avg time per loop)
                avg_per_loop = elapsed / out["loops_done"]
                remaining = out["max_loops"] - out["loops_done"]
                out["eta_min"] = (remaining * avg_per_loop) / 60.0
        except Exception:
            pass

    if RESULTS_JSON.exists():
        out["status"] = "complete"

    return out


def run_follow_up() -> None:
    """Generate report, play notification."""
    print("\n[Follow-up] Generating final report...")
    subprocess.run([sys.executable, str(REPORT_SCRIPT)], cwd=str(repo_root), timeout=120)
    print("  -> reports/deliverables/FPCB_Crack_Detection_Final_Report.docx")
    try:
        import winsound
        winsound.Beep(1000, 500)
        winsound.Beep(1200, 500)
        print("  -> Notification sound played.")
    except ImportError:
        print("\a")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-sec", type=int, default=60, help="Poll interval (default 60)")
    ap.add_argument("--timeout-min", type=float, default=480, help="Max wait minutes (default 480=8h)")
    args = ap.parse_args()

    print("=" * 60)
    print("Precision Loop Monitor - Poll until results, then follow-up")
    print("=" * 60)
    print(f"Poll every {args.poll_sec}s, timeout {args.timeout_min}min")
    print()

    deadline = time.time() + args.timeout_min * 60
    last_status = None

    while time.time() < deadline:
        p = get_progress()
        line = (
            f"  loops_done={p['loops_done']}/{p['max_loops']} "
            f"current={p['current_loop']} status={p['status']}"
        )
        if p.get("eta_min") is not None:
            line += f" ETA~{p['eta_min']:.0f}min"
        if line != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] {line}")
            last_status = line

        if p["status"] == "complete":
            print("\n*** precision_results.json found. Running follow-up. ***")
            run_follow_up()
            return

        time.sleep(args.poll_sec)

    print("\nTimeout reached. Results not yet available.")


if __name__ == "__main__":
    main()
