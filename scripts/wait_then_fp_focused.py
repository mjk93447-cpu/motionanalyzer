"""
Wait for fp_focused dataset manifest, then run FP-minimization analysis and report.

ETA from current progress, sleep, poll, then analyze (dataset-level, zero-FP) + report.

Usage:
  python scripts/wait_then_fp_focused.py
  python scripts/wait_then_fp_focused.py --no-sleep  # poll only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
BASE_DIR = repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"

# fp_focused targets (normal folder = normal + light_dist)
TARGETS = {
    "normal": 16_000,   # 15k normal + 1k light_dist
    "crack_in_bending": 1_050,  # 600 crack+uv + 200 micro + 100 over + 100 under + 50 jig
    "pre_damaged": 150,
    "thick_panel": 2_800,
}
TOTAL_TARGET = 20_000

# Rate: ~330 samples/min from observation (5280 in 16 min)
SAMPLES_PER_MIN = 300


def get_counts(base_dir: Path) -> dict[str, int]:
    counts = {}
    for folder in ("normal", "crack_in_bending", "pre_damaged", "thick_panel"):
        p = base_dir / folder
        if not p.exists():
            counts[folder] = 0
        else:
            try:
                counts[folder] = sum(1 for x in p.iterdir() if x.is_dir())
            except OSError:
                counts[folder] = 0
    counts["total"] = sum(counts.get(k, 0) for k in TARGETS)
    return counts


def estimate_remaining_minutes(counts: dict[str, int]) -> float:
    remaining = 0
    for k, target in TARGETS.items():
        remaining += max(0, target - counts.get(k, 0))
    return remaining / SAMPLES_PER_MIN if SAMPLES_PER_MIN > 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Wait for fp_focused manifest, then run FP-focused analysis")
    ap.add_argument("--base-dir", default=str(BASE_DIR), help="Dataset directory")
    ap.add_argument("--sleep-minutes", type=float, default=90, help="Max sleep minutes")
    ap.add_argument("--poll-interval", type=int, default=60, help="Poll interval (sec)")
    ap.add_argument("--no-sleep", action="store_true", help="Skip sleep, poll immediately")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    manifest_path = base / "manifest.json"

    print("=" * 60)
    print("FP-Focused Pipeline: Wait -> Analyze -> Report")
    print("=" * 60)
    print(f"Base dir: {base}")
    print(f"Target: {TOTAL_TARGET} samples")
    print()

    if manifest_path.exists():
        print("[OK] manifest.json exists. Proceeding to analysis.")
    else:
        counts = get_counts(base)
        eta_min = estimate_remaining_minutes(counts)
        print(f"Current: normal={counts.get('normal',0)} crack={counts.get('crack_in_bending',0)} "
              f"predam={counts.get('pre_damaged',0)} thick={counts.get('thick_panel',0)} total={counts.get('total',0)}")
        print(f"ETA: ~{eta_min:.0f} minutes")
        print()

        if not args.no_sleep and eta_min > 1:
            sleep_min = min(eta_min * 0.9, args.sleep_minutes)
            sleep_sec = int(sleep_min * 60)
            print(f"Sleeping {sleep_min:.0f} min ({sleep_sec}s)...")
            time.sleep(sleep_sec)
            print("Sleep done. Polling for manifest...")
        else:
            print("Polling for manifest...")

        while not manifest_path.exists():
            time.sleep(args.poll_interval)
            counts = get_counts(base)
            print(f"  waiting... total={counts.get('total',0)}/{TOTAL_TARGET}")

        print("[OK] manifest.json detected.")

    # Analysis with FP minimization
    print()
    print("[1/2] Running crack detection analysis (dataset-level, zero-FP)...")
    analyze_script = repo_root / "scripts" / "analyze_crack_detection.py"
    r1 = subprocess.run(
        [
            sys.executable, str(analyze_script),
            "--base-dir", str(base),
            "--dataset-level-eval",
            "--zero-fp-priority",
            "--min-precision", "0.99",
        ],
        cwd=str(repo_root),
        timeout=3600,
    )
    if r1.returncode != 0:
        print("[WARN] analyze_crack_detection.py exited with code", r1.returncode)

    # Report
    print()
    print("[2/2] Generating final report...")
    report_script = repo_root / "scripts" / "generate_final_report_docx.py"
    r2 = subprocess.run([sys.executable, str(report_script)], cwd=str(repo_root), timeout=120)
    if r2.returncode != 0:
        print("[WARN] generate_final_report_docx.py exited with code", r2.returncode)

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print("  Analysis: reports/crack_detection_analysis/")
    print("  Report: reports/deliverables/FPCB_Crack_Detection_Final_Report.docx")
    print("=" * 60)

    try:
        import winsound
        winsound.Beep(1000, 500)
        winsound.Beep(1200, 500)
    except ImportError:
        print("\a")


if __name__ == "__main__":
    main()
