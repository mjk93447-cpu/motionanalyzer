"""
Wait for 100k dataset manifest.json, then run ML analysis and paper generation.

Usage:
  python scripts/wait_then_analyze.py --base-dir data/synthetic/ml_dataset_100k_v2
  python scripts/wait_then_analyze.py --sleep-minutes 90  # cap sleep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

# 100k scale targets
TARGETS = {
    "normal": 75_000,       # 70k + 5k light_dist
    "crack_in_bending": 16_000,  # 8k crack + 3k micro + 3k over + 1k under + 1k jig
    "pre_damaged": 4_000,
    "thick_panel": 5_000,
}
TOTAL_TARGET = 100_000

# Rate from pipeline_orchestrator.log: ~250-280 samples per 2 min during crack phase
SAMPLES_PER_MIN = 130  # conservative


def get_counts(base_dir: Path) -> dict[str, int]:
    counts = {}
    for folder in ("normal", "crack_in_bending", "pre_damaged", "thick_panel"):
        p = base_dir / folder
        if not p.exists():
            counts[folder] = 0
        else:
            counts[folder] = sum(1 for _ in p.iterdir() if _.is_dir())
    counts["total"] = sum(counts.get(k, 0) for k in TARGETS)
    return counts


def estimate_remaining_minutes(counts: dict[str, int]) -> float:
    remaining = 0
    remaining += max(0, TARGETS["normal"] - counts.get("normal", 0))
    remaining += max(0, TARGETS["crack_in_bending"] - counts.get("crack_in_bending", 0))
    remaining += max(0, TARGETS["pre_damaged"] - counts.get("pre_damaged", 0))
    remaining += max(0, TARGETS["thick_panel"] - counts.get("thick_panel", 0))
    return remaining / SAMPLES_PER_MIN if SAMPLES_PER_MIN > 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Wait for manifest then run analysis")
    ap.add_argument("--base-dir", default="data/synthetic/ml_dataset_100k_v2", help="Dataset directory")
    ap.add_argument("--sleep-minutes", type=float, default=180, help="Max sleep minutes (default 180)")
    ap.add_argument("--poll-interval", type=int, default=60, help="Poll interval in seconds after sleep")
    ap.add_argument("--no-sleep", action="store_true", help="Skip initial sleep, poll immediately")
    args = ap.parse_args()

    base = (repo_root / args.base_dir).resolve()
    manifest_path = base / "manifest.json"

    print("=" * 60)
    print("Wait & Analyze Pipeline")
    print("=" * 60)
    print(f"Base dir: {base}")
    print(f"Manifest: {manifest_path}")
    print()

    if manifest_path.exists():
        print("[OK] manifest.json already exists. Proceeding to analysis.")
    else:
        counts = get_counts(base)
        eta_min = estimate_remaining_minutes(counts)
        print(f"Current: normal={counts.get('normal',0)} crack={counts.get('crack_in_bending',0)} "
              f"predam={counts.get('pre_damaged',0)} thick={counts.get('thick_panel',0)} total={counts.get('total',0)}")
        print(f"ETA: ~{eta_min:.0f} minutes remaining")
        print()

        if not args.no_sleep and eta_min > 1:
            sleep_min = min(eta_min * 0.9, args.sleep_minutes)  # 90% of ETA, cap at max
            sleep_sec = int(sleep_min * 60)
            print(f"Sleeping {sleep_min:.0f} minutes ({sleep_sec}s)...")
            time.sleep(sleep_sec)
            print("Sleep done. Polling for manifest...")
        else:
            print("Skipping sleep. Polling for manifest...")

        while not manifest_path.exists():
            time.sleep(args.poll_interval)
            counts = get_counts(base)
            print(f"  waiting... total={counts.get('total',0)}/{TOTAL_TARGET}")

        print("[OK] manifest.json detected.")

    # Run analysis
    print()
    print("[1/2] Running crack detection analysis...")
    analyze_script = repo_root / "scripts" / "analyze_crack_detection.py"
    r1 = subprocess.run(
        [sys.executable, str(analyze_script), "--base-dir", str(base)],
        cwd=str(repo_root),
    )
    if r1.returncode != 0:
        print("[WARN] analyze_crack_detection.py exited with code", r1.returncode)

    # Generate report
    print()
    print("[2/2] Generating final report (docx)...")
    report_script = repo_root / "scripts" / "generate_final_report_docx.py"
    r2 = subprocess.run([sys.executable, str(report_script)], cwd=str(repo_root))
    if r2.returncode != 0:
        print("[WARN] generate_final_report_docx.py exited with code", r2.returncode)

    print()
    print("=" * 60)
    print("Pipeline complete.")
    print("  - Analysis: reports/crack_detection_analysis/")
    print("  - Report: reports/deliverables/FPCB_Crack_Detection_Final_Report.docx")
    print("=" * 60)


if __name__ == "__main__":
    main()
