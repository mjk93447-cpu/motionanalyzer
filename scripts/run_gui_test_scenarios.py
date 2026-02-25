"""
Run analysis test scenarios that mirror GUI user workflows.

Simulates: Analyze Tab (normal + crack), Compare Tab, Scale (mm/px) input.
Uses the same run_analysis/compare_summaries as the GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.analysis import compare_summaries, load_summary, run_analysis

BASE = repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"
EXPORTS = repo_root / "exports" / "vectors" / "gui_test_scenarios"


def main() -> None:
    print("=" * 60)
    print("GUI Test Scenarios (CLI simulation)")
    print("=" * 60)

    EXPORTS.mkdir(parents=True, exist_ok=True)

    # Scenario 1: Analyze normal_0001 (no scale override - use metadata)
    print("\n[Scenario 1] Analyze normal_0001 (metadata scale)")
    in1 = BASE / "normal" / "normal_0001"
    out1 = EXPORTS / "normal_0001"
    if in1.exists():
        s1 = run_analysis(input_dir=in1, output_dir=out1, fps=30.0)
        print(f"  mean_speed: {s1.mean_speed:.3f} px/s")
        if s1.mean_speed_m_s:
            print(f"  mean_speed_m_s: {s1.mean_speed_m_s:.6f} m/s")
        print(f"  meters_per_pixel: {s1.meters_per_pixel}")
    else:
        print("  SKIP: input not found")

    # Scenario 2: Analyze normal_0001 with Scale (mm/px) = 0.1 override
    print("\n[Scenario 2] Analyze normal_0001 with Scale 0.1 mm/px (user override)")
    out2 = EXPORTS / "normal_0001_scale01"
    if in1.exists():
        s2 = run_analysis(
            input_dir=in1,
            output_dir=out2,
            fps=30.0,
            meters_per_pixel_override=0.1 * 1e-3,  # 0.1 mm/px -> m/px
        )
        print(f"  mean_speed_m_s: {s2.mean_speed_m_s:.6f} m/s")
        print(f"  max_acceleration_m_s2: {s2.max_acceleration_m_s2:.6f} m/s²")
    else:
        print("  SKIP: input not found")

    # Scenario 3: Analyze crack_0001
    print("\n[Scenario 3] Analyze crack_0001")
    in3 = BASE / "crack_in_bending" / "crack_0001"
    out3 = EXPORTS / "crack_0001"
    if in3.exists():
        s3 = run_analysis(
            input_dir=in3,
            output_dir=out3,
            fps=30.0,
            meters_per_pixel_override=0.1 * 1e-3,
        )
        print(f"  mean_speed_m_s: {s3.mean_speed_m_s:.6f} m/s")
        print(f"  max_crack_risk: {s3.max_crack_risk}")
    else:
        print("  SKIP: input not found")

    # Scenario 4: Compare normal vs crack
    print("\n[Scenario 4] Compare normal_0001 vs crack_0001")
    sum1 = EXPORTS / "normal_0001_scale01" / "summary.json"
    sum3 = EXPORTS / "crack_0001" / "summary.json"
    if sum1.exists() and sum3.exists():
        b = load_summary(sum1)
        c = load_summary(sum3)
        delta = compare_summaries(b, c)
        for k, v in delta.items():
            print(f"  {k}: {v:+.4f}")
    else:
        print("  SKIP: summaries not found")

    # Scenario 5: Analyze a few more normals (batch check)
    print("\n[Scenario 5] Batch analyze normal_0050, normal_0100")
    for vid in ["normal_0050", "normal_0100"]:
        inp = BASE / "normal" / vid
        oup = EXPORTS / vid
        if inp.exists():
            s = run_analysis(input_dir=inp, output_dir=oup, fps=30.0)
            print(f"  {vid}: mean_speed={s.mean_speed:.2f} px/s, frames={s.frame_count}")
    print("\nDone.")


if __name__ == "__main__":
    main()
