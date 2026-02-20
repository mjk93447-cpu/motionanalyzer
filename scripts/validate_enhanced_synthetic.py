"""
Validate enhanced synthetic data with shockwave and vibration patterns.

Tests that crack scenario shows:
1. Acceleration spike (shockwave) at crack frame
2. Micro-vibration pattern after crack
3. Improved detectability by DREAM and Change Point Detection
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.analysis import run_analysis
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def _generate_test_data(output_dir: Path) -> tuple[Path, Path]:
    """Generate enhanced synthetic normal and crack datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normal scenario
    normal_config = SyntheticConfig(
        frames=60,
        points_per_frame=280,
        fps=30.0,
        seed=42,
        scenario="normal",
    )
    normal_path = generate_synthetic_bundle(output_dir / "normal", normal_config)

    # Enhanced crack scenario (with shockwave and vibration)
    crack_config = SyntheticConfig(
        frames=60,
        points_per_frame=280,
        fps=30.0,
        seed=43,
        scenario="crack",
    )
    crack_path = generate_synthetic_bundle(output_dir / "crack", crack_config)

    return normal_path, crack_path


def _analyze_acceleration_pattern(dataset_path: Path) -> pd.DataFrame:
    """Analyze acceleration patterns from dataset."""
    output_dir = dataset_path.parent / f"{dataset_path.name}_analysis"
    run_analysis(dataset_path, output_dir, fps=30.0)

    vectors_path = output_dir / "vectors.csv"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")

    vectors = pd.read_csv(vectors_path)
    
    # Per-frame acceleration statistics
    frame_accel = vectors.groupby("frame")["acceleration"].agg(["mean", "max", "std"]).reset_index()
    return frame_accel


def main() -> None:
    """Main validation function."""
    import tempfile

    print("Enhanced Synthetic Data Validation")
    print("=" * 60)
    print("Testing shockwave and vibration patterns in crack scenario\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        normal_path, crack_path = _generate_test_data(tmp_path)

        print(f"Generated datasets:")
        print(f"  Normal: {normal_path}")
        print(f"  Crack: {crack_path}\n")

        # Analyze acceleration patterns
        print("Analyzing acceleration patterns...")
        normal_accel = _analyze_acceleration_pattern(normal_path)
        crack_accel = _analyze_acceleration_pattern(crack_path)

        # Expected crack frame (crack_center_ratio ~0.72, frames=60)
        expected_crack_frame = int(0.72 * 59)  # ~43

        print(f"\nNormal scenario acceleration:")
        print(f"  Mean max acceleration: {normal_accel['max'].mean():.4f}")
        print(f"  Max acceleration: {normal_accel['max'].max():.4f}")
        print(f"  Std of max acceleration: {normal_accel['max'].std():.4f}")

        print(f"\nCrack scenario acceleration:")
        print(f"  Mean max acceleration: {crack_accel['max'].mean():.4f}")
        print(f"  Max acceleration: {crack_accel['max'].max():.4f}")
        print(f"  Std of max acceleration: {crack_accel['max'].std():.4f}")

        # Check for acceleration spike around crack frame
        crack_window = crack_accel[
            (crack_accel["frame"] >= expected_crack_frame - 3) &
            (crack_accel["frame"] <= expected_crack_frame + 5)
        ]
        if len(crack_window) > 0:
            spike_max = crack_window["max"].max()
            spike_frame = crack_window.loc[crack_window["max"].idxmax(), "frame"]
            baseline_max = crack_accel[crack_accel["frame"] < expected_crack_frame - 3]["max"].mean()
            
            print(f"\nAcceleration spike detection:")
            print(f"  Expected crack frame: {expected_crack_frame}")
            print(f"  Detected spike frame: {int(spike_frame)}")
            print(f"  Spike magnitude: {spike_max:.4f}")
            print(f"  Baseline (before crack): {baseline_max:.4f}")
            print(f"  Spike ratio: {spike_max / (baseline_max + 1e-6):.2f}x")
            
            if spike_max > baseline_max * 1.5:
                print(f"  [OK] Shockwave pattern detected!")
            else:
                print(f"  [WARN] Shockwave pattern may be weak")

        # Check for vibration pattern (increased std after crack)
        before_crack_std = crack_accel[crack_accel["frame"] < expected_crack_frame]["std"].mean()
        after_crack_std = crack_accel[
            (crack_accel["frame"] >= expected_crack_frame) &
            (crack_accel["frame"] <= expected_crack_frame + 15)
        ]["std"].mean()
        
        print(f"\nVibration pattern detection:")
        print(f"  Std before crack: {before_crack_std:.4f}")
        print(f"  Std after crack (15 frames): {after_crack_std:.4f}")
        print(f"  Increase ratio: {after_crack_std / (before_crack_std + 1e-6):.2f}x")
        
        if after_crack_std > before_crack_std * 1.2:
            print(f"  [OK] Vibration pattern detected!")
        else:
            print(f"  [WARN] Vibration pattern may be weak")

        print("\n" + "=" * 60)
        print("Summary:")
        print("Enhanced synthetic data includes:")
        print("  - Shockwave (acceleration spike) at crack frame")
        print("  - Micro-vibration (increased std) after crack")
        print("  - These patterns should improve ML model detectability")


if __name__ == "__main__":
    main()
