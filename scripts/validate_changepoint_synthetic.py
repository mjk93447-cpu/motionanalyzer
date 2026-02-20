"""
Validate change point detection on synthetic FPCB bending data.

Tests CUSUM, Window-based, and PELT detectors on synthetic normal and crack scenarios
to verify that crack occurrence frames are accurately detected.
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
from motionanalyzer.time_series.changepoint import (
    CUSUMDetector,
    WindowBasedDetector,
    detect_change_points_pelt,
)


def _generate_test_data(output_dir: Path) -> tuple[Path, Path]:
    """Generate synthetic normal and crack datasets."""
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

    # Crack scenario (crack occurs around frame 30-40 based on crack_center_ratio)
    crack_config = SyntheticConfig(
        frames=60,
        points_per_frame=280,
        fps=30.0,
        seed=43,
        scenario="crack",
    )
    crack_path = generate_synthetic_bundle(output_dir / "crack", crack_config)

    return normal_path, crack_path


def _extract_time_series_features(dataset_path: Path) -> pd.DataFrame:
    """Extract time series features from dataset."""
    # Run analysis to get vectors
    output_dir = dataset_path.parent / f"{dataset_path.name}_analysis"
    run_analysis(dataset_path, output_dir, fps=30.0)

    # Load vectors
    vectors_path = output_dir / "vectors.csv"
    if not vectors_path.exists():
        raise FileNotFoundError(f"Vectors file not found: {vectors_path}")

    vectors = pd.read_csv(vectors_path)

    # Aggregate per frame
    frame_features = vectors.groupby("frame").agg({
        "acceleration": ["mean", "max", "std"],
        "curvature_like": ["mean", "max", "std"],
        "strain_surrogate": ["mean", "max", "std"],
        "impact_surrogate": ["mean", "max"],
    }).reset_index()

    frame_features.columns = ["frame"] + [f"{col[0]}_{col[1]}" for col in frame_features.columns[1:]]

    return frame_features


def _test_detector_on_signal(
    signal: np.ndarray,
    detector_name: str,
    detector_func,
    expected_range: tuple[int, int] | None = None,
) -> dict[str, any]:
    """Test a detector on a signal and return results."""
    try:
        result = detector_func(signal)
        change_points = result.change_points

        # Check if change point is in expected range
        detected = False
        if expected_range and change_points:
            detected = any(expected_range[0] <= cp <= expected_range[1] for cp in change_points)

        return {
            "detector": detector_name,
            "change_points": change_points,
            "num_detections": len(change_points),
            "detected_in_range": detected,
            "success": detected if expected_range else len(change_points) > 0,
        }
    except Exception as e:
        return {
            "detector": detector_name,
            "change_points": [],
            "num_detections": 0,
            "detected_in_range": False,
            "success": False,
            "error": str(e),
        }


def main() -> None:
    """Main validation function."""
    import tempfile

    print("Change Point Detection Validation on Synthetic FPCB Data")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        normal_path, crack_path = _generate_test_data(tmp_path)

        print(f"\nGenerated datasets:")
        print(f"  Normal: {normal_path}")
        print(f"  Crack: {crack_path}")

        # Extract features
        print("\nExtracting time series features...")
        normal_features = _extract_time_series_features(normal_path)
        crack_features = _extract_time_series_features(crack_path)

        print(f"  Normal: {len(normal_features)} frames")
        print(f"  Crack: {len(crack_features)} frames")

        # Test on acceleration_max (should spike when crack occurs)
        print("\n" + "=" * 60)
        print("Testing on acceleration_max (expected crack around frame 30-40)")
        print("=" * 60)

        normal_signal = normal_features["acceleration_max"].values
        crack_signal = crack_features["acceleration_max"].values

        # Expected crack frame range (based on crack_center_ratio ~0.5, frames=60)
        expected_crack_range = (25, 45)

        results = []

        # CUSUM detector
        cusum_detector = CUSUMDetector(threshold=2.0, min_size=5)
        results.append(_test_detector_on_signal(
            crack_signal,
            "CUSUM",
            lambda s: cusum_detector.detect(s),
            expected_crack_range,
        ))

        # Window-based detector
        window_detector = WindowBasedDetector(window_size=10, threshold_ratio=1.5, min_size=5)
        results.append(_test_detector_on_signal(
            crack_signal,
            "WindowBased",
            lambda s: window_detector.detect(s),
            expected_crack_range,
        ))

        # PELT detector (if available)
        try:
            results.append(_test_detector_on_signal(
                crack_signal,
                "PELT",
                lambda s: detect_change_points_pelt(s, min_size=5, pen=3.0),
                expected_crack_range,
            ))
        except ImportError:
            print("  PELT: ruptures library not installed (skipped)")

        # Print results
        print("\nResults:")
        for r in results:
            print(f"\n  {r['detector']}:")
            print(f"    Change points: {r['change_points']}")
            print(f"    Number of detections: {r['num_detections']}")
            if "expected_range" in locals():
                print(f"    Detected in expected range ({expected_crack_range[0]}-{expected_crack_range[1]}): {r['detected_in_range']}")
            print(f"    Success: {r['success']}")
            if "error" in r:
                print(f"    Error: {r['error']}")

        # Test on normal signal (should have few or no change points)
        print("\n" + "=" * 60)
        print("Testing on normal signal (should have few/no change points)")
        print("=" * 60)

        normal_results = []
        normal_results.append(_test_detector_on_signal(
            normal_signal,
            "CUSUM",
            lambda s: cusum_detector.detect(s),
            None,  # No expected range for normal
        ))
        normal_results.append(_test_detector_on_signal(
            normal_signal,
            "WindowBased",
            lambda s: window_detector.detect(s),
            None,
        ))

        print("\nNormal signal results:")
        for r in normal_results:
            print(f"  {r['detector']}: {r['num_detections']} change points detected")

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        crack_detections = [r for r in results if r.get("detected_in_range", False)]
        print(f"Crack detection success: {len(crack_detections)}/{len(results)} detectors")
        if crack_detections:
            print("Successful detectors:", [r["detector"] for r in crack_detections])


if __name__ == "__main__":
    main()
