"""
Generate FPCB bending side-view synthetic dataset for user scenario testing.

Creates:
- 200 normal bending process bundles (2 sec @ 30 fps = 60 frames each)
- 10 crack (abnormal) bending process bundles (2 sec @ 30 fps = 60 frames each)

Each bundle = one "video" extracted as frame_*.txt, fps.txt, frame_metrics.csv, metadata.json.
Realistic variation: seeds, noise, points_per_frame, panel_length, pixels_per_mm.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def main() -> None:
    base_dir = repo_root / "data" / "synthetic" / "fpcb_test_scenario"
    normal_dir = base_dir / "normal"
    crack_dir = base_dir / "crack"

    frames_per_video = 60
    fps = 30.0

    # Realistic parameter ranges for variation
    points_range = (200, 260)
    panel_length_range = (210, 250)
    pixels_per_mm_range = (8.0, 12.0)
    noise_range = (0.3, 1.2)

    def make_config(scenario: str, video_idx: int, seed: int) -> SyntheticConfig:
        import random
        r = random.Random(seed)
        points = r.randint(*points_range)
        panel_len = r.uniform(*panel_length_range)
        px_per_mm = r.uniform(*pixels_per_mm_range)
        noise = r.uniform(*noise_range)
        return SyntheticConfig(
            frames=frames_per_video,
            points_per_frame=points,
            fps=fps,
            panel_length_px=panel_len,
            pixels_per_mm=px_per_mm,
            noise_std=noise,
            seed=seed,
            scenario=scenario,
        )

    print("Generating FPCB bending test dataset for user scenario...")
    print(f"  Normal: 200 bundles x {frames_per_video} frames @ {fps} fps (2 sec each)")
    print(f"  Crack:  10 bundles x {frames_per_video} frames @ {fps} fps (2 sec each)")
    print(f"  Output: {base_dir}")
    print()

    # Normal (200)
    normal_dir.mkdir(parents=True, exist_ok=True)
    for i in range(200):
        out = normal_dir / f"video_{i:04d}"
        cfg = make_config("normal", i, seed=10000 + i)
        generate_synthetic_bundle(out, cfg)
        if (i + 1) % 50 == 0:
            print(f"  [normal] {i + 1}/200 done")

    print("  [normal] 200/200 done")
    print()

    # Crack (10)
    crack_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        out = crack_dir / f"video_{i:04d}"
        cfg = make_config("crack", i, seed=50000 + i)
        generate_synthetic_bundle(out, cfg)
        print(f"  [crack] {i + 1}/10 done")

    print()
    print("Dataset generation complete!")
    print(f"  Normal: {normal_dir} ({len(list(normal_dir.iterdir()))} folders)")
    print(f"  Crack:  {crack_dir} ({len(list(crack_dir.iterdir()))} folders)")
    print()
    print("Usage:")
    print("  1. Analyze single normal: input = data/synthetic/fpcb_test_scenario/normal/video_0000")
    print("  2. Analyze single crack:  input = data/synthetic/fpcb_test_scenario/crack/video_0000")
    print("  3. ML tab: add normal folders for training, crack for validation")


if __name__ == "__main__":
    main()
