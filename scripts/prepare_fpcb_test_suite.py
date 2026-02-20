"""
Prepare FPCB bending side-view test suite for GUI analysis scenarios.

Generates:
- 200 normal bending datasets (varied within normal process range)
- 10 abnormal (crack) datasets (various crack types with random variation)

Each dataset = 2 seconds at 30 fps = 60 frames, organized in per-video folders.
Realistic variability: seed, points_per_frame, noise, panel geometry, crack subtypes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle

FRAMES = 60
FPS = 30.0
NORMAL_COUNT = 200
CRACK_COUNT = 10

# Normal range bounds (realistic process variation)
NORMAL_POINTS_RANGE = (200, 260)
NORMAL_NOISE_RANGE = (0.12, 0.38)
NORMAL_PANEL_LENGTH_RANGE = (210, 250)
NORMAL_PANEL_THICKNESS_RANGE = (85, 95)
NORMAL_PIXELS_PER_MM_RANGE = (8.0, 12.0)

# Crack subtypes: (scenario, description) for diversity
CRACK_SUBTYPES = [
    ("crack", "full_crack"),           # 6x: severe crack with shockwave
    ("crack", "full_crack"),
    ("crack", "full_crack"),
    ("crack", "full_crack"),
    ("crack", "full_crack"),
    ("crack", "full_crack"),
    ("pre_damage", "mild_crack"),      # 2x: pre-damage / mild crack
    ("pre_damage", "mild_crack"),
    ("uv_overcured", "snap_crack"),    # 2x: UV over-cure snap behavior
    ("uv_overcured", "snap_crack"),
]


def _rand(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def main() -> None:
    base_dir = repo_root / "data" / "synthetic" / "fpcb_test_suite"
    normal_dir = base_dir / "normal"
    crack_dir = base_dir / "crack"

    rng = np.random.default_rng(seed=20260219)

    print("=" * 60)
    print("FPCB Bending Test Suite Generator")
    print("=" * 60)
    print(f"Output: {base_dir}")
    print(f"  - {NORMAL_COUNT} normal (2 s, 60 frames @ 30 fps)")
    print(f"  - {CRACK_COUNT} crack (various types)")
    print()

    # --- Normal datasets ---
    normal_dir.mkdir(parents=True, exist_ok=True)
    print("Generating normal bending datasets...")
    for i in range(1, NORMAL_COUNT + 1):
        vid_name = f"normal_{i:03d}"
        out = normal_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        seed = 10000 + i
        points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
        noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
        panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
        thickness = _rand(rng, NORMAL_PANEL_THICKNESS_RANGE[0], NORMAL_PANEL_THICKNESS_RANGE[1])
        px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])

        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=points,
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=panel_len,
            panel_thickness_um=thickness,
            pixels_per_mm=px_per_mm,
            meters_per_pixel=1e-3 / px_per_mm,
            noise_std=noise,
            seed=seed,
            scenario="normal",
        )
        generate_synthetic_bundle(out, config)
        if i % 50 == 0 or i == NORMAL_COUNT:
            print(f"  [OK] {i}/{NORMAL_COUNT}: {vid_name}")

    print()

    # --- Crack datasets ---
    crack_dir.mkdir(parents=True, exist_ok=True)
    print("Generating crack bending datasets...")
    for i in range(CRACK_COUNT):
        scenario, subtype = CRACK_SUBTYPES[i]
        vid_name = f"crack_{i+1:02d}_{subtype}"
        out = crack_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        seed = 20000 + i * 137  # varied seeds for different crack realizations
        points = _randint(rng, 210, 250)
        noise = _rand(rng, 0.15, 0.35)
        panel_len = _rand(rng, 215, 245)

        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=points,
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=panel_len,
            panel_thickness_um=90.0,
            pixels_per_mm=10.0,
            meters_per_pixel=1e-4,
            noise_std=noise,
            seed=seed,
            scenario=scenario,
        )
        generate_synthetic_bundle(out, config)
        print(f"  [OK] {i+1}/{CRACK_COUNT}: {vid_name} ({scenario})")

    print()
    print("Test suite ready.")
    print()
    print("Folder structure:")
    print(f"  {normal_dir}/")
    print(f"    normal_001/ ... normal_{NORMAL_COUNT:03d}/  (frame_00000.txt .. frame_00059.txt, fps.txt, etc.)")
    print(f"  {crack_dir}/")
    print(f"    crack_01_* ... crack_10_*/")
    print()
    print("GUI test scenario:")
    print("  1. Analyze Tab: input=data/synthetic/fpcb_test_suite/normal/normal_001")
    print("  2. Scale (mm/px): 0.1  (or leave empty to use metadata)")
    print("  3. FPS: 30")
    print("  4. Run Analysis")
    print("  5. Compare Tab: base=normal_001 summary, candidate=crack_01 summary")
    print()


if __name__ == "__main__":
    main()
