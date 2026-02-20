"""
Generate ML training/evaluation synthetic dataset.

Scale: 1000 normal, 80 crack_in_bending (Goal 1), 20 pre_damaged (Goal 2).
Tags: goal, label, scenario, crack_frame, split.
Output: data/synthetic/ml_dataset/ + manifest.json.

Usage:
  python scripts/generate_ml_dataset.py
  python scripts/generate_ml_dataset.py --dry-run  # print plan only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle

FRAMES = 60
FPS = 30.0
NORMAL_COUNT = 1000
LIGHT_DISTORTION_COUNT = 50   # 5% of normal (Phase 1.1): illumination-induced edge distortion (label=0)
CRACK_IN_BENDING_COUNT = 80   # Goal 1: 50 crack + 30 uv_overcured
MICRO_CRACK_COUNT = 10       # Subtle crack, harder to detect (label=1)
PRE_DAMAGED_COUNT = 20       # Goal 2
THICK_PANEL_COUNT = 20       # Variant scenario (boundary, label=0)
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

NORMAL_POINTS_RANGE = (200, 260)
NORMAL_NOISE_RANGE = (0.12, 0.38)
NORMAL_PANEL_LENGTH_RANGE = (210, 250)
NORMAL_PIXELS_PER_MM_RANGE = (8.0, 12.0)


def _rand(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _assign_split(rng: np.random.Generator, n: int) -> list[str]:
    train_n = int(n * SPLIT_RATIOS[0])
    val_n = int(n * SPLIT_RATIOS[1])
    test_n = n - train_n - val_n
    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
    rng.shuffle(splits)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML synthetic dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not generate")
    parser.add_argument("--small", action="store_true", help="Small set: 100 normal, 10 crack, 5 predam (quick test)")
    parser.add_argument("--seed", type=int, default=20260219)
    args = parser.parse_args()

    n_normal = 100 if args.small else NORMAL_COUNT
    n_light_dist = 3 if args.small else LIGHT_DISTORTION_COUNT
    n_crack = 10 if args.small else CRACK_IN_BENDING_COUNT
    n_micro_crack = 2 if args.small else MICRO_CRACK_COUNT
    n_predam = 5 if args.small else PRE_DAMAGED_COUNT
    n_thick = 5 if args.small else THICK_PANEL_COUNT

    base_dir = repo_root / "data" / "synthetic" / "ml_dataset"
    rng = np.random.default_rng(seed=args.seed)

    print("=" * 60)
    print("ML Dataset Generator")
    print("=" * 60)
    print(f"Output: {base_dir}")
    print(f"  - {n_normal} normal")
    print(f"  - {n_light_dist} normal_light_distortion (1-2% harder normal)")
    print(f"  - {n_crack} crack_in_bending (Goal 1)")
    print(f"  - {n_micro_crack} micro_crack (harder crack)")
    print(f"  - {n_predam} pre_damaged (Goal 2)")
    print(f"  - {n_thick} thick_panel (variant, boundary)")
    print()

    if args.dry_run:
        print("[DRY RUN] Would create:")
        print(f"  {base_dir}/normal/normal_0001..normal_{n_normal:04d}")
        print(f"  {base_dir}/normal/normal_ld_0001..normal_ld_{n_light_dist:04d}")
        print(f"  {base_dir}/crack_in_bending/crack_0001..crack_{n_crack:04d}")
        print(f"  {base_dir}/crack_in_bending/micro_0001..micro_{n_micro_crack:04d}")
        print(f"  {base_dir}/pre_damaged/predam_0001..predam_{n_predam:04d}")
        print(f"  {base_dir}/thick_panel/thick_0001..thick_{n_thick:04d}")
        print(f"  {base_dir}/manifest.json")
        return

    manifest_entries: list[dict] = []
    normal_splits = _assign_split(rng, n_normal)
    light_dist_splits = _assign_split(rng, n_light_dist)
    crack_splits = _assign_split(rng, n_crack)
    micro_crack_splits = _assign_split(rng, n_micro_crack)
    predam_splits = _assign_split(rng, n_predam)
    thick_splits = _assign_split(rng, n_thick)

    # --- Normal ---
    normal_dir = base_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    print("Generating normal...")
    for i in range(1, n_normal + 1):
        vid_name = f"normal_{i:04d}"
        out = normal_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        seed = 100000 + i
        points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
        noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
        panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
        px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])

        extra = {
            "goal": "normal",
            "scenario": "normal",
            "label": 0,
            "crack_frame": -1,
            "split": normal_splits[i - 1],
            "dataset_id": vid_name,
        }
        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=points,
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=panel_len,
            panel_thickness_um=90.0,
            pixels_per_mm=px_per_mm,
            meters_per_pixel=1e-3 / px_per_mm,
            noise_std=noise,
            seed=seed,
            scenario="normal",
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"normal/{vid_name}",
            "goal": "normal",
            "label": 0,
            "split": extra["split"],
        })
        if i % 200 == 0 or i == n_normal:
            print(f"  [OK] {i}/{n_normal}")

    # --- Normal + light distortion (1-2% harder normal, label=0) ---
    print("\nGenerating normal_light_distortion...")
    for i in range(1, n_light_dist + 1):
        vid_name = f"normal_ld_{i:04d}"
        out = normal_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        seed = 150000 + i
        points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
        noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
        panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
        px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])

        extra = {
            "goal": "normal",
            "scenario": "light_distortion",
            "label": 0,
            "crack_frame": -1,
            "split": light_dist_splits[i - 1],
            "dataset_id": vid_name,
        }
        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=points,
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=panel_len,
            panel_thickness_um=90.0,
            pixels_per_mm=px_per_mm,
            meters_per_pixel=1e-3 / px_per_mm,
            noise_std=noise,
            seed=seed,
            scenario="light_distortion",
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"normal/{vid_name}",
            "goal": "normal",
            "label": 0,
            "split": extra["split"],
            "scenario": "light_distortion",
        })
        print(f"  [OK] {i}/{n_light_dist}")

    # --- Crack in bending (Goal 1): 50 crack + 30 uv_overcured ---
    crack_dir = base_dir / "crack_in_bending"
    crack_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating crack_in_bending (Goal 1)...")
    n_crack_scenario = min(50, n_crack)  # crack count; rest uv_overcured
    for i in range(1, n_crack + 1):
        use_crack = i <= n_crack_scenario
        scenario = "crack" if use_crack else "uv_overcured"
        vid_name = f"crack_{i:04d}"
        out = crack_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        crack_center = _rand(rng, 0.65, 0.80)
        crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1

        extra = {
            "goal": "goal1",
            "scenario": scenario,
            "label": 1,
            "crack_frame": crack_frame,
            "split": crack_splits[i - 1],
            "dataset_id": vid_name,
        }
        seed = 200000 + i * 137
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
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"crack_in_bending/{vid_name}",
            "goal": "goal1",
            "label": 1,
            "split": extra["split"],
            "crack_frame": crack_frame,
        })
        if i % 20 == 0 or i == n_crack:
            print(f"  [OK] {i}/{n_crack} ({scenario})")

    # --- Micro crack (harder crack, label=1) ---
    print("\nGenerating micro_crack...")
    for i in range(1, n_micro_crack + 1):
        vid_name = f"micro_{i:04d}"
        out = crack_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        crack_center = _rand(rng, 0.65, 0.80)
        crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1

        extra = {
            "goal": "goal1",
            "scenario": "micro_crack",
            "label": 1,
            "crack_frame": crack_frame,
            "split": micro_crack_splits[i - 1],
            "dataset_id": vid_name,
        }
        seed = 250000 + i * 211
        points = _randint(rng, 210, 250)
        noise = _rand(rng, 0.18, 0.38)  # Slightly higher noise for harder detection

        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=points,
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=_rand(rng, 215, 245),
            panel_thickness_um=90.0,
            pixels_per_mm=10.0,
            meters_per_pixel=1e-4,
            noise_std=noise,
            seed=seed,
            scenario="micro_crack",
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"crack_in_bending/{vid_name}",
            "goal": "goal1",
            "label": 1,
            "split": extra["split"],
            "crack_frame": crack_frame,
            "scenario": "micro_crack",
        })
        print(f"  [OK] {i}/{n_micro_crack}")

    # --- Pre-damaged (Goal 2) ---
    predam_dir = base_dir / "pre_damaged"
    predam_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating pre_damaged (Goal 2)...")
    for i in range(1, n_predam + 1):
        vid_name = f"predam_{i:04d}"
        out = predam_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        extra = {
            "goal": "goal2",
            "scenario": "pre_damage",
            "label": 1,
            "crack_frame": -1,
            "split": predam_splits[i - 1],
            "dataset_id": vid_name,
        }
        seed = 300000 + i * 97
        points = _randint(rng, 210, 250)
        noise = _rand(rng, 0.18, 0.35)
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
            scenario="pre_damage",
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"pre_damaged/{vid_name}",
            "goal": "goal2",
            "label": 1,
            "split": extra["split"],
        })
        print(f"  [OK] {i}/{n_predam}")

    # --- Thick panel (variant, boundary case, label=0) ---
    thick_dir = base_dir / "thick_panel"
    thick_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating thick_panel (variant)...")
    for i in range(1, n_thick + 1):
        vid_name = f"thick_{i:04d}"
        out = thick_dir / vid_name
        out.mkdir(parents=True, exist_ok=True)

        extra = {
            "goal": "variant",
            "scenario": "thick_panel",
            "label": 0,
            "crack_frame": -1,
            "split": thick_splits[i - 1],
            "dataset_id": vid_name,
        }
        seed = 400000 + i * 73
        points = _randint(rng, 210, 250)
        noise = _rand(rng, 0.12, 0.30)
        panel_len = _rand(rng, 220, 248)

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
            scenario="thick_panel",
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"thick_panel/{vid_name}",
            "goal": "variant",
            "label": 0,
            "split": extra["split"],
        })
        if i % 5 == 0 or i == n_thick:
            print(f"  [OK] {i}/{n_thick}")

    # --- Manifest ---
    train_count = sum(1 for e in manifest_entries if e["split"] == "train")
    val_count = sum(1 for e in manifest_entries if e["split"] == "val")
    test_count = sum(1 for e in manifest_entries if e["split"] == "test")
    manifest = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "total_count": len(manifest_entries),
        "normal": n_normal,
        "normal_light_distortion": n_light_dist,
        "crack_in_bending": n_crack,
        "micro_crack": n_micro_crack,
        "pre_damaged_panel": n_predam,
        "thick_panel": n_thick,
        "splits": {"train": train_count, "val": val_count, "test": test_count},
        "entries": manifest_entries,
    }
    (base_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print()
    print("Manifest written.")
    print(f"  train={train_count}, val={val_count}, test={test_count}")
    print()
    print("Next: see docs/ML_SYNTHETIC_DATA_GUIDE.md for ML training usage.")
    print()


if __name__ == "__main__":
    main()
