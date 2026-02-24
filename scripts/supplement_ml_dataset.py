"""
Supplement existing ML dataset to reach 30k scale.

Keeps existing 10k data intact, generates additional ~20k samples with offset indices.
Output: appends to data/synthetic/ml_dataset/ + merges manifest.json.

Usage:
  python scripts/supplement_ml_dataset.py
  python scripts/supplement_ml_dataset.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
from motionanalyzer.synthetic import NoiseMode

FRAMES = 60
FPS = 30.0
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# 10k config for reference
CONFIG_10K = (7000, 500, 500, 300, 300, 500, 400)  # ~9.5k
# 30k = 3x 10k proportions
CONFIG_30K = (21000, 1500, 1500, 900, 900, 1500, 1200)  # ~28.5k

NOISE_MODES: tuple[NoiseMode, ...] = ("gaussian", "outlier", "temporal_drift", "scale_jitter", "mixed")
NORMAL_POINTS_RANGE = (200, 260)
NORMAL_NOISE_RANGE = (0.12, 0.38)
NORMAL_PANEL_LENGTH_RANGE = (210, 250)
NORMAL_PIXELS_PER_MM_RANGE = (8.0, 12.0)

# Seed offset for supplement to avoid collision with 10k
SEED_OFFSET = 500000


def _rand(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _noise_mode_for_seed(seed: int) -> NoiseMode:
    return NOISE_MODES[seed % len(NOISE_MODES)]


def _assign_split(rng: np.random.Generator, n: int) -> list[str]:
    train_n = int(n * SPLIT_RATIOS[0])
    val_n = int(n * SPLIT_RATIOS[1])
    test_n = n - train_n - val_n
    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
    rng.shuffle(splits)
    return splits


def _generate_one_sample(args: tuple[Any, ...]) -> dict[str, Any]:
    out_dir, config_dict, extra_dict = args
    from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
    fields = {f for f in SyntheticConfig.__dataclass_fields__ if f in config_dict}
    config = SyntheticConfig(**{k: config_dict[k] for k in fields})
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generate_synthetic_bundle(out_path, config, extra_metadata=dict(extra_dict))
    return extra_dict


def _count_existing(entries: list[dict]) -> dict[str, int]:
    """Count existing per category. crack = crack+uv (excl micro)."""
    counts = {
        "normal": 0,
        "light_distortion": 0,
        "crack": 0,  # crack + uv_overcured
        "micro_crack": 0,
        "pre_damaged": 0,
        "thick_panel": 0,
    }
    for e in entries:
        path = e.get("path", "")
        if path.startswith("normal/normal_ld_"):
            counts["light_distortion"] += 1
        elif path.startswith("normal/normal_"):
            counts["normal"] += 1
        elif path.startswith("crack_in_bending/micro_"):
            counts["micro_crack"] += 1
        elif path.startswith("crack_in_bending/crack_"):
            counts["crack"] += 1
        elif path.startswith("pre_damaged/"):
            counts["pre_damaged"] += 1
        elif path.startswith("thick_panel/"):
            counts["thick_panel"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement ML dataset to 30k")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260224)
    args = parser.parse_args()

    base_dir = repo_root / "data" / "synthetic" / "ml_dataset"
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print("Error: manifest.json not found. Run generate_ml_dataset.py --scale 10k first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    existing = _count_existing(entries)

    # Delta to reach 30k (target = 3x 10k proportions)
    target = {
        "normal": CONFIG_30K[0],
        "light_distortion": CONFIG_30K[1],
        "crack": CONFIG_30K[2] + CONFIG_30K[3],  # crack + uv
        "micro_crack": CONFIG_30K[4],
        "pre_damaged": CONFIG_30K[5],
        "thick_panel": CONFIG_30K[6],
    }
    delta = {k: max(0, target[k] - existing.get(k, 0)) for k in target}
    # For crack: first CONFIG_30K[2] of total are crack scenario, rest uv
    n_crack_scenario = CONFIG_30K[2]
    n_crack_total_target = CONFIG_30K[2] + CONFIG_30K[3]
    delta_crack = delta["crack"]
    # Of new crack samples: first min(delta, 700) are crack (to reach 1500 from 800), rest uv
    existing_crack_plus_uv = existing.get("crack", 0)
    # existing crack = crack+uv (excl micro). 10k has 500 crack + 300 uv = 800.
    need_more_crack = max(0, n_crack_scenario - min(existing_crack_plus_uv, 500))
    delta_crack_main = min(delta_crack, need_more_crack) if need_more_crack else 0

    total_delta = sum(delta.values())
    if total_delta == 0:
        print("Dataset already at 30k. Nothing to supplement.")
        return

    print("=" * 60)
    print("ML Dataset Supplement (10k -> 30k)")
    print("=" * 60)
    print(f"Existing: {sum(existing.values())} total")
    print(f"Target:   {sum(target.values())} total")
    print(f"Delta:    {total_delta} to generate")
    print(f"  - normal: +{delta['normal']}")
    print(f"  - light_distortion: +{delta['light_distortion']}")
    print(f"  - crack+uv: +{delta['crack']}")
    print(f"  - micro_crack: +{delta['micro_crack']}")
    print(f"  - pre_damaged: +{delta['pre_damaged']}")
    print(f"  - thick_panel: +{delta['thick_panel']}")
    print()

    rng = np.random.default_rng(seed=args.seed)
    manifest_entries = list(entries)
    normal_dir = base_dir / "normal"
    crack_dir = base_dir / "crack_in_bending"
    predam_dir = base_dir / "pre_damaged"
    thick_dir = base_dir / "thick_panel"

    # Offsets (1-based next index; use 5 digits for supplement to avoid collision)
    off_n = existing["normal"] + 1
    off_ld = existing["light_distortion"] + 1
    off_c = existing["crack"] + 1  # crack_0001..crack_0800 exist, next is crack_0801
    off_m = existing["micro_crack"] + 1
    off_p = existing["pre_damaged"] + 1
    off_t = existing["thick_panel"] + 1

    # --- Normal ---
    if delta["normal"] > 0:
        normal_splits = _assign_split(rng, delta["normal"])
        print(f"Generating normal (+{delta['normal']})...")
        for i in range(delta["normal"]):
            vid_name = f"normal_{off_n + i:05d}"  # 5 digits for supplement (07001..21000)
            out = normal_dir / vid_name
            seed = SEED_OFFSET + off_n + i
            points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
            noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
            panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
            px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])
            noise_mode = _noise_mode_for_seed(seed)
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=points, fps=FPS, width=1920, height=1080,
                panel_length_px=panel_len, panel_thickness_um=90.0, pixels_per_mm=px_per_mm,
                meters_per_pixel=1e-3 / px_per_mm, noise_std=noise, seed=seed,
                scenario="normal", noise_mode=noise_mode,
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "normal", "scenario": "normal", "label": 0, "crack_frame": -1,
                "split": normal_splits[i], "dataset_id": vid_name, "noise_mode": noise_mode,
            })
            manifest_entries.append({"path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": normal_splits[i]})
            if (i + 1) % 500 == 0 or i == delta["normal"] - 1:
                print(f"  [OK] {i + 1}/{delta['normal']}")

    # --- Light distortion ---
    if delta["light_distortion"] > 0:
        ld_splits = _assign_split(rng, delta["light_distortion"])
        print(f"\nGenerating light_distortion (+{delta['light_distortion']})...")
        for i in range(delta["light_distortion"]):
            vid_name = f"normal_ld_{off_ld + i:05d}"
            out = normal_dir / vid_name
            seed = SEED_OFFSET + 100000 + off_ld + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS, width=1920, height=1080,
                panel_length_px=_rand(rng, 215, 245), panel_thickness_um=90.0, pixels_per_mm=10.0,
                meters_per_pixel=1e-4, noise_std=_rand(rng, 0.15, 0.35), seed=seed,
                scenario="light_distortion", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "normal", "scenario": "light_distortion", "label": 0, "crack_frame": -1,
                "split": ld_splits[i], "dataset_id": vid_name,
            })
            manifest_entries.append({
                "path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": ld_splits[i],
                "scenario": "light_distortion",
            })
            if (i + 1) % 100 == 0 or i == delta["light_distortion"] - 1:
                print(f"  [OK] {i + 1}/{delta['light_distortion']}")

    # --- Crack + UV ---
    if delta["crack"] > 0:
        crack_splits = _assign_split(rng, delta["crack"])
        n_crack_scenario = delta_crack_main  # first N are crack, rest uv
        print(f"\nGenerating crack_in_bending (+{delta['crack']})...")
        for i in range(delta["crack"]):
            use_crack = i < n_crack_scenario
            scenario = "crack" if use_crack else "uv_overcured"
            vid_name = f"crack_{off_c + i:05d}"
            out = crack_dir / vid_name
            crack_center = _rand(rng, 0.65, 0.80)
            crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1
            seed = SEED_OFFSET + 200000 + off_c + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS, width=1920, height=1080,
                panel_length_px=_rand(rng, 215, 245), panel_thickness_um=90.0, pixels_per_mm=10.0,
                meters_per_pixel=1e-4, noise_std=_rand(rng, 0.15, 0.35), seed=seed,
                scenario=scenario, noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "goal1", "scenario": scenario, "label": 1, "crack_frame": crack_frame,
                "split": crack_splits[i], "dataset_id": vid_name,
            })
            manifest_entries.append({
                "path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1,
                "split": crack_splits[i], "crack_frame": crack_frame,
            })
            if (i + 1) % 200 == 0 or i == delta["crack"] - 1:
                print(f"  [OK] {i + 1}/{delta['crack']} ({scenario})")

    # --- Micro crack ---
    if delta["micro_crack"] > 0:
        mc_splits = _assign_split(rng, delta["micro_crack"])
        print(f"\nGenerating micro_crack (+{delta['micro_crack']})...")
        for i in range(delta["micro_crack"]):
            vid_name = f"micro_{off_m + i:05d}"
            out = crack_dir / vid_name
            crack_center = _rand(rng, 0.65, 0.80)
            crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1
            seed = SEED_OFFSET + 250000 + off_m + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS, width=1920, height=1080,
                panel_length_px=_rand(rng, 215, 245), panel_thickness_um=90.0, pixels_per_mm=10.0,
                meters_per_pixel=1e-4, noise_std=_rand(rng, 0.18, 0.38), seed=seed,
                scenario="micro_crack", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "goal1", "scenario": "micro_crack", "label": 1, "crack_frame": crack_frame,
                "split": mc_splits[i], "dataset_id": vid_name,
            })
            manifest_entries.append({
                "path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1,
                "split": mc_splits[i], "crack_frame": crack_frame, "scenario": "micro_crack",
            })
            if (i + 1) % 100 == 0 or i == delta["micro_crack"] - 1:
                print(f"  [OK] {i + 1}/{delta['micro_crack']}")

    # --- Pre-damaged ---
    if delta["pre_damaged"] > 0:
        pd_splits = _assign_split(rng, delta["pre_damaged"])
        print(f"\nGenerating pre_damaged (+{delta['pre_damaged']})...")
        for i in range(delta["pre_damaged"]):
            vid_name = f"predam_{off_p + i:05d}"
            out = predam_dir / vid_name
            seed = SEED_OFFSET + 300000 + off_p + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS, width=1920, height=1080,
                panel_length_px=_rand(rng, 215, 245), panel_thickness_um=90.0, pixels_per_mm=10.0,
                meters_per_pixel=1e-4, noise_std=_rand(rng, 0.18, 0.35), seed=seed,
                scenario="pre_damage", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "goal2", "scenario": "pre_damage", "label": 1, "crack_frame": -1,
                "split": pd_splits[i], "dataset_id": vid_name,
            })
            manifest_entries.append({"path": f"pre_damaged/{vid_name}", "goal": "goal2", "label": 1, "split": pd_splits[i]})
            if (i + 1) % 100 == 0 or i == delta["pre_damaged"] - 1:
                print(f"  [OK] {i + 1}/{delta['pre_damaged']}")

    # --- Thick panel ---
    if delta["thick_panel"] > 0:
        thick_splits = _assign_split(rng, delta["thick_panel"])
        print(f"\nGenerating thick_panel (+{delta['thick_panel']})...")
        for i in range(delta["thick_panel"]):
            vid_name = f"thick_{off_t + i:05d}"
            out = thick_dir / vid_name
            seed = SEED_OFFSET + 400000 + off_t + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS, width=1920, height=1080,
                panel_length_px=_rand(rng, 220, 248), panel_thickness_um=90.0, pixels_per_mm=10.0,
                meters_per_pixel=1e-4, noise_std=_rand(rng, 0.12, 0.30), seed=seed,
                scenario="thick_panel", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "variant", "scenario": "thick_panel", "label": 0, "crack_frame": -1,
                "split": thick_splits[i], "dataset_id": vid_name,
            })
            manifest_entries.append({"path": f"thick_panel/{vid_name}", "goal": "variant", "label": 0, "split": thick_splits[i]})
            if (i + 1) % 100 == 0 or i == delta["thick_panel"] - 1:
                print(f"  [OK] {i + 1}/{delta['thick_panel']}")

    # --- Merge manifest ---
    train_count = sum(1 for e in manifest_entries if e["split"] == "train")
    val_count = sum(1 for e in manifest_entries if e["split"] == "val")
    test_count = sum(1 for e in manifest_entries if e["split"] == "test")
    new_counts = _count_existing(manifest_entries)
    manifest_out = {
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "total_count": len(manifest_entries),
        "normal": new_counts["normal"],
        "normal_light_distortion": new_counts["light_distortion"],
        "crack_in_bending": new_counts["crack"],
        "micro_crack": new_counts["micro_crack"],
        "pre_damaged_panel": new_counts["pre_damaged"],
        "thick_panel": new_counts["thick_panel"],
        "splits": {"train": train_count, "val": val_count, "test": test_count},
        "entries": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=True, indent=2), encoding="utf-8")
    print()
    print(f"Supplement complete. Total: {len(manifest_entries)}")
    print(f"  train={train_count}, val={val_count}, test={test_count}")


if __name__ == "__main__":
    main()
