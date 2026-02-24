"""
Supplement ML dataset with edge_scorch scenario and diversity improvements.

Adds:
- edge_scorch: Laser cutting edge scorch → weakened bonding → edge gape during bend (real-world reported)
- Optional: additional normal/crack for diversity

Keeps existing data intact. Merges into manifest.json.

Usage:
  python scripts/supplement_edge_scorch.py
  python scripts/supplement_edge_scorch.py --with-diversity  # add +5% normal/light_dist/crack
"""

from __future__ import annotations

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
from motionanalyzer.synthetic import NoiseMode

FRAMES = 60
FPS = 30.0
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED_OFFSET = 600000
NORMAL_POINTS_RANGE = (200, 260)
NORMAL_NOISE_RANGE = (0.12, 0.38)
NORMAL_PANEL_LENGTH_RANGE = (210, 250)
NORMAL_PIXELS_PER_MM_RANGE = (8.0, 12.0)
NOISE_MODES: tuple[NoiseMode, ...] = ("gaussian", "outlier", "temporal_drift", "scale_jitter", "mixed")


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


def _max_index(entries: list[dict], prefix: str) -> int:
    """Find max numeric suffix for paths like {prefix}_0001."""
    max_idx = 0
    for e in entries:
        path = e.get("path", "")
        if path.startswith(prefix):
            try:
                suffix = path.split("_")[-1]
                idx = int(suffix)
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                pass
    return max_idx


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Add edge_scorch + optional diversity")
    parser.add_argument("--with-diversity", action="store_true", help="Add +5% normal/light_dist/crack")
    parser.add_argument("--n-edge-scorch", type=int, default=600, help="Number of edge_scorch samples (default 600)")
    parser.add_argument("--seed", type=int, default=20260225)
    args = parser.parse_args()

    base_dir = repo_root / "data" / "synthetic" / "ml_dataset"
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        print("Error: manifest.json not found. Run generate_ml_dataset.py and supplement_ml_dataset.py first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    rng = np.random.default_rng(seed=args.seed)

    edge_dir = base_dir / "edge_scorch"
    edge_dir.mkdir(parents=True, exist_ok=True)
    normal_dir = base_dir / "normal"
    crack_dir = base_dir / "crack_in_bending"

    edge_splits = _assign_split(rng, args.n_edge_scorch)
    manifest_entries = list(entries)

    print("=" * 60)
    print("Supplement: edge_scorch (laser edge scorch → edge gape)")
    print("=" * 60)
    print(f"Adding {args.n_edge_scorch} edge_scorch samples (goal1, label=1)")
    print()

    for i in range(args.n_edge_scorch):
        vid_name = f"edge_scorch_{i + 1:05d}"
        out = edge_dir / vid_name
        seed = SEED_OFFSET + i
        config = SyntheticConfig(
            frames=FRAMES,
            points_per_frame=_randint(rng, 210, 250),
            fps=FPS,
            width=1920,
            height=1080,
            panel_length_px=_rand(rng, 215, 245),
            panel_thickness_um=90.0,
            pixels_per_mm=10.0,
            meters_per_pixel=1e-4,
            noise_std=_rand(rng, 0.15, 0.35),
            seed=seed,
            scenario="edge_scorch",
            noise_mode=_noise_mode_for_seed(seed),
        )
        out.mkdir(parents=True, exist_ok=True)
        generate_synthetic_bundle(out, config, extra_metadata={
            "goal": "goal1",
            "scenario": "edge_scorch",
            "label": 1,
            "crack_frame": -1,
            "split": edge_splits[i],
            "dataset_id": vid_name,
        })
        manifest_entries.append({
            "path": f"edge_scorch/{vid_name}",
            "goal": "goal1",
            "label": 1,
            "split": edge_splits[i],
            "scenario": "edge_scorch",
            "crack_frame": -1,
        })
        if (i + 1) % 100 == 0 or i == args.n_edge_scorch - 1:
            print(f"  [OK] {i + 1}/{args.n_edge_scorch}")

    if args.with_diversity:
        n_add = max(100, len(entries) // 20)  # ~5%
        n_norm = min(n_add // 2, 500)
        n_ld = min(n_add // 6, 100)
        n_crack = min(n_add // 3, 200)
        print(f"\nAdding diversity: +{n_norm} normal, +{n_ld} light_distortion, +{n_crack} crack...")
        off_n = _max_index(entries, "normal/normal_") + 1
        off_ld = _max_index(entries, "normal/normal_ld_") + 1
        off_c = _max_index(entries, "crack_in_bending/crack_") + 1
        off_m = _max_index(entries, "crack_in_bending/micro_") + 1
        for i in range(n_norm):
            vid_name = f"normal_{off_n + i:05d}"
            out = normal_dir / vid_name
            seed = SEED_OFFSET + 100000 + off_n + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, *NORMAL_POINTS_RANGE), fps=FPS,
                width=1920, height=1080, panel_length_px=_rand(rng, *NORMAL_PANEL_LENGTH_RANGE),
                panel_thickness_um=90.0, pixels_per_mm=_rand(rng, *NORMAL_PIXELS_PER_MM_RANGE),
                meters_per_pixel=1e-3 / 10, noise_std=_rand(rng, *NORMAL_NOISE_RANGE),
                seed=seed, scenario="normal", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            sp = _assign_split(rng, 1)[0]
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "normal", "scenario": "normal", "label": 0, "crack_frame": -1,
                "split": sp, "dataset_id": vid_name,
            })
            manifest_entries.append({"path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": sp})
        for i in range(n_ld):
            vid_name = f"normal_ld_{off_ld + i:05d}"
            out = normal_dir / vid_name
            seed = SEED_OFFSET + 150000 + off_ld + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS,
                width=1920, height=1080, panel_length_px=_rand(rng, 215, 245),
                panel_thickness_um=90.0, pixels_per_mm=10.0, meters_per_pixel=1e-4,
                noise_std=_rand(rng, 0.15, 0.35), seed=seed,
                scenario="light_distortion", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            sp = _assign_split(rng, 1)[0]
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "normal", "scenario": "light_distortion", "label": 0, "crack_frame": -1,
                "split": sp, "dataset_id": vid_name,
            })
            manifest_entries.append({
                "path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": sp,
                "scenario": "light_distortion",
            })
        for i in range(n_crack):
            vid_name = f"crack_{off_c + i:05d}"
            out = crack_dir / vid_name
            crack_center = _rand(rng, 0.65, 0.80)
            crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1
            seed = SEED_OFFSET + 200000 + off_c + i
            config = SyntheticConfig(
                frames=FRAMES, points_per_frame=_randint(rng, 210, 250), fps=FPS,
                width=1920, height=1080, panel_length_px=_rand(rng, 215, 245),
                panel_thickness_um=90.0, pixels_per_mm=10.0, meters_per_pixel=1e-4,
                noise_std=_rand(rng, 0.15, 0.35), seed=seed,
                scenario="crack", noise_mode=_noise_mode_for_seed(seed),
            )
            out.mkdir(parents=True, exist_ok=True)
            sp = _assign_split(rng, 1)[0]
            generate_synthetic_bundle(out, config, extra_metadata={
                "goal": "goal1", "scenario": "crack", "label": 1, "crack_frame": crack_frame,
                "split": sp, "dataset_id": vid_name,
            })
            manifest_entries.append({
                "path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1,
                "split": sp, "crack_frame": crack_frame,
            })
        print(f"  [OK] +{n_norm} normal, +{n_ld} light_distortion, +{n_crack} crack")

    train_count = sum(1 for e in manifest_entries if e["split"] == "train")
    val_count = sum(1 for e in manifest_entries if e["split"] == "val")
    test_count = sum(1 for e in manifest_entries if e["split"] == "test")
    n_edge = sum(1 for e in manifest_entries if e.get("scenario") == "edge_scorch")
    manifest_out = dict(manifest)
    manifest_out.update({
        "version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "total_count": len(manifest_entries),
        "edge_scorch": n_edge,
        "splits": {"train": train_count, "val": val_count, "test": test_count},
        "entries": manifest_entries,
    })
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=True, indent=2), encoding="utf-8")
    print()
    print(f"Supplement complete. Total: {len(manifest_entries)}")
    print(f"  edge_scorch: {n_edge}")
    print(f"  train={train_count}, val={val_count}, test={test_count}")


if __name__ == "__main__":
    main()
