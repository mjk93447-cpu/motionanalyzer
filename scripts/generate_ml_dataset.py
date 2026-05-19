"""
Generate ML training/evaluation synthetic dataset.

Scale: configurable up to 100k with diversified anomaly scenarios.
Tags: goal, label, scenario, crack_frame, split.
Output: data/synthetic/ml_{scale}_* (see docs/DATASET_FOLDER_STRUCTURE.md). manifest.json includes dataset_id, scale.

Usage:
  python scripts/generate_ml_dataset.py
  python scripts/generate_ml_dataset.py --scale 100k --workers 4  # parallel
  python scripts/generate_ml_dataset.py --dry-run  # print plan only
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
DEFAULT_SPLIT_RATIOS = (0.90, 0.05, 0.05)  # train, val, test (precision-focused)

# Scale presets:
# (normal, light_dist, crack, uv, micro, predam, thick, overbend, underbend, jig_vibration)
SCALE_CONFIGS: dict[str, tuple[int, ...]] = {
    "default": (1000, 50, 50, 30, 10, 20, 20, 20, 10, 10),    # 1,220
    "10k": (7000, 500, 500, 300, 300, 500, 400, 300, 100, 100),  # 10,000
    "100k": (70000, 5000, 5000, 3000, 3000, 4000, 5000, 3000, 1000, 1000),  # 100,000
    # Normal 94%, Crack 6%: 실제 공정 반영, crack은 학습에 충분
    "fp_focused": (15000, 1000, 300, 300, 200, 150, 2800, 100, 100, 50),  # 20,000
    # Hard 10k: 복잡성/경계 케이스 강화 — 공장 데이터 추가 시 변수·예측불가 현상 상정
    "hard_10k": (6000, 1200, 350, 350, 450, 350, 100, 400, 250, 150),  # 10,000
    # Asymmetric 10k: 3D 불균형 접힘 모방 — 좌우 비대칭 휘어짐으로 정사영이 복잡한 패턴
    "asymmetric_10k": (6000, 1200, 350, 350, 450, 350, 100, 400, 250, 150),  # 10,000
    # Balanced NG diversity for ~30min pretrain (7:2:1 split); crack ~7%
    "pretrain_balanced": (2100, 140, 100, 70, 50, 40, 100, 30, 30, 20),  # ~3,030
}
# Canonical output dir by scale (docs/DATASET_FOLDER_STRUCTURE.md)
DEFAULT_DATASET_DIRS: dict[str, str] = {
    "default": "data/synthetic/ml_default_1k_60f",
    "10k": "data/synthetic/ml_10k_60f",
    "100k": "data/synthetic/ml_100k_60f",
    "fp_focused": "data/synthetic/ml_fp_focused_20k_60f",
    "hard_10k": "data/synthetic/ml_hard_10k_60f",
    "asymmetric_10k": "data/synthetic/ml_asymmetric_10k_60f",
    "pretrain_balanced": "data/synthetic/ml_pretrain_balanced_3k_60f",
    "small": "data/synthetic/ml_small_60f",
}
NOISE_MODES: tuple[NoiseMode, ...] = ("gaussian", "outlier", "temporal_drift", "scale_jitter", "mixed")

NORMAL_POINTS_RANGE = (200, 260)
NORMAL_NOISE_RANGE = (0.12, 0.38)
NORMAL_PANEL_LENGTH_RANGE = (210, 250)
NORMAL_PIXELS_PER_MM_RANGE = (8.0, 12.0)


def _rand(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _noise_mode_for_seed(seed: int) -> NoiseMode:
    """Assign noise mode from seed for reproducible diversity."""
    return NOISE_MODES[seed % len(NOISE_MODES)]


def _asymmetric_fold_strength(rng: np.random.Generator, scale: str) -> float:
    """3D asymmetric fold strength when scale is asymmetric_10k; 0 otherwise."""
    return float(rng.uniform(0.4, 0.9)) if scale == "asymmetric_10k" else 0.0


def _assign_split(rng: np.random.Generator, n: int, split_ratios: tuple[float, float, float]) -> list[str]:
    train_n = int(n * split_ratios[0])
    val_n = int(n * split_ratios[1])
    test_n = n - train_n - val_n
    splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
    rng.shuffle(splits)
    return splits


def _generate_one_sample(args: tuple[Any, ...]) -> dict[str, Any]:
    """Worker: generate one synthetic sample. Args = (out_dir, config_dict, extra_dict)."""
    out_dir, config_dict, extra_dict = args
    from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
    fields = {f for f in SyntheticConfig.__dataclass_fields__ if f in config_dict}
    config = SyntheticConfig(**{k: config_dict[k] for k in fields})
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generate_synthetic_bundle(out_path, config, extra_metadata=dict(extra_dict))
    return extra_dict


def _run_parallel(
    tasks: list[tuple[str, dict, dict]],
    manifest_entries: list[dict],
    manifest_entry_fn: Callable[[dict], dict],
    workers: int,
    total: int,
    label: str,
) -> None:
    """Run parallel generation with progress logging."""
    done = 0
    log_every = 5000 if total > 10000 else 500 if total > 1000 else 50
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_generate_one_sample, t): t for t in tasks}
        for f in as_completed(futures):
            extra = futures[f][2]
            manifest_entries.append(manifest_entry_fn(extra))
            done += 1
            if done % log_every == 0 or done == total:
                print(f"  [OK] {done}/{total} ({label})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML synthetic dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not generate")
    parser.add_argument("--small", action="store_true", help="Small set: 100 normal, 10 crack, 5 predam (quick test)")
    parser.add_argument("--scale", choices=["default", "10k", "100k", "fp_focused", "hard_10k", "asymmetric_10k", "pretrain_balanced"], default="default",
                        help="Dataset scale: default, 10k, 100k, fp_focused, hard_10k, asymmetric_10k, pretrain_balanced (~3k)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for generation (default 1 = sequential)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dataset directory (default: by scale, see docs/DATASET_FOLDER_STRUCTURE.md)")
    parser.add_argument("--split-train", type=float, default=DEFAULT_SPLIT_RATIOS[0], help="Train split ratio")
    parser.add_argument("--split-val", type=float, default=DEFAULT_SPLIT_RATIOS[1], help="Validation split ratio")
    parser.add_argument("--split-test", type=float, default=DEFAULT_SPLIT_RATIOS[2], help="Test split ratio")
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--resume", action="store_true", help="Skip existing samples (metadata.json present)")
    args = parser.parse_args()

    split_ratios = (args.split_train, args.split_val, args.split_test)
    if abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {split_ratios}")

    if args.small:
        n_normal, n_light_dist, n_crack_main, n_uv, n_micro_crack, n_predam, n_thick, n_overbend, n_underbend, n_jig = (
            100, 3, 7, 3, 2, 5, 5, 4, 1, 1
        )
    else:
        cfg = SCALE_CONFIGS[args.scale]
        (
            n_normal,
            n_light_dist,
            n_crack_main,
            n_uv,
            n_micro_crack,
            n_predam,
            n_thick,
            n_overbend,
            n_underbend,
            n_jig,
        ) = cfg
    n_crack = n_crack_main + n_uv  # crack_in_bending total

    output_dir = args.output_dir
    if output_dir is None:
        scale_key = "small" if args.small else args.scale
        output_dir = DEFAULT_DATASET_DIRS.get(scale_key, "data/synthetic/ml_default_1k_60f") if not args.small else "data/synthetic/ml_small_60f"
    base_dir = (repo_root / Path(output_dir)).resolve()
    rng = np.random.default_rng(seed=args.seed)
    dataset_id = base_dir.name

    total = n_normal + n_light_dist + n_crack + n_micro_crack + n_predam + n_thick + n_overbend + n_underbend + n_jig
    print("=" * 60)
    print("ML Dataset Generator")
    print("=" * 60)
    print(f"Output: {base_dir}")
    print(f"Scale: {args.scale if not args.small else 'small'} (total ~{total:,})")
    print(f"  - {n_normal} normal (noise modes: {NOISE_MODES})")
    print(f"  - {n_light_dist} normal_light_distortion")
    print(f"  - {n_crack} crack_in_bending (Goal 1: {n_crack_main} crack + {n_uv} uv_overcured)")
    print(f"  - {n_micro_crack} micro_crack (micro-scale crack)")
    print(f"  - {n_predam} pre_damaged (Goal 2)")
    print(f"  - {n_thick} thick_panel (variant)")
    print(f"  - {n_overbend} over_bending (trajectory anomaly)")
    print(f"  - {n_underbend} under_bending (trajectory anomaly)")
    print(f"  - {n_jig} jig_vibration (jig micro-shake anomaly)")
    crack_like_total = n_crack + n_micro_crack + n_predam + n_overbend + n_underbend + n_jig
    print(f"Crack-like ratio: {100.0 * crack_like_total / total:.2f}%")
    print(f"Split ratios: train={split_ratios[0]:.2f}, val={split_ratios[1]:.2f}, test={split_ratios[2]:.2f}")
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
    normal_splits = _assign_split(rng, n_normal, split_ratios)
    light_dist_splits = _assign_split(rng, n_light_dist, split_ratios)
    crack_splits = _assign_split(rng, n_crack, split_ratios)
    micro_crack_splits = _assign_split(rng, n_micro_crack, split_ratios)
    predam_splits = _assign_split(rng, n_predam, split_ratios)
    thick_splits = _assign_split(rng, n_thick, split_ratios)
    overbend_splits = _assign_split(rng, n_overbend, split_ratios)
    underbend_splits = _assign_split(rng, n_underbend, split_ratios)
    jig_splits = _assign_split(rng, n_jig, split_ratios)

    # --- Normal ---
    normal_dir = base_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, getattr(args, "workers", 1))
    use_parallel = workers > 1 and n_normal >= 50

    if use_parallel:
        print(f"Generating normal (workers={workers})...")
        tasks = []
        for i in range(1, n_normal + 1):
            vid_name = f"normal_{i:04d}"
            out = normal_dir / vid_name
            if args.resume and (out / "metadata.json").exists():
                manifest_entries.append({"path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": normal_splits[i - 1]})
                continue
            seed = 100000 + i
            points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
            noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
            panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
            px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])
            noise_mode = _noise_mode_for_seed(seed)
            extra = {
                "goal": "normal", "scenario": "normal", "label": 0, "crack_frame": -1,
                "split": normal_splits[i - 1], "dataset_id": vid_name, "noise_mode": noise_mode,
            }
            asym = _asymmetric_fold_strength(np.random.default_rng(seed), args.scale)
            cfg = SyntheticConfig(
                frames=FRAMES, points_per_frame=points, fps=FPS, width=1920, height=1080,
                panel_length_px=panel_len, panel_thickness_um=90.0, pixels_per_mm=px_per_mm,
                meters_per_pixel=1e-3 / px_per_mm, noise_std=noise, seed=seed,
                scenario="normal", noise_mode=noise_mode,
                asymmetric_fold_strength=asym,
            )
            from dataclasses import asdict
            tasks.append((str(out), asdict(cfg), extra))
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_generate_one_sample, t): t for t in tasks}
            for f in as_completed(futures):
                manifest_entries.append({
                    "path": f"normal/{futures[f][2]['dataset_id']}",
                    "goal": "normal", "label": 0, "split": futures[f][2]["split"],
                })
                done += 1
                if done % 5000 == 0 or done == n_normal:
                    print(f"  [OK] {done}/{n_normal}")
    else:
        print("Generating normal...")
        for i in range(1, n_normal + 1):
            vid_name = f"normal_{i:04d}"
            out = normal_dir / vid_name
            if args.resume and (out / "metadata.json").exists():
                manifest_entries.append({"path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": normal_splits[i - 1]})
                if i % 5000 == 0 or i == n_normal:
                    print(f"  [OK] {i}/{n_normal} (resumed)")
                continue
            out.mkdir(parents=True, exist_ok=True)

            seed = 100000 + i
            points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
            noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
            panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
            px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])

            noise_mode = _noise_mode_for_seed(seed)
            extra = {
                "goal": "normal",
                "scenario": "normal",
                "label": 0,
                "crack_frame": -1,
                "split": normal_splits[i - 1],
                "dataset_id": vid_name,
                "noise_mode": noise_mode,
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
                noise_mode=noise_mode,
                asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
            )
            generate_synthetic_bundle(out, config, extra_metadata=extra)
            manifest_entries.append({
                "path": f"normal/{vid_name}",
                "goal": "normal",
                "label": 0,
                "split": extra["split"],
            })
            log_every = 5000 if n_normal > 10000 else 200
            if i % log_every == 0 or i == n_normal:
                print(f"  [OK] {i}/{n_normal}")

    # --- Normal + light distortion (1-2% harder normal, label=0) ---
    print("\nGenerating normal_light_distortion...")
    for i in range(1, n_light_dist + 1):
        vid_name = f"normal_ld_{i:04d}"
        out = normal_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"normal/{vid_name}", "goal": "normal", "label": 0, "split": light_dist_splits[i - 1], "scenario": "light_distortion"})
            continue
        out.mkdir(parents=True, exist_ok=True)

        seed = 150000 + i
        points = _randint(rng, NORMAL_POINTS_RANGE[0], NORMAL_POINTS_RANGE[1])
        noise = _rand(rng, NORMAL_NOISE_RANGE[0], NORMAL_NOISE_RANGE[1])
        panel_len = _rand(rng, NORMAL_PANEL_LENGTH_RANGE[0], NORMAL_PANEL_LENGTH_RANGE[1])
        px_per_mm = _rand(rng, NORMAL_PIXELS_PER_MM_RANGE[0], NORMAL_PIXELS_PER_MM_RANGE[1])

        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "normal",
            "scenario": "light_distortion",
            "label": 0,
            "crack_frame": -1,
            "split": light_dist_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
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
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
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

    # --- Crack in bending (Goal 1): crack + uv_overcured (vibration/shockwave patterns) ---
    crack_dir = base_dir / "crack_in_bending"
    crack_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating crack_in_bending (Goal 1)...")
    n_crack_scenario = n_crack_main  # crack count; rest uv_overcured
    for i in range(1, n_crack + 1):
        use_crack = i <= n_crack_scenario
        scenario = "crack" if use_crack else "uv_overcured"
        vid_name = f"crack_{i:04d}"
        out = crack_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            rng_i = np.random.default_rng(200000 + i * 137)
            crack_frame = int(_rand(rng_i, 0.65, 0.80) * (FRAMES - 1)) if FRAMES > 1 else -1
            manifest_entries.append({"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": crack_splits[i - 1], "crack_frame": crack_frame})
            if i % 500 == 0 or i == n_crack:
                print(f"  [OK] {i}/{n_crack} (resumed)")
            continue
        out.mkdir(parents=True, exist_ok=True)

        crack_center = _rand(rng, 0.65, 0.80)
        crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1

        seed = 200000 + i * 137
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "goal1",
            "scenario": scenario,
            "label": 1,
            "crack_frame": crack_frame,
            "split": crack_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"crack_in_bending/{vid_name}",
            "goal": "goal1",
            "label": 1,
            "split": extra["split"],
            "crack_frame": crack_frame,
        })
        log_every = 500 if n_crack > 1000 else 20
        if i % log_every == 0 or i == n_crack:
            print(f"  [OK] {i}/{n_crack} ({scenario})")

    # --- Micro crack (harder crack, label=1) ---
    print("\nGenerating micro_crack...")
    for i in range(1, n_micro_crack + 1):
        vid_name = f"micro_{i:04d}"
        out = crack_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": micro_crack_splits[i - 1], "scenario": "micro_crack"})
            continue
        out.mkdir(parents=True, exist_ok=True)

        crack_center = _rand(rng, 0.65, 0.80)
        crack_frame = int(crack_center * (FRAMES - 1)) if FRAMES > 1 else -1

        seed = 250000 + i * 211
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "goal1",
            "scenario": "micro_crack",
            "label": 1,
            "crack_frame": crack_frame,
            "split": micro_crack_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
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
        log_every = 500 if n_micro_crack > 1000 else 1
        if i % log_every == 0 or i == n_micro_crack:
            print(f"  [OK] {i}/{n_micro_crack}")

    # --- Pre-damaged (Goal 2) ---
    predam_dir = base_dir / "pre_damaged"
    predam_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating pre_damaged (Goal 2)...")
    for i in range(1, n_predam + 1):
        vid_name = f"predam_{i:04d}"
        out = predam_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"pre_damaged/{vid_name}", "goal": "goal2", "label": 1, "split": predam_splits[i - 1]})
            continue
        out.mkdir(parents=True, exist_ok=True)

        seed = 300000 + i * 97
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "goal2",
            "scenario": "pre_damage",
            "label": 1,
            "crack_frame": -1,
            "split": predam_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"pre_damaged/{vid_name}",
            "goal": "goal2",
            "label": 1,
            "split": extra["split"],
        })
        log_every = 500 if n_predam > 1000 else 1
        if i % log_every == 0 or i == n_predam:
            print(f"  [OK] {i}/{n_predam}")

    # --- Thick panel (variant, boundary case, label=0) ---
    thick_dir = base_dir / "thick_panel"
    thick_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating thick_panel (variant)...")
    for i in range(1, n_thick + 1):
        vid_name = f"thick_{i:04d}"
        out = thick_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"thick_panel/{vid_name}", "goal": "variant", "label": 0, "split": thick_splits[i - 1]})
            if i % 500 == 0 or i == n_thick:
                print(f"  [OK] {i}/{n_thick} (resumed)")
            continue
        out.mkdir(parents=True, exist_ok=True)

        seed = 400000 + i * 73
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "variant",
            "scenario": "thick_panel",
            "label": 0,
            "crack_frame": -1,
            "split": thick_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append({
            "path": f"thick_panel/{vid_name}",
            "goal": "variant",
            "label": 0,
            "split": extra["split"],
        })
        log_every = 500 if n_thick > 1000 else 5
        if i % log_every == 0 or i == n_thick:
            print(f"  [OK] {i}/{n_thick}")

    # --- Over-bending anomaly (label=1) ---
    print("\nGenerating over_bending...")
    for i in range(1, n_overbend + 1):
        vid_name = f"overbend_{i:04d}"
        out = crack_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": overbend_splits[i - 1], "scenario": "over_bending"})
            continue
        out.mkdir(parents=True, exist_ok=True)
        seed = 450000 + i * 61
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "goal1",
            "scenario": "over_bending",
            "label": 1,
            "crack_frame": int(_rand(rng, 0.62, 0.82) * (FRAMES - 1)),
            "split": overbend_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_std=_rand(rng, 0.18, 0.35),
            seed=seed,
            scenario="over_bending",
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append(
            {"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": extra["split"], "scenario": "over_bending"}
        )
        if i % 250 == 0 or i == n_overbend:
            print(f"  [OK] {i}/{n_overbend}")

    # --- Under-bending anomaly (label=1) ---
    print("\nGenerating under_bending...")
    for i in range(1, n_underbend + 1):
        vid_name = f"underbend_{i:04d}"
        out = crack_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": underbend_splits[i - 1], "scenario": "under_bending"})
            continue
        out.mkdir(parents=True, exist_ok=True)
        seed = 470000 + i * 53
        noise_mode = _noise_mode_for_seed(seed)
        extra = {
            "goal": "goal1",
            "scenario": "under_bending",
            "label": 1,
            "crack_frame": int(_rand(rng, 0.62, 0.82) * (FRAMES - 1)),
            "split": underbend_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_std=_rand(rng, 0.12, 0.30),
            seed=seed,
            scenario="under_bending",
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append(
            {"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": extra["split"], "scenario": "under_bending"}
        )
        if i % 250 == 0 or i == n_underbend:
            print(f"  [OK] {i}/{n_underbend}")

    # --- Jig vibration anomaly (label=1) ---
    print("\nGenerating jig_vibration...")
    for i in range(1, n_jig + 1):
        vid_name = f"jigvib_{i:04d}"
        out = crack_dir / vid_name
        if args.resume and (out / "metadata.json").exists():
            manifest_entries.append({"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": jig_splits[i - 1], "scenario": "jig_vibration"})
            continue
        out.mkdir(parents=True, exist_ok=True)
        seed = 490000 + i * 47
        noise_mode = "mixed"
        extra = {
            "goal": "goal1",
            "scenario": "jig_vibration",
            "label": 1,
            "crack_frame": int(_rand(rng, 0.62, 0.82) * (FRAMES - 1)),
            "split": jig_splits[i - 1],
            "dataset_id": vid_name,
            "noise_mode": noise_mode,
        }
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
            noise_std=_rand(rng, 0.18, 0.38),
            seed=seed,
            scenario="jig_vibration",
            noise_mode=noise_mode,
            asymmetric_fold_strength=_asymmetric_fold_strength(rng, args.scale),
        )
        generate_synthetic_bundle(out, config, extra_metadata=extra)
        manifest_entries.append(
            {"path": f"crack_in_bending/{vid_name}", "goal": "goal1", "label": 1, "split": extra["split"], "scenario": "jig_vibration"}
        )
        if i % 250 == 0 or i == n_jig:
            print(f"  [OK] {i}/{n_jig}")

    # --- Manifest ---
    train_count = sum(1 for e in manifest_entries if e["split"] == "train")
    val_count = sum(1 for e in manifest_entries if e["split"] == "val")
    test_count = sum(1 for e in manifest_entries if e["split"] == "test")
    scale_used = "small" if args.small else args.scale
    manifest = {
        "version": "1.0",
        "dataset_id": dataset_id,
        "scale": scale_used,
        "created_at": datetime.now(UTC).isoformat(),
        "total_count": len(manifest_entries),
        "normal": n_normal,
        "normal_light_distortion": n_light_dist,
        "crack_in_bending": n_crack,
        "micro_crack": n_micro_crack,
        "pre_damaged_panel": n_predam,
        "thick_panel": n_thick,
        "over_bending": n_overbend,
        "under_bending": n_underbend,
        "jig_vibration": n_jig,
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
