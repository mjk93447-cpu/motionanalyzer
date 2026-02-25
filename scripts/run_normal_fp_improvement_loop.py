"""
Normal FP Rate 0.1% 달성까지 반복 실행.

Roadmap Phase 1~2를 순차 적용, 각 단계에서 학습·테스트 후 analysis.json 확인.
목표: normal_fp_rate <= 0.001 (0.1%)

Usage:
  python scripts/run_normal_fp_improvement_loop.py
  python scripts/run_normal_fp_improvement_loop.py --max-train 3000  # 빠른 반복
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
ANALYSIS_JSON = repo_root / "reports" / "crack_detection_analysis" / "analysis.json"
LOOP1_JSON = repo_root / "reports" / "loop1_threshold_comparison.json"
TARGET_FP_RATE = 0.001  # 0.1%


def get_step_result(cfg: dict) -> dict | None:
    """Read analysis.json and return step record for loop1_threshold_comparison."""
    if not ANALYSIS_JSON.exists():
        return None
    data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    n_normal = data.get("n_normal", 0)
    models = data.get("models", {})
    ens = models.get("Ensemble", models.get("DREAM", {}))
    dream = models.get("DREAM", {})
    pc = models.get("PatchCore", {})
    out = {
        "label": cfg.get("label", ""),
        "normal_fp_max": cfg.get("normal_fp_max"),
        "threshold_margin": cfg.get("threshold_margin"),
        "threshold_percentile": cfg.get("threshold_percentile"),
        "fp": ens.get("fp", 0),
        "tn": ens.get("tn", 0),
        "fn": ens.get("fn", 0),
        "tp": ens.get("tp", 0),
        "recall": ens.get("recall"),
        "precision": ens.get("precision"),
        "normal_fp_rate": ens.get("normal_fp_rate", 0) if n_normal else 0,
        "dream_threshold": dream.get("threshold"),
        "patchcore_threshold": pc.get("threshold"),
        "status": "target_achieved" if (ens.get("fp", 0) / max(1, n_normal)) <= TARGET_FP_RATE else "above_target",
    }
    return out


def get_normal_fp_rate() -> tuple[float, int, int] | None:
    """(normal_fp_rate, fp, n_normal) or None if not found."""
    if not ANALYSIS_JSON.exists():
        return None
    data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    n_normal = data.get("n_normal", 0)
    models = data.get("models", {})
    ens = models.get("Ensemble", models.get("DREAM", {}))
    fp = ens.get("fp", 0)
    if n_normal <= 0:
        return None
    rate = ens.get("normal_fp_rate", fp / n_normal if n_normal else 0)
    return float(rate), fp, n_normal


def run_analysis(
    base_dir: Path,
    normal_fp_max: int | None = None,
    threshold_margin: float = 0.0,
    threshold_percentile: float | None = None,
    max_train: int | None = None,
) -> int:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_crack_detection.py"),
        "--base-dir", str(base_dir),
        "--dataset-level-eval",
    ]
    if normal_fp_max is not None:
        cmd += ["--normal-fp-max", str(normal_fp_max)]
    if threshold_margin > 0:
        cmd += ["--threshold-margin", str(threshold_margin)]
    if threshold_percentile is not None:
        cmd += ["--threshold-percentile", str(threshold_percentile)]
    if max_train is not None:
        cmd += ["--max-train", str(max_train)]

    return subprocess.run(cmd, cwd=str(repo_root), timeout=3600).returncode


def _write_loop1_json(base: Path, steps: list[dict]) -> None:
    from datetime import date
    payload = {
        "base_dir": str(base),
        "run_date": str(date.today()),
        "target_fp_rate": TARGET_FP_RATE,
        "steps": steps,
    }
    LOOP1_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOOP1_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Saved] {LOOP1_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Normal FP improvement loop")
    ap.add_argument("--base-dir", default=str(repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"))
    ap.add_argument("--max-train", type=int, default=None, help="Limit train for faster runs")
    ap.add_argument("--max-steps", type=int, default=6, help="Max improvement steps")
    ap.add_argument("--full-comparison", action="store_true", help="Run all 6 steps (P1); default: exit when target reached")
    ap.add_argument("--no-early-exit", action="store_true", help="Same as --full-comparison")
    args = ap.parse_args()
    # Fallback: some shells may not pass flags; check argv for explicit request
    _argv_lower = " ".join(getattr(sys, "argv", [])).lower()
    run_all_steps = (
        getattr(args, "full_comparison", False) or getattr(args, "no_early_exit", False)
        or "--full-comparison" in _argv_lower or "--no-early-exit" in _argv_lower
    )

    base = Path(args.base_dir)
    if not (base / "manifest.json").exists():
        print(f"[ERROR] manifest.json not found in {base}")
        sys.exit(1)

    steps = [
        {"normal_fp_max": 0, "threshold_margin": 0.0, "threshold_percentile": None, "label": "Phase 1: normal-fp-max=0"},
        {"normal_fp_max": 0, "threshold_margin": 0.1, "threshold_percentile": None, "label": "Phase 2a: +margin=0.1"},
        {"normal_fp_max": 0, "threshold_margin": 0.2, "threshold_percentile": None, "label": "Phase 2b: +margin=0.2"},
        {"normal_fp_max": None, "threshold_margin": 0.0, "threshold_percentile": 99.9, "label": "Phase 2e: percentile=99.9 (~0.1% FP)"},
        {"normal_fp_max": None, "threshold_margin": 0.1, "threshold_percentile": 99.9, "label": "Phase 2f: percentile=99.9 +margin=0.1"},
        {"normal_fp_max": 1, "threshold_margin": 0.0, "threshold_percentile": None, "label": "Phase 1b: normal-fp-max=1"},
    ]

    print("=" * 60)
    print("Normal FP Rate Improvement Loop (target <= 0.1%)")
    print("=" * 60)
    print(f"Base: {base}")
    print()

    results: list[dict] = []

    for i, cfg in enumerate(steps[: args.max_steps]):
        print(f"[Step {i+1}] {cfg['label']}")
        ret = run_analysis(
            base,
            normal_fp_max=cfg["normal_fp_max"],
            threshold_margin=cfg["threshold_margin"],
            threshold_percentile=cfg.get("threshold_percentile"),
            max_train=args.max_train,
        )
        if ret != 0:
            print(f"  [WARN] analyze exit code {ret}")
        r = get_normal_fp_rate()
        step_rec = get_step_result(cfg)
        if step_rec:
            results.append(step_rec)
        if r is None:
            print("  [WARN] Could not read analysis.json")
            continue
        rate, fp, n = r
        print(f"  Normal FP rate: {rate:.4%} (FP={fp}, n_normal={n})")
        if rate <= TARGET_FP_RATE and not run_all_steps:
            _write_loop1_json(base, results)
            print()
            print("[OK] Target achieved: Normal FP rate <= 0.1%")
            sys.exit(0)

    _write_loop1_json(base, results)
    print()
    print("[INFO] Max steps reached. Consider Phase 3+ (loss/feature improvements).")
    r = get_normal_fp_rate()
    if r:
        print(f"  Final: {r[0]:.4%} (FP={r[1]}, n_normal={r[2]})")
    sys.exit(1)
