"""
P1 6단계 임계값 비교를 강제로 실행합니다.
run_normal_fp_improvement_loop의 early-exit 문제를 우회합니다.

Usage:
  python scripts/run_p1_full_comparison.py --max-train 2000
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
ANALYSIS_JSON = repo_root / "reports" / "crack_detection_analysis" / "analysis.json"
LOOP1_JSON = repo_root / "reports" / "loop1_threshold_comparison.json"

STEPS = [
    {"normal_fp_max": 0, "threshold_margin": 0.0, "threshold_percentile": None, "label": "Phase 1: normal-fp-max=0"},
    {"normal_fp_max": 0, "threshold_margin": 0.1, "threshold_percentile": None, "label": "Phase 2a: +margin=0.1"},
    {"normal_fp_max": 0, "threshold_margin": 0.2, "threshold_percentile": None, "label": "Phase 2b: +margin=0.2"},
    {"normal_fp_max": None, "threshold_margin": 0.0, "threshold_percentile": 99.9, "label": "Phase 2e: percentile=99.9"},
    {"normal_fp_max": None, "threshold_margin": 0.1, "threshold_percentile": 99.9, "label": "Phase 2f: percentile=99.9 +margin=0.1"},
    {"normal_fp_max": 1, "threshold_margin": 0.0, "threshold_percentile": None, "label": "Phase 1b: normal-fp-max=1"},
]


def run_step(base: Path, cfg: dict, max_train: int | None) -> dict | None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_crack_detection.py"),
        "--base-dir", str(base),
        "--dataset-level-eval",
    ]
    if cfg.get("normal_fp_max") is not None:
        cmd += ["--normal-fp-max", str(cfg["normal_fp_max"])]
    if cfg.get("threshold_margin", 0) > 0:
        cmd += ["--threshold-margin", str(cfg["threshold_margin"])]
    if cfg.get("threshold_percentile") is not None:
        cmd += ["--threshold-percentile", str(cfg["threshold_percentile"])]
    if max_train is not None:
        cmd += ["--max-train", str(max_train)]

    ret = subprocess.run(cmd, cwd=str(repo_root), timeout=3600).returncode
    if ret != 0:
        print(f"  [WARN] analyze exit {ret}")
        return None

    if not ANALYSIS_JSON.exists():
        return None
    data = json.loads(ANALYSIS_JSON.read_text(encoding="utf-8"))
    models = data.get("models", {})
    ens = models.get("Ensemble", models.get("DREAM", {}))
    dream = models.get("DREAM", {})
    pc = models.get("PatchCore", {})
    n_normal = data.get("n_normal", 0)
    rate = ens.get("normal_fp_rate", 0) if n_normal else 0
    return {
        "label": cfg["label"],
        "normal_fp_max": cfg.get("normal_fp_max"),
        "threshold_margin": cfg.get("threshold_margin"),
        "threshold_percentile": cfg.get("threshold_percentile"),
        "fp": ens.get("fp", 0),
        "tn": ens.get("tn", 0),
        "fn": ens.get("fn", 0),
        "tp": ens.get("tp", 0),
        "recall": ens.get("recall"),
        "precision": ens.get("precision"),
        "normal_fp_rate": rate,
        "dream_threshold": dream.get("best_threshold"),
        "patchcore_threshold": pc.get("best_threshold"),
        "status": "target_achieved" if rate <= 0.001 else "above_target",
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="P1 6-step threshold comparison (no early exit)")
    ap.add_argument("--base-dir", default=str(repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"))
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not (base / "manifest.json").exists():
        print(f"ERROR: manifest.json not found in {base}")
        sys.exit(1)

    print("=" * 60)
    print("P1 Full 6-Step Threshold Comparison")
    print("=" * 60)
    print(f"Base: {base}\n")

    results = []
    for i, cfg in enumerate(STEPS):
        print(f"[Step {i+1}/6] {cfg['label']}")
        rec = run_step(base, cfg, args.max_train)
        if rec:
            results.append(rec)
            print(f"  FP={rec['fp']} Recall={rec.get('recall', 0):.4f} Precision={rec.get('precision', 0):.4f}")

    LOOP1_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_dir": str(base),
        "run_date": str(date.today()),
        "target_fp_rate": 0.001,
        "steps": results,
        "note": "run_p1_full_comparison.py (all 6 steps)",
    }
    LOOP1_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved {LOOP1_JSON} ({len(results)} steps)")


if __name__ == "__main__":
    main()
