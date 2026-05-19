#!/usr/bin/env python3
"""Benchmark pretrained bundle on synthetic manifest test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.paths import (
    get_default_ml_dataset_dir,
    get_fp_focused_dataset_dir,
    get_pretrain_balanced_dataset_dir,
    get_user_models_dir,
)
from motionanalyzer.services.ml_inference import predict_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--ci-fast", action="store_true")
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--max-fp", type=int, default=None)
    args = ap.parse_args()

    models_dir = args.models_dir or get_user_models_dir()
    if not (models_dir / "bundle_manifest.json").exists():
        release = repo_root / "release" / "models" / "bundle_manifest.json"
        if release.exists():
            models_dir = release.parent

    balanced = get_pretrain_balanced_dataset_dir()
    base = args.manifest.parent if args.manifest else (
        get_default_ml_dataset_dir()
        if args.ci_fast
        else (balanced if (balanced / "manifest.json").exists() else get_fp_focused_dataset_dir())
    )
    manifest = args.manifest or (base / "manifest.json")
    if not manifest.exists():
        print(f"Missing manifest: {manifest}")
        sys.exit(1)

    max_bundles = 60 if args.ci_fast else None
    min_p = args.min_precision if args.min_precision is not None else (0.50 if args.ci_fast else 0.0)
    max_fp = args.max_fp if args.max_fp is not None else (35 if args.ci_fast else 999999)

    report: dict[str, object] = {"manifest": str(manifest), "models_dir": str(models_dir)}
    failed = False
    for mode in ("draem", "patchcore", "ensemble"):
        rep = predict_manifest(
            manifest,
            mode,  # type: ignore[arg-type]
            models_dir=models_dir,
            split="test",
            max_bundles=max_bundles,
        )
        m = rep["metrics"]
        report[mode] = m
        print(f"{mode}: precision={m['precision']:.3f} recall={m['recall']:.3f} FP={m['fp']} FN={m['fn']}")
        if args.ci_fast and mode == "patchcore" and m["precision"] < min_p:
            print(f"FAIL: patchcore precision {m['precision']} < {min_p}")
            failed = True
        if args.ci_fast and mode == "ensemble" and m["fp"] > max_fp:
            print(f"FAIL: ensemble FP {m['fp']} > {max_fp}")
            failed = True

    out = repo_root / "reports" / "benchmark_ml_models.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
