#!/usr/bin/env python3
"""Batch preflight for bundle directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionanalyzer.preflight import PreflightConfig, preflight_realdata_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Root to scan for frame_*.txt parents")
    p.add_argument("--report", type=Path, default=Path("reports/preflight/batch_preflight.json"))
    args = p.parse_args()

    cfg = PreflightConfig()
    seen: set[str] = set()
    results: list[dict] = []
    for frame in sorted(args.root.rglob("frame_*.txt")):
        bundle = frame.parent
        key = str(bundle.resolve())
        if key in seen:
            continue
        seen.add(key)
        summary, errors = preflight_realdata_bundle(bundle, cfg)
        results.append(
            {
                "path": key,
                "passed": summary.passed,
                "frame_count": summary.frame_count,
                "errors": errors[:20],
            }
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"bundles": results}, indent=2), encoding="utf-8")
    n_pass = sum(1 for r in results if r["passed"])
    print(f"Preflight: {n_pass}/{len(results)} passed → {args.report}")


if __name__ == "__main__":
    main()
