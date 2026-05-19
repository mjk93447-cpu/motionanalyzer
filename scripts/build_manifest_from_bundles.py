#!/usr/bin/env python3
"""Build manifest.json from bundle tree under data/raw."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Root containing video/bundle folders")
    p.add_argument("--output", type=Path, required=True, help="Output dataset dir (manifest parent)")
    p.add_argument("--dataset-id", type=str, default="")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_id = args.dataset_id or args.output.name
    bundles: list[Path] = []
    for frame in sorted(args.root.rglob("frame_*.txt")):
        b = frame.parent
        if b not in bundles:
            bundles.append(b)

    rng = random.Random(args.seed)
    rng.shuffle(bundles)
    n = len(bundles)
    n_test = max(1, int(n * args.test_ratio)) if n > 2 else 0
    n_val = max(1, int(n * args.val_ratio)) if n > 3 else 0
    test_set = set(bundles[:n_test])
    val_set = set(bundles[n_test : n_test + n_val])

    entries: list[dict] = []
    for b in bundles:
        if b in test_set:
            split = "test"
        elif b in val_set:
            split = "val"
        else:
            split = "train"
        try:
            rel = b.relative_to(args.root)
        except ValueError:
            rel = b.name
        entries.append(
            {
                "path": rel.as_posix() if hasattr(rel, "as_posix") else str(rel),
                "label": 0,
                "goal": "goal1",
                "split": split,
                "scenario": "normal",
            }
        )

    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_count": len(entries),
        "entries": entries,
    }
    out_path = args.output / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} entries → {out_path}")


if __name__ == "__main__":
    main()
