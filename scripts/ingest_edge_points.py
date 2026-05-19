#!/usr/bin/env python3
"""Ingest flat edge point TXT exports into MotionAnalyzer bundle layout."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def _parse_frame_index(name: str, pattern: str) -> int | None:
    m = re.search(pattern, name)
    if not m:
        digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
        return int(digits) if digits else None
    return int(m.group(1))


def _read_points(path: Path) -> list[tuple[int, int, int]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    start = 1 if lines[0].strip().startswith("#") else 0
    points: list[tuple[int, int, int]] = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        x, y = int(float(parts[0])), int(float(parts[1]))
        idx = int(parts[2]) if len(parts) >= 3 else len(points) + 1
        points.append((x, y, idx))
    return points


def ingest(
    source_dir: Path,
    output_root: Path,
    *,
    video_pattern: str = r"(.+?)_frame_(\d+)",
    file_glob: str = "*.txt",
    default_fps: float = 30.0,
    synthetic_index: bool = False,
) -> dict[str, int]:
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    buckets: dict[tuple[str, str], dict[int, list[tuple[int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for path in sorted(source_dir.rglob(file_glob)):
        if path.name.lower() == "fps.txt":
            continue
        rel = path.relative_to(source_dir)
        video_id = rel.parts[0] if len(rel.parts) > 1 else "default"
        bundle_id = "run_001"
        frame_idx: int | None = None
        if len(rel.parts) >= 2:
            bundle_id = rel.parts[1] if len(rel.parts) > 2 else rel.stem
            frame_idx = _parse_frame_index(path.name, r"(\d+)")
        else:
            m = re.match(video_pattern, path.stem)
            if m:
                video_id, frame_idx = m.group(1), int(m.group(2))
            else:
                frame_idx = _parse_frame_index(path.name, r"(\d+)")
        if frame_idx is None:
            continue
        pts = _read_points(path)
        if not pts:
            continue
        if synthetic_index:
            pts = [(x, y, i + 1) for i, (x, y, _) in enumerate(sorted(pts, key=lambda t: t[2]))]
        buckets[(video_id, bundle_id)][frame_idx] = pts

    n_bundles = 0
    for (video_id, bundle_id), frames in buckets.items():
        out_dir = output_root / video_id / bundle_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for fi in sorted(frames.keys()):
            pts = frames[fi]
            out_path = out_dir / f"frame_{fi:05d}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                f.write("# x,y,index\n")
                for x, y, idx in pts:
                    f.write(f"{x},{y},{idx}\n")
        (out_dir / "fps.txt").write_text(f"{default_fps}\n", encoding="utf-8")
        meta = {"synthetic_index": synthetic_index, "source": str(source_dir)}
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        n_bundles += 1
    return {"bundles": n_bundles, "output_root": str(output_root)}


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest edge TXT into bundle folders")
    p.add_argument("--source", type=Path, required=True, help="Flat or nested source TXT root")
    p.add_argument("--output", type=Path, required=True, help="Output data/raw root")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--synthetic-index", action="store_true", help="Assign index 1..N per frame")
    args = p.parse_args()
    stats = ingest(args.source, args.output, default_fps=args.fps, synthetic_index=args.synthetic_index)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
