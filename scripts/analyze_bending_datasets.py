"""
Analyze all bending/synthetic video datasets in the project.
Output: structured report with unique IDs, counts, frames, creation dates.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "synthetic"


def _count_frames(bundle_dir: Path) -> int:
    """Count frame_*.txt files in a bundle directory."""
    if not bundle_dir.is_dir():
        return 0
    frames = sorted(bundle_dir.glob("frame_*.txt"))
    return len(frames)


def _is_sequence_bundle(p: Path) -> bool:
    """True if directory contains frame_*.txt (bending sequence)."""
    return p.is_dir() and any(p.glob("frame_*.txt"))


def _get_creation_time(p: Path) -> str | None:
    """Get oldest file mtime in dir as creation hint."""
    if not p.is_dir():
        return None
    times = []
    for f in p.rglob("*"):
        if f.is_file():
            try:
                times.append(f.stat().st_mtime)
            except OSError:
                pass
    if not times:
        return None
    return datetime.fromtimestamp(min(times)).strftime("%Y-%m-%d %H:%M")


def _scan_dataset(base: Path, ds_id: str, name: str) -> dict:
    """Scan one dataset directory. Returns analysis dict."""
    out = {
        "id": ds_id,
        "name": name,
        "path": str(base.relative_to(REPO)) if base.is_relative_to(REPO) else str(base),
        "total_sequences": 0,
        "frames_per_sequence": [],
        "frame_min": None,
        "frame_max": None,
        "frame_median": None,
        "scenarios": {},
        "creation_hint": None,
        "manifest": None,
        "structure": "",
    }
    if not base.exists():
        return out

    # Manifest
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            out["manifest"] = True
            out["total_sequences"] = m.get("total_count", len(m.get("entries", [])))
            entries = m.get("entries", [])
            if entries:
                scenarios = {}
                for e in entries:
                    s = e.get("scenario", e.get("label", "unknown"))
                    scenarios[s] = scenarios.get(s, 0) + 1
                out["scenarios"] = scenarios
                out["structure"] = f"manifest + {len(m.get('splits',{}))} splits"
        except Exception:
            out["manifest"] = "parse_error"

    # Direct scan for sequence bundles (scenario/subdir with frame_*.txt)
    scenario_names = ("normal", "crack", "pre_damaged", "thick_panel", "uv_overcured",
                      "over_bending", "under_bending", "jig_vibration", "micro_crack",
                      "light_distortion", "crack_in_bending", "normal_light_distortion")
    sequences = []
    for sub in base.rglob("*"):
        if sub.is_dir() and _is_sequence_bundle(sub):
            parent_name = sub.parent.name
            if parent_name in scenario_names or "normal_" in str(sub.parent) or "crack" in str(sub.parent):
                n = _count_frames(sub)
                if n > 0:
                    sequences.append((sub, n))

    # Deduplicate: if we have manifest count, use it; else use scanned
    if not out["total_sequences"] and sequences:
        out["total_sequences"] = len(sequences)
        out["frames_per_sequence"] = [s[1] for s in sequences[:100]]  # sample
        if sequences:
            frames = [s[1] for s in sequences]
            out["frame_min"] = min(frames)
            out["frame_max"] = max(frames)
            out["frame_median"] = sorted(frames)[len(frames) // 2]
    elif sequences and out["total_sequences"]:
        frames = [s[1] for s in sequences[:500]]
        if frames:
            out["frame_min"] = min(frames)
            out["frame_max"] = max(frames)
            out["frame_median"] = sorted(frames)[len(frames) // 2]
        out["frames_per_sequence"] = frames[:50]

    # Creation hint
    out["creation_hint"] = _get_creation_time(base)

    # Structure from directory layout
    if not out["structure"]:
        subdirs = [d.name for d in base.iterdir() if d.is_dir()]
        if subdirs:
            out["structure"] = ", ".join(sorted(subdirs)[:8])
            if len(subdirs) > 8:
                out["structure"] += f" (+{len(subdirs)-8})"

    return out


def main() -> None:
    datasets = [
        (DATA / "ml_dataset_100k_v2", "DS-260220-ml-100k-100k-60f", "ml_dataset_100k_v2 (100k scale)"),
        (DATA / "ml_dataset_fp_focused", "DS-260223-ml-fp-20k-60f", "ml_dataset_fp_focused (20k, FP 개선)"),
    ]

    results = []
    for base, ds_id, name in datasets:
        r = _scan_dataset(base, ds_id, name)
        results.append(r)

    # Also try to read manifest via subprocess if permission denied
    for r in results:
        if r["manifest"] is None and r["path"]:
            p = REPO / r["path"] / "manifest.json"
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        m = json.load(f)
                    r["manifest"] = True
                    r["total_sequences"] = m.get("total_count", len(m.get("entries", [])))
                    r["scenarios"] = {}
                    for e in m.get("entries", []):
                        s = e.get("scenario", str(e.get("label", "?")))
                        r["scenarios"][s] = r["scenarios"].get(s, 0) + 1
                    r["structure"] = f"train/val/test = {m.get('splits',{})}"
                except Exception as ex:
                    r["manifest"] = str(ex)[:30]

    # Print table
    print("\n" + "=" * 120)
    print("BENDING 영상 데이터셋 분석 보고서")
    print("=" * 120)
    print(f"\n{'ID':<8} {'폴더':<35} {'전체수':>10} {'프레임/시퀀스':<18} {'생성시점':<18} {'구성':<25}")
    print("-" * 120)
    for r in results:
        if r["total_sequences"] == 0 and not (REPO / r["path"]).exists():
            continue
        frames_str = f"{r['frame_min'] or '-'}~{r['frame_max'] or '-'}" if r["frame_min"] is not None else "60"
        if r["frame_median"]:
            frames_str += f" (중앙:{r['frame_median']})"
        print(f"{r['id']:<8} {r['path']:<35} {r['total_sequences']:>10} {frames_str:<18} {(r['creation_hint'] or '-'):<18} {str(r['scenarios'])[:24]:<25}")
    print("-" * 120)

    # Detailed table
    print("\n\n[상세 표]")
    print("-" * 100)
    for r in results:
        if r["total_sequences"] == 0 and not (REPO / r["path"]).exists():
            continue
        print(f"\n{r['id']} | {r['name']}")
        print(f"  경로: {r['path']}")
        print(f"  전체 시퀀스 수: {r['total_sequences']}")
        print(f"  프레임/시퀀스: min={r['frame_min']}, max={r['frame_max']}, median={r['frame_median']}")
        print(f"  시나리오 구성: {r['scenarios']}")
        print(f"  생성 시점(추정): {r['creation_hint']}")
        print(f"  구조: {r['structure']}")
        print(f"  manifest: {r['manifest']}")

    # Save JSON
    out_path = REPO / "reports" / "bending_datasets_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": results, "generated": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    print(f"\n\nJSON 저장: {out_path}")


if __name__ == "__main__":
    main()
