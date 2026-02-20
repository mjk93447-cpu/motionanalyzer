"""
Evaluate Goal 1 (bending-in-process crack) via Change Point Detection.

Compares detected change point frame vs ground-truth crack_frame from metadata.
Output: reports/goal1_cpd_evaluation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.analysis import load_bundle, compute_vectors
from motionanalyzer.crack_model import load_frame_metrics
from motionanalyzer.time_series.changepoint import CUSUMDetector, WindowBasedDetector

BASE = repo_root / "data" / "synthetic" / "ml_dataset"
REPORTS = repo_root / "reports"


def _load_metadata(dataset_path: Path) -> dict:
    meta = dataset_path / "metadata.json"
    if not meta.exists():
        return {}
    return json.loads(meta.read_text(encoding="utf-8"))


def _run_cpd(input_dir: Path, fps: float = 30.0) -> list[int]:
    """Run CPD on acceleration_max, return detected frame indices."""
    try:
        df, fps_val, m_per_px = load_bundle(input_dir=input_dir, fps=fps)
        vectors = compute_vectors(df=df, fps=fps_val, meters_per_pixel=m_per_px)
        frame_metrics = load_frame_metrics(input_dir / "frame_metrics.csv")

        if "acceleration" in vectors.columns:
            acc_by_frame = vectors.groupby("frame")["acceleration"].max()
        else:
            acc_by_frame = frame_metrics.groupby("frame")["est_max_strain"].max()
        frames_sorted = sorted(acc_by_frame.index.astype(int))
        feature = acc_by_frame.reindex(frames_sorted).fillna(0).to_numpy(dtype=float)
        if len(feature) < 5:
            return []

        cusum = CUSUMDetector(threshold=3.0)
        result = cusum.detect(feature)
        if result.change_points:
            return [int(frames_sorted[i]) for i in result.change_points if i < len(frames_sorted)]
        window_d = WindowBasedDetector(window_size=10, threshold_ratio=2.0)
        result_w = window_d.detect(feature)
        return [int(frames_sorted[i]) for i in result_w.change_points if i < len(frames_sorted)]
    except Exception:
        return []


def main() -> None:
    manifest_path = BASE / "manifest.json"
    if not manifest_path.exists():
        print("Run: python scripts/generate_ml_dataset.py --small  (or full)")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    goal1_entries = [e for e in manifest["entries"] if e.get("goal") == "goal1"]
    if not goal1_entries:
        print("No goal1 entries in manifest.")
        sys.exit(0)

    errors: list[float] = []
    results: list[dict] = []

    for entry in goal1_entries:
        path = BASE / entry["path"]
        meta = _load_metadata(path)
        crack_frame = meta.get("crack_frame", -1)
        if crack_frame < 0:
            continue

        detected = _run_cpd(path)
        best = min(detected, key=lambda d: abs(d - crack_frame)) if detected else -1
        err = abs(best - crack_frame) if best >= 0 else 999
        errors.append(err)
        results.append({
            "path": entry["path"],
            "crack_frame": crack_frame,
            "detected": detected,
            "best_match": best,
            "error_frames": err,
        })

    mean_err = sum(errors) / len(errors) if errors else 999
    within_5 = sum(1 for e in errors if e <= 5) / len(errors) * 100 if errors else 0

    out = {
        "goal": "goal1",
        "metric": "CPD_accuracy",
        "n_evaluated": len(results),
        "mean_error_frames": round(mean_err, 2),
        "within_5_frames_pct": round(within_5, 1),
        "results": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "goal1_cpd_evaluation.json").write_text(
        json.dumps(out, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Goal 1 CPD: mean_error={mean_err:.1f} frames, within_5={within_5:.1f}%")
    print(f"  Saved: {REPORTS / 'goal1_cpd_evaluation.json'}")


if __name__ == "__main__":
    main()
