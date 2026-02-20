"""
Run physics-based FPCB bending pipeline: generate -> analyze -> export images.
Execute from repo root: python scripts/run_fpcb_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path when run from repo root or scripts/
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from motionanalyzer.analysis import run_analysis
from motionanalyzer.synthetic import generate_synthetic_bundle, high_fidelity_fpcb_config
from motionanalyzer.visualization import plot_frame_metrics


def main() -> None:
    base = _repo
    data_dir = base / "data" / "synthetic" / "fpcb_high_fidelity"
    export_vectors_dir = base / "exports" / "vectors" / "fpcb_high_fidelity"
    plots_dir = base / "exports" / "plots"

    for d in (base / "data" / "synthetic", base / "exports" / "vectors", base / "exports" / "plots"):
        d.mkdir(parents=True, exist_ok=True)

    print("1/3 Generating high-fidelity FPCB bending dataset (~2 s bend, physics-based)...")
    config = high_fidelity_fpcb_config(points_per_frame=280, fps=30.0, bend_duration_s=2.0, seed=42)
    generate_synthetic_bundle(output_dir=data_dir, config=config)
    print(f"   Generated {data_dir}")

    print("2/3 Running motion analysis (vectors, summary, vector_map.png)...")
    summary = run_analysis(input_dir=data_dir, output_dir=export_vectors_dir)
    print(f"   Analysis done: frames={summary.frame_count}, mean_speed={summary.mean_speed:.3f} px/s")

    print("3/3 Plotting frame metrics...")
    metrics_csv = data_dir / "frame_metrics.csv"
    metrics_plot = plots_dir / "fpcb_metrics.png"
    plot_frame_metrics(metrics_csv, metrics_plot)
    print(f"   Saved {metrics_plot}")

    vector_map = export_vectors_dir / "vector_map.png"
    print("\nGenerated images:")
    print(f"  Vector map: {vector_map}")
    print(f"  Metrics:    {metrics_plot}")


if __name__ == "__main__":
    main()
