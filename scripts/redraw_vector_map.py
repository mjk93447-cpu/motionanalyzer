"""Re-draw vector map from existing vectors.csv and summary.json. Run from repo root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from motionanalyzer.visualization import plot_full_vector_map

def main() -> None:
    out_dir = _repo / "exports" / "vectors" / "fpcb_high_fidelity"
    csv_path = out_dir / "vectors.csv"
    summary_path = out_dir / "summary.json"
    image_path = out_dir / "vector_map.png"
    if not csv_path.exists() or not summary_path.exists():
        print("Run full pipeline first: python scripts/run_fpcb_pipeline.py")
        sys.exit(1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fps = float(summary["fps"])
    mpp = summary.get("meters_per_pixel")
    plot_full_vector_map(csv_path, image_path, fps=fps, meters_per_pixel=mpp)
    print(f"Saved {image_path}")

if __name__ == "__main__":
    main()
