"""
Generate example synthetic datasets for users.

Creates example datasets in data/synthetic/examples/ for quick testing and learning.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle


def main() -> None:
    """Generate example datasets."""
    examples_dir = repo_root / "data" / "synthetic" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    print("Generating example synthetic datasets...")
    print(f"Output directory: {examples_dir}")
    print()

    scenarios = [
        ("normal", "정상 공정"),
        ("crack", "크랙 발생"),
        ("pre_damage", "사전 손상"),
        ("thick_panel", "두꺼운 패널"),
        ("uv_overcured", "UV 과경화"),
    ]

    for scenario, description in scenarios:
        output_dir = examples_dir / scenario
        print(f"Generating {scenario} ({description})...")

        config = SyntheticConfig(
            frames=120,
            points_per_frame=230,
            fps=30.0,
            seed=42,
            scenario=scenario,
        )

        generate_synthetic_bundle(output_dir, config)
        print(f"  [OK] Created: {output_dir}")
        print(f"    Files: frame_*.txt, fps.txt, frame_metrics.csv, metadata.json")
        print()

    print("Example datasets generation complete!")
    print()
    print("Usage examples:")
    print("  # Analyze normal case")
    print("  motionanalyzer analyze-bundle \\")
    print("    --input-dir data/synthetic/examples/normal \\")
    print("    --output-dir exports/vectors/example_normal")
    print()
    print("  # Compare normal vs crack")
    print("  motionanalyzer compare-runs \\")
    print("    --base-summary exports/vectors/example_normal/summary.json \\")
    print("    --candidate-summary exports/vectors/example_crack/summary.json")


if __name__ == "__main__":
    main()
