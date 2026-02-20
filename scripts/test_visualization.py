"""Test script to generate vector map visualization from synthetic data."""

from pathlib import Path

from motionanalyzer.analysis import run_analysis
from motionanalyzer.synthetic import SyntheticConfig, generate_synthetic_bundle
from motionanalyzer.visualization import plot_vector_map_by_frame, plot_vector_map_by_index

if __name__ == "__main__":
    # Create test data
    test_dir = Path("test_vis_output")
    test_dir.mkdir(exist_ok=True)

    input_dir = test_dir / "synthetic_input"
    output_dir = test_dir / "analysis_output"
    vis_dir = test_dir / "visualizations"

    print("Generating synthetic test data...")
    config = SyntheticConfig(
        frames=30,  # Smaller for faster test
        points_per_frame=50,
        fps=30.0,
        scenario="normal",
        seed=42,
    )
    generate_synthetic_bundle(output_dir=input_dir, config=config)
    print(f"Generated test data in: {input_dir}")

    print("Running analysis...")
    summary = run_analysis(input_dir=input_dir, output_dir=output_dir, fps=30.0)
    print(f"Analysis complete. Frames: {summary.frame_count}, Points: {summary.unique_index_count}")

    vectors_csv = output_dir / "vectors.csv"
    if not vectors_csv.exists():
        raise FileNotFoundError(f"vectors.csv not found at {vectors_csv}")

    print("Generating vector map visualizations...")

    # Plot colored by index
    vis_by_index = vis_dir / "vector_map_by_index.png"
    plot_vector_map_by_index(
        vectors_csv=vectors_csv,
        output_image=vis_by_index,
        scale=0.1,  # Small arrows
        width=0.001,
        alpha=0.7,
        point_size=0.3,
    )
    print(f"Saved: {vis_by_index}")

    # Plot colored by frame
    vis_by_frame = vis_dir / "vector_map_by_frame.png"
    plot_vector_map_by_frame(
        vectors_csv=vectors_csv,
        output_image=vis_by_frame,
        scale=0.1,
        width=0.001,
        alpha=0.7,
        point_size=0.3,
    )
    print(f"Saved: {vis_by_frame}")

    print("\nVisualization test complete!")
    print(f"Check images in: {vis_dir}")
