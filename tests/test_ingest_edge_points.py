"""Tests for flat edge TXT ingest into bundle layout."""

from __future__ import annotations

from pathlib import Path

from scripts.ingest_edge_points import ingest


def test_ingest_flat_txt_to_bundle(tmp_path: Path) -> None:
    src = tmp_path / "flat"
    src.mkdir()
    (src / "camA_frame_0.txt").write_text("# x,y,index\n0,0,1\n5,0,2\n", encoding="utf-8")
    (src / "camA_frame_1.txt").write_text("1,1,1\n6,1,2\n", encoding="utf-8")
    out = tmp_path / "data" / "raw"
    stats = ingest(src, out, video_pattern=r"(.+?)_frame_(\d+)", default_fps=25.0)
    assert stats.get("bundles", 0) >= 1
    bundle = out / "camA" / "run_001"
    assert (bundle / "fps.txt").exists()
    assert (bundle / "frame_00000.txt").exists()
    assert (bundle / "frame_00001.txt").exists()
    text = (bundle / "frame_00000.txt").read_text(encoding="utf-8")
    assert "0,0,1" in text
