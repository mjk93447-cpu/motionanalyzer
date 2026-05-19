# Real-data edge point ingestion

Last updated: 2026-05-19

## Purpose

Convert extractor output (thousands of flat `*.txt` files) into MotionAnalyzer **bundle** layout so `load_bundle`, preflight, `run_analysis`, and ML manifests work the same as synthetic data.

## Directory layout

```text
data/raw/
  <video_id>/                 # camera / lot / session
    <bundle_id>/              # one bending run
      fps.txt
      frame_00000.txt ...     # # x,y,index (integer); fixed index set per frame
      metadata.json           # optional: meters_per_pixel, label, scenario
data/real/ml_<dataset_id>/
  manifest.json               # entries[].path, split, label, goal
exports/vectors/<bundle_slug>/
reports/preflight/<batch_id>.json
```

See also [DATASET_FOLDER_STRUCTURE.md](DATASET_FOLDER_STRUCTURE.md).

## Ingest workflow

1. Place flat TXT under a staging folder (or `data/raw/` after partial ingest).
2. Run ingest:

   ```powershell
   python scripts/ingest_edge_points.py --source path/to/flat --output data/raw --fps 30
   ```

3. Validate bundles:

   ```powershell
   python scripts/validate_bundle_preflight.py --root data/raw --report reports/preflight/batch.json
   ```

4. Build ML manifest:

   ```powershell
   python scripts/build_manifest_from_bundles.py --root data/raw --output data/real/ml_my_set/manifest.json
   ```

5. In the GUI **Data** tab: scan inbox, preflight selected, **Open in Analyze**, or train from ML tab using the manifest path.

## Index / tracking policy

| Case | Handling |
|------|----------|
| Extractor provides `index` | Use as-is |
| No index | Sort points per frame, assign `1..N`; set `metadata.json` → `"synthetic_index": true` |
| Cross-frame index mismatch | Preflight fails; fix in ingest or re-track |

DRAEM temporal features are weaker when `synthetic_index` is true; document this in production reports.

## GUI

- **Data** tab: inbox path (`data/raw` default), bundle tree, **Preflight selected**, **Run ingest script…**, **Build manifest…**, **Open in Analyze**.

## References

- [preflight.py](../src/motionanalyzer/preflight.py) — bundle validation
- [analysis.py](../src/motionanalyzer/analysis.py) — `load_bundle`
- [ANALYSIS_OUTPUT_NAMING.md](ANALYSIS_OUTPUT_NAMING.md)
