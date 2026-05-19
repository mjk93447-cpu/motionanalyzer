# GUI ML end-to-end test checklist

Manual verification after Tracks A–D (DRAEM pipeline + Data tab).

## Prerequisites

- `pip install -e ".[ml]"`
- Trained weights in `%APPDATA%/motionanalyzer/models/` **or** `release/models/` (bundled copy on first GUI launch):
  - `draem_model.pt`, `patchcore_model.npz`, `bundle_manifest.json`

## Checklist

1. **Data tab — inbox scan**  
   Set inbox to `data/synthetic/...` or `data/raw`; click **Scan bundles**; tree lists bundles with frame counts.

2. **Import (optional)**  
   Run ingest on a small flat-TXT fixture; confirm `data/raw/<video>/<bundle>/frame_*.txt`.

3. **Analyze — physics + SI**  
   Open a synthetic bundle; set mm/px if needed; mode **physics**; **Run Analysis**; confirm vector map Y-axis matches image coordinates.

4. **Analyze — DRAEM**  
   Mode **draem**; confirm model status row shows Manifest OK; run analysis; `draem_anomaly_scores.csv` / plot under output dir.

5. **Analyze — ensemble**  
   Mode **ensemble**; enable **Dataset-level max**; compare qualitative TN/FP vs `reports/crack_detection_analysis/analysis.json` (both_agree).

6. **ML tab — retrain small**  
   Paper/CLI preset; train DRAEM + PatchCore on tiny manifest; refresh manifest; re-run Analyze **draem**.

7. **Automated**  
   `pytest tests/test_ml_bundle.py tests/test_ml_inference.py tests/test_ingest_edge_points.py`  
   `python scripts/run_gui_test_scenarios.py` (if configured)

## Regression

- `.\scripts\verify_draem_rename.ps1`
- Full `pytest`
