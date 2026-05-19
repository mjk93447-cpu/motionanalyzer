# PatchCore real-data refinement

Last updated: 2026-05-19

## Purpose

Extend a **synthetic pretrained** PatchCore memory bank with **real normal** bending bundles, then recalibrate the anomaly threshold—without adding crack samples to the bank.

## Workflow (GUI)

1. **Data** tab: ingest real edge TXT → `data/raw/...` → preflight → **Build manifest** → `data/real/ml_<name>/manifest.json`
2. **ML** tab: set manifest path → **Load manifest normals** (or Data tab → **PatchCore refine (ML tab)**)
3. Select **Refine pretrained (real normal)** under PatchCore training
4. **Paper/CLI preset** → **Prepare Data** → **Run** with mode **PatchCore**
5. **Evaluate PatchCore** on prepared data (optional confusion metrics)
6. **Analyze** tab: mode `patchcore` or `ensemble` (requires `bundle_manifest.json`)

## API

```python
model.load("patchcore_model.npz")
model.fit_incremental(normal_features_df, source_tag="real_normal_refine")
model.refit_threshold(validation_normal_df, percentile=95.0)
model.save("patchcore_model.npz")
```

## Constraints

- `feature_cols` / dimension must match `bundle_manifest.json` from pretraining
- Crack rows are used only for **evaluation**, not memory bank updates
- Cross-frame index issues: fix via ingest + preflight ([REALDATA_EDGE_INGESTION.md](REALDATA_EDGE_INGESTION.md))

## CLI

```powershell
python scripts/train_release_bundle.py --out-dir release/models
python scripts/benchmark_ml_models.py --models-dir release/models
```
