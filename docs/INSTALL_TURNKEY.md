# Turnkey install (EXE + pretrained models)

**Balanced synthetic pretrain** (`ml_pretrain_balanced_3k_60f`, 7:2:1) is bundled for Analyze/ML.  
**Real-data refine** on the factory LAN only: [INTERNAL_DEPLOYMENT_WORKFLOW.md](INTERNAL_DEPLOYMENT_WORKFLOW.md).

## GitHub Actions artifacts (recommended)

| Artifact | Contents |
|----------|----------|
| `motionanalyzer-turnkey` | `motionanalyzer-gui-ml.exe` + `models/` + install docs (single zip) |
| `motionanalyzer-pretrained-models` | `release/models/` + metrics JSON |
| `motionanalyzer-windows-exe` | EXE + zips |

## Build ML EXE with bundled weights (dev)

```powershell
pip install -e ".[ml,build]"
# Copy trained models to release/models (from APPDATA or training output)
.\scripts\export_release_model_bundle.ps1
.\scripts\build_exe.ps1 -IncludeML -BundleModels
```

Output: `dist/motionanalyzer-gui-ml.exe`

## First run

On launch, the GUI copies missing files from `_MEIPASS/models` to `%APPDATA%/motionanalyzer/models/`.

Required for ML Analyze modes:

- `draem_model.pt`
- `patchcore_model.npz`
- `bundle_manifest.json`

## Path overrides (other PCs / custom layout)

| Variable | Effect |
|----------|--------|
| `MOTIONANALYZER_MODELS_DIR` | Directory containing `draem_model.pt`, `patchcore_model.npz`, `bundle_manifest.json` |
| `MOTIONANALYZER_APP_DIR` | User app root (default `%APPDATA%/motionanalyzer`) |

Verify after install:

```powershell
.\scripts\verify_turnkey_setup.ps1 -ModelsDir release\models
```

PatchCore real-data refinement: [PATCHCORE_REALDATA_FINETUNE.md](PATCHCORE_REALDATA_FINETUNE.md)

## GitHub Release (manual)

1. Upload `motionanalyzer-gui-ml.exe`
2. Upload `dist/motionanalyzer-models.zip` from `export_release_model_bundle.ps1`
3. Point users to this doc and [GUI_ML_E2E_TEST_CHECKLIST.md](GUI_ML_E2E_TEST_CHECKLIST.md)
