# Bundled pretrained models (balanced pretrain)

Canonical turnkey weights for GUI / PyInstaller `-BundleModels`:

| File | Role |
|------|------|
| `draem_model.pt` | DRAEM checkpoint |
| `patchcore_model.npz` | PatchCore memory bank |
| `bundle_manifest.json` | Features, norm stats, thresholds (`both_agree`) |

Trained with `scripts/train_balanced_pretrain.py` on `ml_pretrain_balanced_3k_60f`.  
Meta: `release/TURNKEY_BUNDLE.json`, metrics: `reports/balanced_pretrain_metrics.json`.

On first GUI launch, files copy to `%APPDATA%/motionanalyzer/models/` if missing.

Regenerate locally:

```powershell
python scripts/train_balanced_pretrain.py
.\scripts\build_turnkey_release.ps1 -SkipTrain
```
