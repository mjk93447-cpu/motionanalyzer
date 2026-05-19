# Balanced synthetic pretrain (P~0.9, R~0.7)

## Dataset

- **ID**: `ml_pretrain_balanced_3k_60f`
- **Scale**: `pretrain_balanced` (~2,680 bundles)
- **Split**: train **70%** / val **20%** / test **10%** (7:2:1)
- **NG diversity**: normal, light_distortion, crack, uv_overcured, micro_crack, thick_panel, over/under_bending, jig_vibration

Generate:

```powershell
python scripts/generate_ml_dataset.py --scale pretrain_balanced --split-train 0.7 --split-val 0.2 --split-test 0.1 --workers 4
```

## Train (~30 min on typical dev CPU)

```powershell
pip install -e ".[ml]"
python scripts/train_balanced_pretrain.py
# or one-shot:
python scripts/train_balanced_pretrain.py --generate
```

Outputs:

- `release/models/draem_model.pt`, `patchcore_model.npz`, `bundle_manifest.json`
- `reports/balanced_pretrain_metrics.json`

Thresholds are tuned on **validation** at **dataset level** (max score per bundle) toward **precision≈0.9, recall≈0.7**, not recall=1.

## Real-data refine (GUI)

1. Ingest real normals → `data/real/ml_<name>/manifest.json`
2. ML tab → **Refine pretrained (real normal)** → Prepare Data → Run PatchCore
3. Analyze: `patchcore` / `ensemble` (both_agree)

See [PATCHCORE_REALDATA_FINETUNE.md](PATCHCORE_REALDATA_FINETUNE.md).
