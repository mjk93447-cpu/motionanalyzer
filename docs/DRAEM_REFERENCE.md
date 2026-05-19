# DRAEM — canonical model reference

Single name **DRAEM** in code, UI, JSON, and docs (ICCV 2021). Do not use legacy `DRAEM` / `draem`.

## Paper

- **Title**: DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection
- **Authors**: Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj
- **Venue**: ICCV 2021, pp. 8330–8339
- **arXiv**: [2108.07610](https://arxiv.org/abs/2108.07610)
- **Reference code**: [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)

## Strategy (FPCB tabular)

1. **Reconstructive**: map inputs toward normal reconstruction.
2. **Discriminative**: classify (input, reconstruction) as normal vs anomaly.
3. **Training**: normal data only; synthetic anomalies for the discriminative head.
4. **Inference**: reconstruction error combined with discriminator output.

## This repository

| Item | Location |
|------|----------|
| Implementation | `src/motionanalyzer/ml_models/draem.py` — `DRAEMPyTorch`, `DRAEMAnomalyDetector` |
| Default weights | `%APPDATA%/motionanalyzer/models/draem_model.pt` (Windows) |
| GUI mode | Analyze / ML: `draem` |
| JSON metrics key | `"DRAEM"` in `analysis.json` |
| Validation script | `scripts/validate_draem_synthetic.py` |

## Legacy migration

- Old file `draem_model.pt` is loaded with a warning if `draem_model.pt` is missing. See `MIGRATION_GUIDE.md`.

## Related docs

- `docs/DRAEM_CRACK_LIKE_ANOMALY.md` — synthetic NG / crack-like features
- `docs/DRAEM_FEWSHOT_REAL_STRATEGY.md` — few-shot real crack strategy
- `docs/REFERENCE_PAPERS_FOR_PAPER_WRITING.md` — citations
