# Migration Guide (vNext Refactor)

This guide maps legacy paths to the new AI-first layout.

## Backup Baseline

- Backup branch: `backup/v0.2.0-pre-refactor`
- Backup tag: `v0.2.0-backup`
- Snapshot manifest: `backup-manifest.json`

## Documentation Moves

| Previous Path | New Path |
|---|---|
| `docs/CURRENT_STATUS.md` | `docs/archive/2026-02/CURRENT_STATUS.md` |
| `docs/IMPLEMENTATION_STATUS.md` | `docs/archive/2026-02/IMPLEMENTATION_STATUS.md` |
| `docs/PROGRESS_SUMMARY.md` | `docs/archive/2026-02/PROGRESS_SUMMARY.md` |
| `docs/VER7_DEV_PLAN.md` | `docs/archive/2026-02/VER7_DEV_PLAN.md` |
| `docs/DEVELOPMENT_PRIORITIES.md` | `docs/archive/2026-02/DEVELOPMENT_PRIORITIES.md` |
| `docs/DEVELOPMENT_PRIORITIES_SYNTHETIC_FIRST.md` | `docs/archive/2026-02/DEVELOPMENT_PRIORITIES_SYNTHETIC_FIRST.md` |
| `docs/UNIFIED_DEVELOPMENT_PLAN.md` | `docs/archive/2026-02/UNIFIED_DEVELOPMENT_PLAN.md` |
| `docs/DEVELOPMENT_STRATEGY_AND_WORK_ORDER.md` | `docs/archive/2026-02/DEVELOPMENT_STRATEGY_AND_WORK_ORDER.md` |
| `docs/JUPYTER_GPU_SETUP.md` | `docs/archive/2026-02/JUPYTER_GPU_SETUP.md` |
| `docs/ML_GPU_ACCELERATION_STRATEGY.md` | `docs/archive/2026-02/ML_GPU_ACCELERATION_STRATEGY.md` |

## Report Moves

| Previous Path | New Path |
|---|---|
| `reports/deliverables/CRACK_DETECTION_FINAL_REPORT.md` | `reports/archive/2026-02/CRACK_DETECTION_FINAL_REPORT.deliverables.md` |
| `reports/DETECTION_IMPROVEMENT_STRATEGY.md` | `reports/archive/2026-02/DETECTION_IMPROVEMENT_STRATEGY.md` |
| `reports/PRECISION_99_DEVELOPMENT_STRATEGY.md` | `reports/archive/2026-02/PRECISION_99_DEVELOPMENT_STRATEGY.md` |
| `reports/precision_progress_log.md` | `reports/archive/2026-02/precision_progress_log.md` |

## Script Moves

| Previous Path | New Path |
|---|---|
| `scripts/benchmark_phase2_ml.py` | `scripts/archive/2026-02/benchmark_phase2_ml.py` |
| `scripts/generate_fpcb_test_dataset.py` | `scripts/archive/2026-02/generate_fpcb_test_dataset.py` |
| `scripts/redraw_vector_map.py` | `scripts/archive/2026-02/redraw_vector_map.py` |
| `scripts/test_visualization.py` | `scripts/archive/2026-02/test_visualization.py` |

## API/Orchestration Changes

- New public service API for training wrappers:
  - `src/motionanalyzer/services/ml_training.py`
- Scripts now import from public service wrappers instead of private GUI runner symbols.
- `scripts/run_fpcb_pipeline.py` is now a compatibility wrapper that delegates to CLI logic.

## New Entry Points

- `AGENTS.md` - AI retrieval contract
- `docs/INDEX.md` - canonical documentation entry
- `reports/INDEX.md` - canonical reporting entry
- `indexes/corpus-index.json` - machine-readable retrieval map

## Recommended Retrieval Sequence

1. `AGENTS.md`
2. `docs/INDEX.md`
3. `docs/PROJECT_GOALS.md`
4. `docs/DEVELOPMENT_ROADMAP_FINAL.md`
5. `docs/PHASE_B_INSIGHTS.md`
6. `reports/INDEX.md`
7. `indexes/corpus-index.json`
