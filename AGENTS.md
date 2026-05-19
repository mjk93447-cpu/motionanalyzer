# AGENTS: AI Retrieval Contract

This repository is organized for fast AI-assisted development.

## Read Order (Always)

1. `docs/AGENT_HANDOFF_QUICK_START.md` — **에이전트 핸드오프 시 우선**
2. `docs/INDEX.md`
3. `docs/PROJECT_GOALS.md`
4. `docs/DEVELOPMENT_ROADMAP_FINAL.md`
5. `docs/PHASE_B_INSIGHTS.md`
6. `reports/INDEX.md`
7. `indexes/corpus-index.json`

## Agent communication

- **Token efficiency**: caveman **lite** in chat (see `.cursor/rules/caveman-default.mdc`). Code/commits/docs: normal prose.

## Canonical Sources

- **DRAEM model (naming + implementation)**: `docs/DRAEM_REFERENCE.md` — use **DRAEM** only; never `DREAM` / `dream` in new code or docs.
- **Handoff**: `docs/AGENT_HANDOFF_QUICK_START.md` (즉시 실행 가이드)
- Goals: `docs/PROJECT_GOALS.md`
- Plan: `docs/DEVELOPMENT_ROADMAP_FINAL.md`
- Synthetic data contract: `docs/SYNTHETIC_DATA_SPEC.md`
- CPD method: `docs/CHANGEPOINT_DETECTION.md`
- ML modeling checklist: `docs/AI_MODELING_PLAYBOOK.md`
- Operational pipeline: `docs/PIPELINE_SETUP_COMPLETE.md`
- MCP setup: `docs/MCP_SETUP.md`
- Dataset naming rule: `docs/DATASET_NAMING_RULE.md`
- **Dataset folder structure & paths**: `docs/DATASET_FOLDER_STRUCTURE.md`, `data/synthetic/README.md` (dataset_id, ml_*_60f)
- Analysis output naming: `docs/ANALYSIS_OUTPUT_NAMING.md` (run_id, analysis_{run_id}.json, run_meta)
- Dataset inventory: `reports/BENDING_DATASETS_INVENTORY.md`
- Current report summary: `reports/crack_detection_analysis/insights.md` (최신 요약). 인사이트만: `reports/crack_detection_analysis/insights_canonical.md`. 과거 실행 리포트: `reports/crack_detection_analysis/archive/`
- **GitHub workflow**: `docs/GITHUB_WORKFLOW_COMPLETE.md`, `GITHUB_SETUP.md`, `scripts/git_backup.ps1`, `scripts/git_checkpoint.ps1`, `scripts/git_workflow.ps1`

## Archive Policy

- Historical or overlapping documents move to `docs/archive/`.
- Large generated artifacts move to `artifacts/archive/`.
- `reports/` keeps canonical summaries and links, not bulky binaries.

## Compatibility Policy

- Keep CLI command names stable.
- If file moves are required, add migration mapping in `MIGRATION_GUIDE.md`.
