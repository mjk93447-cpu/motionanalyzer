# Reports Index

Canonical report entrypoints for ongoing development.

## Canonical

- `CRACK_DETECTION_FINAL_REPORT.md` - primary integrated report
- `IMRAD_SUMMARY.md` - paper-style IMRAD summary (Introduction, Methods, Results, Discussion)
- `DATASET_AND_EXPERIMENT_SPEC.md` - train/val/test 구분, 데이터 규모, 실험별 결과 명세
- `REPORT_DATA_RECONCILIATION.md` - **보고서–실제 데이터 관계 분석**, 정합성 검토, 실제 기반 결과
- `PAPER_FPCB_CRACK_DETECTION.md` - **논문 초안** (10k/30k, Precision 99%+, edge_scorch)
- `research_validation_report.md` - **연구 방법론 검증** (데이터, 태그, 판정 기준, 산출물)
- `goal_achievement_summary.md` - concise goal achievement view
- `crack_detection_analysis/analysis.json` - machine-readable latest metrics
- `crack_detection_analysis/insights.md` - latest analysis narrative

## Historical / Legacy

- `deliverables/` contains frozen deliverables.
- superseded report narratives are archived under `reports/archive/`.

## Retrieval Rule

When you need current performance/status, read in this order:

1. `crack_detection_analysis/analysis.json`
2. `crack_detection_analysis/insights.md`
3. `goal_achievement_summary.md`
4. `CRACK_DETECTION_FINAL_REPORT.md`
