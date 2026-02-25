# FPCB Crack Detection — Final Deliverables

## Deliverables List

| File | Description |
|------|-------------|
| **FPCB_Crack_Detection_Final_Report.docx** | Final report (Word, IMRaD, IEEE-style layout) |
| **FPCB_Crack_Detection_Final_Report.pdf** | Final report (PDF, auto-generated) |
| **FPCB_Crack_Detection_Final_Report.pptx** | Final report PPT (Samsung theme, section dividers) |
| videos/01_vector_map_visualization.mp4 | Vector map visualization |
| videos/02_analysis_process_log.mp4 | Analysis process log |
| videos/03_confusion_matrix_results.mp4 | Confusion matrix results |
| FINAL_REPORT_EVALUATION_AND_STRATEGY.md | Evaluation & submission strategy |
| FPCB_Crack_Detection_Final_Report.docx | 최종 보고서 (IMRaD) |

## Paper-First Strategy

1. **Paper (Word/PDF)** — Source of truth; tables/figures numbered; side-by-side layout.
2. **PPT** — Derived from paper; section dividers; min 16pt body; Hard subset, Future Work.

## 논문 작성 전 (레거시 아카이빙)

기존 산출물이 있으면 **신규 논문 작성 전** 아카이빙하여 수치·그림 혼용을 막습니다.

```powershell
.\scripts\archive_legacy_deliverables.ps1
```

→ `reports/archive/legacy_deliverables/` 에 보관 (참고 전용). 모든 수치는 최신 `analysis.json` 에서만 사용.

## Generation Commands

```powershell
# One-shot pipeline (analysis → Word → PPT; optional Paper Banana)
.\scripts\run_paper_pipeline.ps1

# Skip re-analysis, only Word + PPT
.\scripts\run_paper_pipeline.ps1 -SkipAnalysis -SkipPaperBanana

# Include 100k inference and Paper Banana figures (requires: pip install paperbanana, paperbanana setup)
.\scripts\run_paper_pipeline.ps1 -Run100k
```

```bash
# Manual steps
pip install python-docx docx2pdf
python scripts/generate_final_report_docx.py
python scripts/generate_final_report_ppt.py
# Videos (optional): pip install moviepy; python scripts/create_process_videos.py
```

**Full plans:**
- [PAPER_WRITING_AND_PAPERBANANA_PLAN.md](../docs/PAPER_WRITING_AND_PAPERBANANA_PLAN.md) — Paper Banana, 100k, one-shot flow
- [RESEARCH_AND_PAPER_MASTER_PLAN.md](../docs/RESEARCH_AND_PAPER_MASTER_PLAN.md) — 개발 루프 3회+, 연구 완료 기준, 논문 순서

## Final Results

- **Precision**: 100% (Ensemble)
- **Normal FP Rate**: 0.00% (FP=0, target ≤0.1% achieved)
- **Recall**: 98.04%
- **light_distortion (normal)**: 100% correct (50/50)
- **micro_crack**: 100% correct (10/10)
