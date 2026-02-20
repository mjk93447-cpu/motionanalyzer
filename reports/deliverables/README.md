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
| CRACK_DETECTION_FINAL_REPORT.md | Consolidated development report |

## Paper-First Strategy

1. **Paper (Word/PDF)** — Source of truth; tables/figures numbered; side-by-side layout.
2. **PPT** — Derived from paper; section dividers; min 16pt body; Hard subset, Future Work.

## Generation Commands

```bash
# Paper (Word + PDF)
pip install python-docx docx2pdf
python scripts/generate_final_report_docx.py

# Videos (optional)
pip install moviepy
python scripts/create_process_videos.py

# PPT
python scripts/generate_final_report_ppt.py
```

## Final Results

- **Precision**: 100% (Ensemble)
- **False Positive**: 0
- **light_distortion (normal)**: 100% correct (8/8)
