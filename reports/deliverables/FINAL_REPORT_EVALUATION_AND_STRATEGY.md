# FPCB Crack Detection — Final Report Evaluation & Submission Strategy

**Document Version**: 1.0  
**Date**: 2026-02-19  
**Purpose**: Comprehensive PPT evaluation and mid-to-long-term final deliverable strategy

---

## Part 1: Current PPT Comprehensive Evaluation

### 1.1 Overall Structure & Flow

| Criterion | Score (1–5) | Assessment |
|-----------|-------------|-------------|
| IMRaD alignment | 4 | Introduction, Methods, Results, Discussion, Appendix present; logical flow |
| Slide count balance | 3 | Methods section thin vs Results heavy; Appendix could be richer |
| Narrative coherence | 4 | Forward-reading flow works; some redundancy (dataset table twice) |

**Gaps**: Section dividers between IMRaD parts; executive summary slide missing; slide numbering inconsistent.

---

### 1.2 Typography & Text Layout

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| Font consistency | 4 | Arial used throughout; sizes vary (11–36pt) |
| Font hierarchy | 3 | Title 28pt, body 14pt — bullet text can overflow; no sub-headings |
| Line breaks & wrapping | 2 | Long bullets (e.g., Introduction) not wrapped; single-line textboxes overflow |
| Character spacing | 3 | Default; no explicit letter-spacing |
| Readability at distance | 3 | Body 14pt may be small for projection; minimum 16pt recommended |

**Gaps**: No line spacing control; bullets lack indentation levels; long sentences not split for readability.

---

### 1.3 Tables

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| Header styling | 4 | Samsung blue, white text; clear |
| Cell alignment | 4 | Center-aligned; consistent |
| Row height | 3 | Fixed 0.45"; dense for 7+ rows |
| Column width | 3 | Equal distribution; long text (e.g., "Ensemble (DREAM ∧ PatchCore)") cramped |
| Border/separator | 2 | Default thin borders; no zebra striping |
| Highlight logic | 4 | Green for improvement columns; clear |
| Professional formatting | 3 | No units in headers; no caption below table |

**Gaps**: Table captions missing; no "Source" or "Note"; scenario descriptions absent in dataset table; numeric alignment (right-align numbers) not applied.

---

### 1.4 Images

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| Size vs slide | 3 | Full width used; aspect ratio may distort |
| Resolution | ? | Depends on source; need to verify 300 DPI for print |
| Caption placement | 2 | Title above image only; no figure number (e.g., "Figure 1") |
| Consistency | 3 | Same margin; VS layout vs single-image layout differ |
| Border/frame | 2 | No frame; white bg blends with slide |

**Gaps**: No figure numbering (Fig. 1, Fig. 2); no scale bar or annotation; vector map interpretation not embedded in caption.

---

### 1.5 Videos

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| Embedding | 3 | add_movie used; poster fallback if fail |
| Size | 4 | Full content width; adequate |
| Caption/instruction | 2 | "Double-click to play" only when poster fallback; no runtime/duration |
| Placement in narrative | 4 | Methods (process log), Results (vector map, confusion matrix) — logical |
| Fallback when missing | 3 | Poster image shown; acceptable |

**Gaps**: No video duration label; no transcript/caption reference; external video files required (not embedded in some viewers).

---

### 1.6 Color Theme & Design

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| Samsung blue (#1428A0) | 4 | Applied to title slide, headers, table headers |
| Contrast | 4 | White bg, dark gray text; sufficient |
| Accent (green) | 4 | Improvement highlight; clear |
| Consistency | 4 | Same palette throughout |
| Visual hierarchy | 3 | Blue for headers; no section divider slides |

**Gaps**: No footer/logo; no slide number on content slides; no section divider slides (e.g., "2. Methods" full-slide divider).

---

### 1.7 Content Completeness

| Criterion | Score | Assessment |
|-----------|-------|-------------|
| All assets included | 3 | Images/videos conditional on file existence; insights_summary may be missing |
| Hard subset table | 2 | light_distortion 8/8, micro_crack 2/2 — mentioned in bullets, not in dedicated table |
| Confusion matrix raw | 2 | TN/FP/FN/TP numbers in insights.md; not in PPT table |
| FPCB domain context | 2 | Brief in Introduction; no dedicated "Background" depth |
| Limitations | 3 | In Discussion; could be more explicit |

**Gaps**: No ROC curve or precision-recall curve; no implementation change summary table; no references/citations slide.

---

### 1.8 Summary Scorecard

| Category | Avg Score | Priority |
|----------|-----------|----------|
| Structure & Flow | 3.7 | Medium |
| Typography | 3.0 | **High** |
| Tables | 3.3 | **High** |
| Images | 2.5 | **High** |
| Videos | 3.3 | Medium |
| Color & Design | 3.8 | Medium |
| Content | 2.4 | **High** |

**Overall**: 3.1 / 5 — Solid foundation; needs typography, table, image, and content refinements for "perfect" deliverable.

---

## Part 2: Paper-First Strategy (Reference for PPT)

### 2.1 Rationale

- **Paper as source of truth**: All text, figures, tables defined once; PPT derives from it.
- **Consistency**: Same terminology, numbers, captions in Word/PDF and PPT.
- **Quality gate**: Paper review catches logic/typo before PPT production.
- **Traceability**: PPT slides can reference "Section 3.2, Table 2" etc.

### 2.2 Paper Structure (IMRaD)

1. **Title, Authors, Abstract** (150–200 words)
2. **Introduction** (1–1.5 pages)
   - Background: FPCB bending, copper crack, production need
   - Problem: FP impact, baseline performance
   - Objectives: Precision ≥99%, FP minimization
   - Scope: Synthetic data, 2D surrogate
3. **Methods** (1.5–2 pages)
   - Dataset: Table 1 (scenario, count, label, description)
   - Models: DREAM, PatchCore, Ensemble
   - Development: 4 validation loops
   - Metrics: Precision, Recall, FP, confusion matrix
4. **Results** (2–2.5 pages)
   - Table 2: Performance overview (baseline vs final)
   - Table 3: Model comparison
   - Table 4: Development-validation loop
   - Table 5: Hard subset (light_distortion, micro_crack)
   - Figure 1: Vector map Normal vs Crack (side-by-side)
   - Figure 2: Confusion matrices (DREAM, PatchCore, Ensemble)
   - Figure 3: Ensemble confusion matrix (final)
5. **Discussion** (0.5–1 page)
   - Conclusions
   - Recommendations
   - Limitations
6. **References** (if any)
7. **Appendix**
   - Confusion matrix definitions
   - Full dataset composition
   - Implementation details

### 2.3 Paper Outputs

| Format | Tool | Output |
|--------|------|--------|
| Word | python-docx or manual | `FPCB_Crack_Detection_Final_Report.docx` |
| PDF | Word export or pandoc | `FPCB_Crack_Detection_Final_Report.pdf` |

---

## Part 3: Mid-to-Long-Term Final Deliverable Plan

### Phase 1: Paper Creation (Weeks 1–2)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 1.1 Draft paper in Word/LaTeX | `.docx` | Script or manual |
| 1.2 Insert all tables (numbered) | Table 1–5 | - |
| 1.3 Insert all figures (numbered) | Figure 1–3+ | - |
| 1.4 Write captions for each | - | - |
| 1.5 Export to PDF | `.pdf` | - |
| 1.6 Internal review | Feedback | - |

### Phase 2: Asset Consolidation (Week 2)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 2.1 Ensure all images exist | PNGs in `crack_detection_analysis/` | `analyze_crack_detection.py` |
| 2.2 Ensure all videos exist | MP4s in `deliverables/videos/` | `create_process_videos.py` |
| 2.3 Add missing graphs (e.g., ROC, P-R) | - | Optional |
| 2.4 High-res export (300 DPI) for print | - | If needed |

### Phase 3: PPT Derived from Paper (Weeks 2–3)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 3.1 Map paper sections → slides | Slide outline | - |
| 3.2 Apply typography standards | Min 16pt body, hierarchy | `generate_final_report_ppt.py` |
| 3.3 Table improvements | Captions, right-align numbers, zebra | - |
| 3.4 Image improvements | Figure numbers, captions, frames | - |
| 3.5 Video improvements | Duration label, caption | - |
| 3.6 Section divider slides | "2. Methods", "3. Results" full-slide | - |
| 3.7 Slide numbers, footer | All content slides | - |
| 3.8 Line break / text overflow fix | Wrapped bullets | - |

### Phase 4: Quality Assurance (Week 3)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 4.1 Cross-check paper vs PPT | Numbers, captions match | - |
| 4.2 Readability test | Projection simulation | - |
| 4.3 Video playback test | All 3 videos play | - |
| 4.4 Export PPT to PDF (backup) | - | Optional |

### Phase 5: Final Packaging (Week 4)

| Task | Deliverable | Owner |
|------|-------------|-------|
| 5.1 Package: Word + PDF + PPT | `deliverables/` folder | - |
| 5.2 Include: videos, images, graphs | All assets | - |
| 5.3 README for deliverables | Usage, structure | - |

---

## Part 4: Detailed Improvement Checklist

### 4.1 Typography

- [ ] Body text minimum 16pt for projection
- [ ] Title 28–32pt; section title 24–26pt
- [ ] Line spacing 1.2–1.5
- [ ] Bullet indentation: 2 levels max
- [ ] Long bullets: split into 2 lines or shorten
- [ ] Sub-headings for dense sections

### 4.2 Tables

- [ ] Table caption below (e.g., "Table 1. Synthetic dataset composition.")
- [ ] Right-align numeric columns
- [ ] Add "Description" column for dataset scenario
- [ ] Zebra striping (alternate row color) for 5+ rows
- [ ] Source/note line if needed

### 4.3 Images

- [ ] Figure number (Fig. 1, Fig. 2)
- [ ] Caption below image
- [ ] Consistent aspect ratio (no stretch)
- [ ] Optional: thin border/frame
- [ ] Verify 300 DPI for print export

### 4.4 Videos

- [ ] Duration label (e.g., "0:45")
- [ ] Caption: "Video 1. Analysis process log."
- [ ] Ensure poster frame shows key frame

### 4.5 Design

- [ ] Section divider slides (e.g., "2. Methods" — full slide, blue bg)
- [ ] Slide number on all content slides
- [ ] Footer: project name or date
- [ ] Remove duplicate content (e.g., dataset table once in Methods, once in Appendix — keep one, reference the other)

### 4.6 Content

- [ ] Hard subset as dedicated table
- [ ] Raw confusion matrix (TN, FP, FN, TP) as table
- [ ] Implementation summary table (file, change)
- [ ] References slide (if citing prior work)

---

## Part 5: Final Deliverables Summary

| # | Deliverable | Format | Location |
|---|-------------|--------|----------|
| 1 | Final Report (Paper) | Word | `deliverables/FPCB_Crack_Detection_Final_Report.docx` |
| 2 | Final Report (Paper) | PDF | `deliverables/FPCB_Crack_Detection_Final_Report.pdf` |
| 3 | Final Report (Presentation) | PPTX | `deliverables/FPCB_Crack_Detection_Final_Report.pptx` |
| 4 | Videos | MP4 | `deliverables/videos/*.mp4` |
| 5 | Images & Graphs | PNG | `deliverables/images/` or embedded |
| 6 | README | MD | `deliverables/README.md` |

**PPT completeness**: All text, tables, images, videos, graphs from paper + process visuals included; professional typography, layout, and design applied.

---

## Part 6: Implementation Scripts Roadmap

| Script | Purpose | Phase |
|--------|---------|-------|
| `generate_final_report_docx.py` | **NEW** — Generate Word from paper structure | Phase 1 |
| `generate_final_report_ppt.py` | **UPDATE** — Apply checklist improvements | Phase 3 |
| `create_process_videos.py` | Existing — Ensure videos | Phase 2 |
| `analyze_crack_detection.py` | Existing — Ensure images | Phase 2 |

---

*End of Evaluation & Strategy Document*
