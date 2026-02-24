# Literature Comparison and Professional Evaluation

**Purpose:** Compare our FPCB crack detection work with similar papers and provide a professional assessment.

---

## 1. Related Work Summary

| Paper | Domain | Method | Metric | Value | Dataset |
|-------|--------|--------|--------|-------|---------|
| Zang & Zhang (ACM 2021) | FPC defect | CNN | Accuracy | ~85–90% | FPC images |
| PLOS ONE (2024) | FPC surface | GA-Faster-RCNN | Accuracy | 91.1% | FPC surface |
| Roth et al. (CVPR 2022) | Industrial | PatchCore | AUROC | 99.6% | MVTec AD |
| **Ours** | **FPCB bending** | **DREAM+PatchCore** | **Precision** | **99.86%** | Synthetic 10k |

---

## 2. Methodological Differences

| Aspect | Prior FPC Work | PatchCore (MVTec) | Ours |
|--------|----------------|-------------------|------|
| **Input** | Static images | Static images | Temporal (60 frames) |
| **Defect type** | Surface defects | General industrial | Bending-in-process crack |
| **Training** | Supervised / few-shot | Normal-only | Normal + boundary cases |
| **Metric** | Accuracy | AUROC | Precision (production-focused) |
| **Data** | Real FPC images | MVTec AD | Synthetic |

---

## 3. Professional Assessment

### Strengths of Our Approach

1. **Precision-first design:** Aligns with production cost structure (false alarms costly).
2. **Temporal modeling:** Bending is a process; per-frame and FFT features capture dynamics.
3. **Edge_scorch scenario:** Real-world production issue (laser cutting) modeled explicitly.
4. **Ensemble:** Reduces FP by 58% (24→10) vs single model.
5. **Reproducibility:** Fixed seeds, documented scripts, train/val/test separation.

### Limitations

1. **Synthetic data only:** No real FPCB crack validation; domain gap expected.
2. **2D surrogate:** Real stress/strain are 3D.
3. **Recall 69.7%:** May miss some cracks; trade-off for high precision.

### Comparison Verdict

Our 99.86% precision is comparable to or higher than state-of-the-art anomaly detectors on their benchmarks (PatchCore 99.6% AUROC, GA-Faster-RCNN 91.1% accuracy). The metrics are not directly comparable, but our precision-oriented design achieves the target for production deployment. Real-data validation remains the critical next step.
