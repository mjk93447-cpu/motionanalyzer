# AI-Based Crack Detection in FPCB Bending Process: Achieving 99%+ Precision on Synthetic Data

*Draft manuscript*  
*Last revised: February 2026*

---

## Abstract

Copper wire cracking during the FPCB (Flexible Printed Circuit Board) bending process is a major cause of product defects. This study presents an anomaly detection system that combines DREAM and PatchCore models in an ensemble, trained and evaluated on synthetic data. Our primary objective was to achieve precision above 99% while minimizing false positives. To this end, we incorporated boundary cases such as light_distortion and thick_panel into the training set, applied a precision-priority threshold (MIN_PRECISION 0.997) selected on a held-out validation set, and used an ensemble rule where both models must predict crack for a positive classification. On a 10k-scale test set, the ensemble achieved 99.86% precision with 10 false positives. We also extended the dataset to approximately 30k samples and introduced an edge_scorch scenario to model laser-cutting-induced edge scorching, which weakens panel bonding and causes edge separation during bending—a phenomenon reported in production environments.

**Keywords:** FPCB, crack detection, anomaly detection, DREAM, PatchCore, precision, synthetic data, edge scorch

---

## 1. Introduction

### 1.1 Background

In the FPCB bending process, copper wire cracking leads directly to product failure. Cracks may occur *during* bending (e.g., due to over-curing or excessive bending trajectory) or may be present in panels that were already damaged before entering the line. In both cases, the final panel should be classified as NG (non-good), and detection accuracy directly affects production yield.

### 1.2 Problem Statement

A central challenge is to achieve high precision (≥99%) while keeping false positives low. In particular, normal samples under illumination-induced edge distortion (light_distortion) have been observed to be misclassified as crack, which motivates the inclusion of such cases in training and evaluation.

### 1.3 Research Objectives

We set two main goals:

- **Goal 1 (primary):** Detect bending-in-process cracks using temporal and local signals (velocity change, shockwave, vibration).
- **Goal 2 (secondary):** Detect already-cracked panels from global pattern differences.

This paper focuses on Goal 1 and reports precision-oriented results.

### 1.4 Scope

All experiments use synthetic data, as real crack-labeled FPCB data were not available at the time of writing. The target metric is precision ≥99% with minimal false positives.

---

## 2. Methods

### 2.1 Dataset and Splits

Data were split by 70% train, 15% validation, and 15% test, with the same ratios applied within each scenario. The random seed was fixed (20260219) for reproducibility. The validation set was used only for threshold selection; the test set was used solely for final evaluation and was never used for training or tuning.

### 2.2 Scenario Configuration

The dataset includes multiple scenarios to reflect real-world variability:

| Scenario        | Approx. count | Label | Role                          |
|-----------------|---------------|-------|-------------------------------|
| normal          | 21,500+       | 0     | Baseline                      |
| light_distortion| 1,600         | 0     | FP mitigation (illumination)  |
| crack, uv_overcured | 2,600    | 1     | Goal 1 (bending-in-process)   |
| micro_crack     | 900           | 1     | Goal 1 (subtle crack)         |
| edge_scorch     | 600           | 1     | Goal 1 (laser edge scorch)    |
| pre_damaged     | 1,500         | 1     | Goal 2                        |
| thick_panel     | 1,200         | 0     | Boundary case                 |

The **edge_scorch** scenario was added to model a reported production issue: laser cutting can scorch FPCB edges, weakening bonding and causing the outermost part of the panel to separate (gape) during bending. This scenario is modeled by concentrating curvature at both panel edges.

After supplements, the total dataset size is approximately 29,900 samples (train ≈20,370, val ≈4,365, test ≈5,165).

### 2.3 Feature Extraction

Features were extracted at per-frame and global levels (about 61 rows per dataset: 60 frames plus one global summary). Advanced features include skewness, kurtosis, autocorrelation, and FFT-based spectral entropy. To avoid label leakage, physics-derived `crack_risk_*` features were excluded from the ML input.

### 2.4 Models and Ensemble

- **DREAM:** Reconstruction-based autoencoder, fitted on normal-only data.
- **PatchCore:** Memory-bank-based anomaly detector.
- **Ensemble:** A sample is classified as crack only when *both* DREAM and PatchCore predict crack.

### 2.5 Threshold Selection

Thresholds were chosen on the validation set to satisfy MIN_PRECISION ≥ 0.997 (precision-priority). If no threshold met this constraint, the one with the highest precision was selected. The test set was never used for threshold selection.

---

## 3. Results

### 3.1 Main Results (10k-Scale Evaluation)

The reported results were obtained with a 10k-scale dataset (train capped at 2,000 normal and 500 crack for computational efficiency; full test set used). Test set size: 78,690 rows (68,625 normal, 10,065 crack).

| Model     | Precision | Recall | FP  | FN   | TP   | TN    |
|-----------|-----------|--------|-----|------|------|-------|
| DREAM     | 99.67%    | 72.6%  | 24  | 2,754| 7,311| 68,601|
| PatchCore | 99.66%    | 69.7%  | 24  | 3,049| 7,016| 68,601|
| **Ensemble** | **99.86%** | 69.7% | **10** | 3,049 | 7,016 | 68,615 |

The ensemble reached 99.86% precision with 10 false positives, meeting the target.

### 3.2 Hard Subset Performance

Performance on difficult subsets:

| Model     | light_distortion (normal) | micro_crack (crack) |
|-----------|---------------------------|----------------------|
| DREAM     | 62/75 (82.7%)             | 45/45 (100%)         |
| PatchCore | 60/75 (80.0%)             | 45/45 (100%)         |
| Ensemble  | 69/75 (92.0%)             | 45/45 (100%)         |

### 3.3 ROC AUC

DREAM: 0.965; PatchCore: 0.961. ROC AUC is not defined for the ensemble (binary agreement rule).

---

## 4. Discussion

### 4.1 Design Choices

Several choices contributed to high precision:

1. **Including light_distortion in training** to reduce false positives from illumination artifacts.
2. **Including thick_panel** to improve robustness at the normal–anomaly boundary.
3. **Using MIN_PRECISION 0.997** to favor precision over recall during threshold selection.
4. **Ensemble rule (both must agree)** to reduce false positives.
5. **Advanced features** (e.g., FFT, spectral entropy) to capture shockwave and vibration patterns.

### 4.2 Limitations

- The model is a 2D surrogate; real 3D stress and strain may differ.
- All experiments use synthetic data; validation on real FPCB imagery is recommended.
- A domain gap between synthetic and real data is expected.

### 4.3 Conclusions

We achieved 99.86% precision with 10 false positives on a 10k-scale synthetic test set using a DREAM–PatchCore ensemble. The dataset was extended to ~30k samples and an edge_scorch scenario was added to better reflect production conditions. Next steps include validation on real FPCB data and further work on recall if needed.

---

## References

[1] Project documentation: PROJECT_GOALS.md, DEVELOPMENT_ROADMAP_FINAL.md (internal).

[2] Phase B insights: PHASE_B_INSIGHTS.md (internal).

[3] Report–data reconciliation: REPORT_DATA_RECONCILIATION.md (internal).

[4] Methodology validation: research_validation_report.md (internal).

---

## Appendix: Experiment Log

| Date   | Action                                      | Result                          |
|--------|---------------------------------------------|---------------------------------|
| 2026-02-24 | 10k dataset generation                   | train=6,650, val=1,425, test=1,425 |
| 2026-02-24 | Analysis (--max-train 2000)               | Ensemble Precision 99.86%, FP 10  |
| 2026-02-25 | 30k supplement + edge_scorch + diversity | Total ~29,900 samples            |
