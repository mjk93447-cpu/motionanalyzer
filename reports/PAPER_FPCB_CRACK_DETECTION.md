# AI-Based Crack Detection in FPCB Bending Process: Achieving 99%+ Precision on 10k Synthetic Dataset

**논문 초안**  
**작성일**: 2026년 2월 24일  
**형식**: 논문식 IMRAD (Introduction, Methods, Results, Discussion)

---

## Abstract

FPCB(Flexible Printed Circuit Board) 굽힘 공정에서 구리 배선 크랙은 제품 불량의 주요 원인이다. 본 연구는 합성 데이터 기반으로 DREAM·PatchCore 앙상블을 활용한 이상 탐지 시스템을 구축하고, 10k 규모 데이터셋에서 Precision 99% 이상 달성을 목표로 하였다. 오탐(False Positive) 최소화를 위해 light_distortion·thick_panel 등 경계 케이스를 학습에 포함하고, MIN_PRECISION 0.997 기준 임계값 선택, DREAM ∧ PatchCore 앙상블 전략을 적용하였다. **결과: Ensemble Precision 99.86%, FP 10을 달성하였다.**

**Keywords**: FPCB, crack detection, anomaly detection, DREAM, PatchCore, precision, synthetic data

---

## 1. Introduction

### 1.1 Background

FPCB 굽힘 공정에서 구리 배선 크랙은 제품 불량의 주요 원인이다. 크랙은 **벤딩 과정 도중** 발생하거나, **이미 손상된 패널**이 투입되는 경우에 발생한다. 두 경우 모두 최종적으로 NG 패널로 판정되어야 하며, 이에 대한 탐지 정확도가 생산성과 직결된다.

### 1.2 Problem Statement

**핵심 과제**: 오탐(False Positive)을 최소화하면서 Precision 99% 이상을 달성하는 것. 특히 light_distortion(조명 왜곡) 등 정상 변동을 크랙으로 오탐하는 문제를 해결해야 한다.

### 1.3 Research Objectives

| Priority | Objective | Detection Target | Approach |
|----------|------------|------------------|----------|
| **Goal 1** | Bending-in-process crack detection | Temporal·local (velocity change, shockwave, vibration) | CPD, DREAM, PatchCore, Ensemble |
| **Goal 2** | Already-cracked panel detection | Global pattern (subtle property/structure difference) | DREAM, PatchCore |

### 1.4 Scope

- **Data**: Synthetic data until real crack data is available
- **Target**: Precision 99%+, FP minimization

---

## 2. Methods

### 2.1 Dataset Split

| Split | Ratio | Purpose |
|-------|-------|---------|
| **Train** | 70% | Model training (DREAM, PatchCore) |
| **Val** | 15% | Threshold selection (precision_priority) |
| **Test** | 15% | **Final evaluation** — not used for training/tuning |

- Class-wise 70/15/15, seed=20260219

### 2.2 10k Dataset Configuration

| Scenario | Count | Label | Goal |
|----------|-------|-------|------|
| normal | 7,000 | 0 | — |
| light_distortion | 500 | 0 | FP mitigation |
| crack | 500 | 1 | Goal 1 |
| uv_overcured | 300 | 1 | Goal 1 |
| micro_crack | 300 | 1 | Goal 1 |
| pre_damaged | 500 | 1 | Goal 2 |
| thick_panel | 400 | 0 | Boundary case |
| **Total** | **9,500** | — | — |

**Split counts**: train=6,650, val=1,425, test=1,425

### 2.3 Feature Extraction

- **Per-frame + global stats**: 61 rows per dataset (60 frames + 1 global)
- **Advanced features**: skewness, kurtosis, autocorrelation, FFT, spectral_entropy
- **Label leakage prevention**: `crack_risk_*` excluded from ML input

### 2.4 Models

| Model | Type | Strategy |
|-------|------|----------|
| **DREAM** | Reconstruction-based | Autoencoder, normal-only fit |
| **PatchCore** | Memory bank | Feature extraction + memory bank |
| **Ensemble** | DREAM ∧ PatchCore | Both predict Crack → Crack |

### 2.5 Threshold Selection

- **Criterion**: MIN_PRECISION 0.997 (precision-priority)
- **Source**: Val set (fallback to test if val too small)
- **Ensemble**: both_agree (both models must predict Crack)

---

## 3. Results

### 3.1 10k Dataset Experiment (Actual)

| Dataset Scale | Train (max 2000) | Val | Test (rows) |
|---------------|------------------|-----|-------------|
| 10k | 2,000 | 1,425 | 78,690 |

- Test: 68,625 normal + 10,065 crack rows (~1,290 datasets)

### 3.2 Achieved Results (10k Scale)

| Model | Precision | Recall | FP | FN | TP | TN |
|-------|-----------|--------|-----|-----|-----|------|
| DREAM | **99.67%** | 72.6% | 24 | 2,754 | 7,311 | 68,601 |
| PatchCore | **99.66%** | 69.7% | 24 | 3,049 | 7,016 | 68,601 |
| **Ensemble** | **99.86%** | 69.7% | **10** | 3,049 | 7,016 | 68,615 |

**Precision 99%+ 달성**: Ensemble Precision 99.86%, FP 10

### 3.3 Hard Subset (light_distortion, micro_crack)

| Model | light_distortion (정상 분류) | micro_crack (크랙 분류) |
|-------|------------------------------|--------------------------|
| DREAM | 62/75 (82.7%) | 45/45 (100%) |
| PatchCore | 60/75 (80.0%) | 45/45 (100%) |
| **Ensemble** | **69/75 (92.0%)** | **45/45 (100%)** |

### 3.4 ROC AUC

| Model | ROC AUC |
|-------|---------|
| DREAM | 0.965 |
| PatchCore | 0.961 |

---

## 4. Discussion

### 4.1 Strategy for Precision 99%+

1. **light_distortion 500**: 5% train share for FP mitigation
2. **thick_panel train 포함**: Boundary case learning
3. **MIN_PRECISION 0.997**: High threshold for FP minimization
4. **Ensemble (DREAM ∧ PatchCore)**: Reduce FP via agreement
5. **Advanced features**: FFT, spectral_entropy for shockwave/vibration

### 4.2 Limitations

- **2D surrogate**: Difference from real 3D stress/strain
- **Synthetic data**: Real FPCB validation recommended
- **Domain gap**: Actual data may differ

### 4.3 Conclusion

- **Goal**: Precision 99%+ on 10k dataset — **달성** (Ensemble 99.86%)
- **Status**: 10k dataset 생성·분석 완료
- **Next**: 실제 FPCB 데이터 검증; Recall 개선 검토

---

## References

- PROJECT_GOALS.md, DEVELOPMENT_ROADMAP_FINAL.md
- PHASE_B_INSIGHTS.md, REPORT_DATA_RECONCILIATION.md

---

## Appendix: Work Log

| Date | Action | Result |
|------|--------|--------|
| 2026-02-24 | 10k dataset generation | train=6650, val=1425, test=1425 |
| 2026-02-24 | analyze_crack_detection --max-train 2000 | Ensemble Precision 99.86%, FP 10 |
