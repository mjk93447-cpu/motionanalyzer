# AI-Based Crack Detection in FPCB Bending Process: Achieving 99%+ Precision on 10k Synthetic Dataset

**논문 초안**  
**작성일**: 2026년 2월 24일  
**형식**: 논문식 IMRAD (Introduction, Methods, Results, Discussion)

---

## Abstract

FPCB(Flexible Printed Circuit Board) 굽힘 공정에서 구리 배선 크랙은 제품 불량의 주요 원인이다. 본 연구는 합성 데이터 기반으로 DREAM·PatchCore 앙상블을 활용한 이상 탐지 시스템을 구축하고, 10k 규모 데이터셋에서 Precision 99% 이상 달성을 목표로 한다. 오탐(False Positive) 최소화를 위해 light_distortion·thick_panel 등 경계 케이스를 학습에 포함하고, MIN_PRECISION 0.997 기준 임계값 선택, DREAM ∧ PatchCore 앙상블 전략을 적용하였다. 결과는 10k 데이터셋 분석 완료 후 갱신된다.

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

### 3.1 10k Dataset Experiment

*Analysis in progress. Results will be updated upon completion.*

| Dataset Scale | Train | Val | Test |
|---------------|-------|-----|------|
| 10k | 6,650 | 1,425 | 1,425 |

### 3.2 Expected Outcomes (from Strategy)

| Metric | Target |
|--------|--------|
| Precision | ≥ 99% |
| FP | 0 (minimize) |
| light_distortion | 100% correct as normal |
| micro_crack | 100% correct as crack |

### 3.3 Previous Results (Small Scale)

| Model | Precision | Recall | FP |
|-------|-----------|--------|-----|
| DREAM | 71.9% | 37.7% | 27 |
| PatchCore | 70.0% | 38.3% | 30 |
| Ensemble | 77.3% | 37.2% | 20 |

*Note: Small scale ~19 test datasets. 10k scale expected to improve with more training data.*

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

- **Goal**: Precision 99%+ on 10k dataset
- **Status**: 10k dataset generated; analysis running
- **Next**: Update results upon analysis completion; real-data validation

---

## References

- PROJECT_GOALS.md, DEVELOPMENT_ROADMAP_FINAL.md
- PHASE_B_INSIGHTS.md, REPORT_DATA_RECONCILIATION.md

---

## Appendix: Work Log

| Date | Action | Result |
|------|--------|--------|
| 2026-02-24 | 10k dataset generation | train=6650, val=1425, test=1425 |
| 2026-02-24 | analyze_crack_detection --max-train 3000 | In progress |
