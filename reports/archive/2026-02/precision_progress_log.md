# Precision 개선 진행 로그

> **통합 문서**: [CRACK_DETECTION_FINAL_REPORT.md](./CRACK_DETECTION_FINAL_REPORT.md) | **PPT**: [FPCB_Crack_Detection_Final_Report.pptx](./FPCB_Crack_Detection_Final_Report.pptx)

---

## Baseline (전략 수립 전)
| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | 86.3% | 130 | 95.7% |
| PatchCore | 88.1% | 93 | 80.2% |
| light_distortion | 0/3 (0%) | - | - |

## Loop 1: light_distortion 50개 + Precision-only threshold + thick_panel train
| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | **99.15%** ✓ | 5 | 68.6% |
| PatchCore | **99.14%** ✓ | 5 | 67.8% |
| light_distortion | 4/8 (50%) | - | - |

**변경**: LIGHT_DISTORTION_COUNT 15→50, MIN_RECALL 0.80→0, normal_train에 thick_panel 포함

## Loop 2: light_distortion 증강 다양화 (offset/spike 파라미터 확대)
| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | **99.13%** ✓ | 5 | 67.2% |
| PatchCore | **99.12%** ✓ | 5 | 66.2% |
| light_distortion | 5/8 (62.5%) DREAM, 4/8 (50%) PatchCore | - | - |

**변경**: synthetic.py light_distortion offset ±1~5, jitter 0.8~2.0, spike 1~5%

## Loop 3: 앙상블 (DREAM ∧ PatchCore)
| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | **99.13%** ✓ | 5 | 67.2% |
| PatchCore | **99.12%** ✓ | 5 | 66.2% |
| **Ensemble** | **99.65%** ✓ | **2** | 66.2% |
| light_distortion | **7/8 (87.5%)** Ensemble | - | - |

**변경**: 두 모델 모두 Crack 예측 시에만 Crack → FP 상호 필터링

## Loop 4: MIN_PRECISION 0.997 (임계값 상향)
| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | **99.83%** ✓ | 1 | 67.8% |
| PatchCore | **99.82%** ✓ | 1 | 65.5% |
| **Ensemble** | **100%** ✓ | **0** | 65.2% |
| light_distortion | **8/8 (100%)** Ensemble | - | - |

**변경**: MIN_PRECISION 0.99→0.997 → FP=0 달성
