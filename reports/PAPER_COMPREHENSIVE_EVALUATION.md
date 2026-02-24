# 논문 종합 평가: 전체과정·데이터 신뢰성·분석흐름·연구방법·완성도

**목적**: PAPER_FPCB_CRACK_DETECTION 논문의 전체 과정, 데이터 신뢰성, 데이터셋 관계, 분석 흐름, 연구 방법 적합성, 결과 논문 완성도 및 이해 용이성을 평가

---

## 1. 전체 과정 평가

### 1.1 연구 흐름 요약

| 단계 | 내용 | 상태 |
|------|------|------|
| 1. 데이터 생성 | generate_ml_dataset (10k) → supplement_ml_dataset → supplement_edge_scorch | ✅ |
| 2. 데이터 분할 | 70/15/15, seed=20260219, 시나리오별 비율 유지 | ✅ |
| 3. 특징 추출 | per-frame, FFT, spectral entropy, crack_risk 제외 | ✅ |
| 4. 모델 학습 | DREAM (normal-only), PatchCore (memory bank) | ✅ |
| 5. 임계값 선택 | Val set, MIN_PRECISION 0.997 | ✅ |
| 6. 평가 | Test set (미사용), Ensemble (both agree) | ✅ |

### 1.2 데이터 → 분석 → 논문 연결

```
manifest.json (29,900) → analyze_crack_detection.py → analysis.json
                                    ↓
                    insights.md, confusion_matrix_*.png
                                    ↓
                    PAPER_FPCB_CRACK_DETECTION.md (실제 결과 반영)
```

---

## 2. 데이터 신뢰성 평가

### 2.1 데이터셋 규모 (실제 manifest 기준)

| 항목 | 값 | 출처 |
|------|-----|------|
| 총 샘플 | 29,900 | research_validation_report |
| Train | 20,370 | manifest |
| Val | 4,365 | manifest |
| Test | 5,165 | manifest |
| Train∩Test 중복 | 0 | validate_research_methodology |

### 2.2 시나리오별 분포 (실제)

| 시나리오 | 개수 | label | 역할 |
|----------|------|-------|------|
| normal | 21,500 | 0 | 기준 |
| crack_in_bending | 2,600 | 1 | Goal 1 |
| light_distortion | 1,600 | 0 | FP 완화 |
| pre_damaged | 1,500 | 1 | Goal 2 |
| thick_panel | 1,200 | 0 | 경계 케이스 |
| micro_crack | 900 | 1 | Goal 1 |
| edge_scorch | 600 | 1 | Goal 1 (레이저 그을림) |

### 2.3 신뢰성 요인

| 요인 | 평가 | 비고 |
|------|------|------|
| Split 분리 | ✅ | train/val/test 완전 분리 |
| 정규화 | ✅ | normal-only fit, label leakage 없음 |
| 시드 고정 | ✅ | 재현 가능 |
| crack_risk 제외 | ✅ | 물리 파생 특징 제외 |
| 합성 데이터 | ⚠️ | 실제 FPCB 데이터 없음, domain gap |

---

## 3. 데이터셋 간 관계 및 분석 흐름 순서

### 3.1 데이터셋 계층

```
[원시 합성] → generate_ml_dataset (10k)
                    ↓
[10k manifest] → supplement_ml_dataset → [~28k manifest]
                    ↓
[~28k] → supplement_edge_scorch → [30k manifest, 29,900]
```

### 3.2 분석 시 사용 데이터 (analyze_crack_detection.py)

| 구분 | 조건 | max-train 2000 시 | max-train 5000 시 |
|------|------|------------------|-------------------|
| Train (normal) | normal + light_dist + thick, split=train | min(2000, N) | min(5000, N) |
| Train (crack) | goal1, split=train | min(500, N) | min(1250, N) |
| Val | normal + goal1, split=val | 전체 | 전체 |
| Test | normal + goal1, split=test | 전체 | 전체 |

### 3.3 분석 흐름 순서 (올바름)

1. manifest 로드 → train/val/test 경로 수집
2. prepare_training_data (normal, crack)
3. normalize_features (fit on normal train only)
4. run_dream_training, run_patchcore_training
5. Val로 threshold 선택 (precision_priority)
6. Test로 최종 평가 (confusion matrix, FP, FN, TP, TN)

---

## 4. 연구 방법의 적합성

### 4.1 목표와 방법 정합

| 목표 | 방법 | 적합성 |
|------|------|--------|
| Precision 99%+ | MIN_PRECISION 0.997, threshold on val | ✅ |
| FP 최소화 | Ensemble (both agree) | ✅ |
| light_distortion 오탐 | light_distortion train 포함 | ✅ |
| edge_scorch | edge_scorch 시나리오 추가 | ✅ |

### 4.2 방법론적 강점

- **One-class 학습**: DREAM, PatchCore 모두 normal 기반 anomaly detection
- **Precision-priority**: 생산 라인 비용 구조 반영
- **Ensemble**: 두 모델 합의로 FP 58% 감소 (24→10)

### 4.3 개선 여지

- **Recall 69.7%**: FN 3,049 — 일부 crack 미검출
- **max-train 제한**: 2000/5000으로 학습량 제한 (연산 효율)

---

## 5. 결과 논문 완성도 및 이해 용이성

### 5.1 강점

| 항목 | 평가 |
|------|------|
| IMRAD 구조 | ✅ Introduction, Methods, Results, Discussion |
| 표·그림 | ✅ Fig 1–5, Table 1–2 |
| 실제 데이터 기반 | ✅ analysis.json 수치 반영 |
| 참고문헌 | ✅ [1]–[3] 외부, [4]–[5] 내부 |
| 재현성 | ✅ Appendix B 스크립트 명시 |

### 5.2 개선 필요

| 항목 | 현재 | 개선 방향 |
|------|------|-----------|
| 10k vs 30k 혼동 | "10k-scale evaluation", "30k dataset" 혼재 | 명확히: 분석 시 train cap (max-train), test는 30k manifest 기반 |
| 시나리오 명칭 | "crack, uv_overcured" | manifest의 crack_in_bending, goal1과 정합 |
| REPORT_DATA_RECONCILIATION | 77.3% 등 구버전 수치 | 30k/analysis.json 기준으로 갱신 |

### 5.3 이해 용이성

- **Abstract**: 핵심 수치(99.86%, FP 10) 명시 ✅
- **Methods**: 파이프라인, 시나리오, 임계값 선택 설명 ✅
- **Results**: 혼동행렬, 표, 유사 논문 비교 ✅
- **Discussion**: 설계 선택, 한계, 결론 ✅

---

## 6. 권장 정제 사항 (Iterative Refinement)

1. **데이터 규모 명확화**: "10k-scale evaluation" → "Evaluation with train capped at 2,000 normal + 500 crack; full test set from 30k manifest (5,165 test samples, 78,690 rows)"
2. **REPORT_DATA_RECONCILIATION 갱신**: analysis.json 99.86% 기준으로 수정
3. **시나리오 표 정합**: manifest의 goal/scenario와 논문 표 일치
4. **데이터 흐름 다이어그램**: 논문에 "데이터셋 관계" 간단 도식 추가
5. **객관성 강화**: "우리 결과가 유리" 대신 "metrics differ; precision-oriented design achieves target" 등 중립 표현
