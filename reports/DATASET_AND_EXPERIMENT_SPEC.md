# 데이터셋 분할 및 실험 결과 명세

**작성일**: 2026년 2월 24일  
**목적**: 학습(Training) 데이터와 검증(Validation/Test) 데이터를 명확히 구분하고, 실험 결과를 데이터 규모·구분에 따라 정리

---

## 1. 데이터 분할 원칙

### 1.1 분할 비율

| 구분 | 비율 | 용도 |
|------|------|------|
| **Train** | 70% | 모델 학습 (DREAM, PatchCore, Temporal 등) |
| **Val** | 15% | 임계값(threshold) 선택, 하이퍼파라미터 튜닝 |
| **Test** | 15% | **최종 성능 평가** — 학습·튜닝에 사용하지 않음 |

- **클래스별 비율 유지**: 각 시나리오(normal, crack, light_distortion 등)별로 70/15/15 적용
- **시드 고정**: `seed=20260219` (재현성)

### 1.2 데이터 누설 방지

- **정규화**: `fit_df`를 train의 **normal-only** 데이터로 고정, val/test에는 transform만 적용
- **임계값 선택**: val 세트로 threshold 선택 (val이 너무 작으면 test 사용, 단 문서화)
- **ML 특징**: `crack_risk_*` 등 Physics 산출물은 ML 검증 시 **제외** (라벨 누설 방지)

---

## 2. 데이터 규모 (Default Scale)

### 2.1 시나리오별 원시 데이터 개수

| 시나리오 | goal | label | 총 개수 | train | val | test |
|----------|------|-------|---------|-------|-----|------|
| normal | normal | 0 | 1,000 | 700 | 150 | 150 |
| light_distortion | normal | 0 | 50 | 35 | 7 | 8 |
| crack | goal1 | 1 | 50 | 35 | 7 | 8 |
| uv_overcured | goal1 | 1 | 30 | 21 | 4 | 5 |
| micro_crack | goal1 | 1 | 10 | 7 | 1 | 2 |
| pre_damaged | goal2 | 1 | 20 | 14 | 3 | 3 |
| thick_panel | variant | 0 | 20 | 14 | 3 | 3 |
| **합계** | — | — | **1,180** | **826** | **185** | **179** |

### 2.2 목표별 사용 데이터

#### Goal 1 (벤딩 중 크랙 감지)

| 구분 | Normal (label=0) | Crack (label=1, goal1) | 합계 |
|------|------------------|-------------------------|------|
| **Train** | normal(700) + light_dist(35) + thick(14) = **749** | crack(35)+uv(21)+micro(7) = **63** | **812** |
| **Val** | normal(150) + light_dist(7) = **157** | crack(7)+uv(4)+micro(1) = **12** | **169** |
| **Test** | normal(150) + light_dist(8) = **158** | crack(8)+uv(5)+micro(2) = **15** | **173** |

- **특징 행 수**: `include_per_frame=True`, `include_global_stats=True` → 데이터셋당 약 61행 (60프레임 + 1 global)
- **Train 행 수**: 812 × 61 ≈ **49,532**
- **Test 행 수**: 173 × 61 ≈ **10,553**

#### Goal 2 (이미 크랙된 패널 감지)

| 구분 | Normal (label=0) | Pre-damaged (label=1, goal2) | 합계 |
|------|------------------|------------------------------|------|
| **Train** | normal(700) + light_dist(35) = **735** | pre_damaged(14) = **14** | **749** |
| **Test** | normal(150) + light_dist(8) = **158** | pre_damaged(3) = **3** | **161** |

---

## 3. 실험별 데이터 사용 및 결과

### 3.1 Goal 1 ML (DREAM / PatchCore / Ensemble)

| 항목 | 내용 |
|------|------|
| **학습 데이터** | Train 812 데이터셋 (normal 749 + crack 63) |
| **임계값 선택** | Val 169 데이터셋 (precision_priority 기준) |
| **평가 데이터** | **Test 173 데이터셋** (학습·튜닝에 미사용) |
| **Hard subset** | Test 내 light_distortion(8), micro_crack(2) |

**실험 결과 (최종, Precision 99%+ 달성)**

| 모델 | Precision | Recall | FP | FN | TP | TN |
|------|-----------|--------|-----|-----|-----|-----|
| DREAM | 99.83% | 67.8% | 1 | — | — | — |
| PatchCore | 99.82% | 65.5% | 1 | — | — | — |
| **Ensemble** | **100%** | **65.2%** | **0** | 297 | 557 | 9,638 |

- **참고**: 위 TN/FP/FN/TP는 10k scale 등 대규모 실행 시 행(row) 단위 집계. Default scale에서는 데이터셋 단위로 집계 시 Test 173개.

### 3.2 Goal 1 CPD (Change Point Detection)

| 항목 | 내용 |
|------|------|
| **평가 데이터** | goal1 crack 데이터 (crack_frame 메타데이터 있음) |
| **n_evaluated** | 80 (목표 1 crack 시나리오) |
| **평가 방식** | 감지된 change point vs crack_frame 차이 |

**실험 결과**

| 지표 | 값 |
|------|-----|
| mean_error_frames | 1.09 |
| within_5_frames_pct | 100.0% |

### 3.3 Goal 2 ML (이미 크랙된 패널)

| 항목 | 내용 |
|------|------|
| **학습 데이터** | normal(735) + pre_damaged(14) |
| **평가 데이터** | normal(158) + pre_damaged(3) |

**실험 결과**

| 모델 | ROC AUC | PR AUC |
|------|---------|--------|
| DREAM | 0.843 | 0.570 |
| PatchCore | 0.842 | 0.642 |

### 3.4 Phase B 벤치마크 (benchmark_phase_b_comprehensive.py)

| 항목 | 내용 |
|------|------|
| **데이터** | 스크립트 내부 합성 데이터 생성 (n_normal=5, n_crack=5 등 소규모) |
| **분할** | 70/30 (데이터셋 레벨) |
| **용도** | Phase B 모델(DREAM, PatchCore, Ensemble, Temporal, Advanced) 상대 비교 |

**실험 결과 (Baseline Features)**

| 모델 | ROC AUC | PR AUC | Precision | Recall |
|------|---------|--------|-----------|--------|
| DREAM | 0.913 | 0.953 | 1.000 | 0.672 |
| PatchCore | 0.908 | 0.954 | 0.982 | 0.775 |
| Ensemble | 0.908 | 0.954 | 0.982 | 0.775 |
| Temporal | 0.100 | 0.286 | 0.286 | 1.000 |

---

## 4. 분석 스크립트별 데이터 사용 요약

| 스크립트 | Train | Val | Test | 비고 |
|----------|-------|-----|------|------|
| `analyze_crack_detection.py` | 812 (normal+crack) | 169 | 173 | Goal 1, light_distortion·thick_panel train 포함 |
| `evaluate_goal1_ml.py` | normal+crack (goal1) | — | normal+crack (goal1) | goal만 사용, val 없음 |
| `evaluate_goal2_ml.py` | normal+predam | — | normal+predam | Goal 2 전용 |
| `evaluate_goal1_cpd.py` | — | — | goal1 crack | CPD 정확도 |
| `benchmark_phase_b_comprehensive.py` | 자체 생성 | — | 자체 생성 | 소규모, 상대 비교용 |

---

## 5. Scale 사전설정 (generate_ml_dataset.py)

| Scale | normal | light_dist | crack | uv | micro | predam | thick | 총계 |
|-------|--------|------------|-------|-----|-------|--------|------|------|
| default | 1,000 | 50 | 50 | 30 | 10 | 20 | 20 | ~1,180 |
| 10k | 7,000 | 500 | 500 | 300 | 300 | 500 | 400 | ~9,500 |
| 100k | 75,000 | 5,000 | 5,000 | 3,000 | 3,000 | 5,000 | 4,000 | ~100,000 |

---

## 6. 참조

- `scripts/generate_ml_dataset.py`: 데이터 생성 및 split 할당
- `scripts/analyze_crack_detection.py`: Goal 1 ML 평가, train/val/test 구분
- `docs/SYNTHETIC_DATA_SPEC.md`: 합성 데이터 규격
