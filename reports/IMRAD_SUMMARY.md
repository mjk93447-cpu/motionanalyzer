# FPCB 굽힘 공정 크랙 탐지 시스템 — IMRAD 형식 연구 요약

**문서 버전**: 1.0  
**작성일**: 2026년 2월 24일  
**형식**: 논문식 IMRAD (Introduction, Methods, Results, Discussion)

---

## 1. Introduction (서론)

### 1.1 배경 및 문제 정의

FPCB(Flexible Printed Circuit Board) 굽힘 공정에서 구리 배선 크랙은 제품 불량의 주요 원인이다. 크랙은 **벤딩 과정 도중** 발생하거나, **이미 손상된 패널**이 투입되는 경우에 발생한다. 두 경우 모두 최종적으로 NG 패널로 판정되어야 하며, 이에 대한 탐지 정확도가 생산성과 직결된다.

**핵심 과제**: 오탐(False Positive)을 최소화하면서 Precision 99% 이상을 달성하는 것. 특히 light_distortion(조명 왜곡) 등 정상 변동을 크랙으로 오탐하는 문제를 해결해야 한다.

### 1.2 연구 목표

| 우선순위 | 목표 | 감지 대상 | 접근 |
|----------|------|-----------|------|
| **목표 1 (최우선)** | 벤딩 중 크랙 감지 | 시계열·국소적 (속도 변화, 충격파, 진동, 길이 변화) | CPD, DREAM, PatchCore, Temporal |
| **목표 2** | 이미 크랙된 패널 감지 | 전체적 패턴 (미묘한 물성·구조 차이) | DREAM, PatchCore, Ensemble |

### 1.3 범위 (Scope)

- **데이터**: 실제 크랙 데이터 확보 전까지 합성 데이터로 구현·검증
- **출력**: Precision-Recall Score 최대화, 특히 Precision 99%+ 달성

---

## 2. Methods (방법)

### 2.0 데이터 분할 (Train / Val / Test)

| 구분 | 비율 | 용도 |
|------|------|------|
| **Train** | 70% | 모델 학습 (DREAM, PatchCore, Temporal) |
| **Val** | 15% | 임계값 선택, 하이퍼파라미터 튜닝 |
| **Test** | 15% | **최종 성능 평가** — 학습·튜닝에 미사용 |

- 클래스별 70/15/15 비율 유지, 시드 고정(20260219)
- 상세 규모: `reports/DATASET_AND_EXPERIMENT_SPEC.md` 참조

### 2.1 합성 데이터 설계

| 시나리오 | 설명 | 라벨 | 물리 특성 |
|----------|------|------|-----------|
| normal | 정상 굽힘 | 0 | 정상 범위 내 변동 |
| light_distortion | 정상 + 조명 왜곡 | 0 | FP 주요 원인 |
| crack | 굽힘 중 크랙 | 1 | 충격파, 진동, 곡률 집중 |
| uv_overcured | UV 과경화 | 1 | 후반 스냅, 충격파 |
| micro_crack | 초미세 크랙 | 1 | 미세 신호 |
| pre_damaged | 사전 손상 | 1 | 전체 궤적·물성 미묘 차이 |
| thick_panel | 두꺼운 패널 | 0 | 경계 케이스 |

**물리 현상 모델링**:
- 충격파(shockwave): 크랙 발생 시 가속도 스파이크 (3.24x 증가), 지수 감쇠 모델
- 미세 진동(micro-vibration): 25 Hz (crack), 15 Hz (pre_damage), 감쇠 진동 모델

### 2.2 특징 추출

- **Baseline (21 features)**: velocity, acceleration, curvature, strain_surrogate, curvature_concentration 등
- **Advanced (75 features)**: 고차 통계(왜도, 첨도, 자기상관), Temporal Features(변화율, 변화 가속도), 주파수 도메인(FFT, spectral_entropy)
- **라벨 누설 방지**: ML 검증 시 Physics 산출물(`crack_risk_*`) 제외

### 2.3 모델 아키텍처

| 모델 | 유형 | 접근 |
|------|------|------|
| **DREAM** | 재구성 기반 | Autoencoder, normal-only 학습 |
| **PatchCore** | 메모리 뱅크 기반 | Feature extraction + 메모리 뱅크 |
| **Ensemble** | DREAM ∧ PatchCore | 둘 다 Crack 시에만 Crack |
| **Temporal** | LSTM/GRU | 시계열 오토인코더, 슬라이딩 윈도우 시퀀스 |

### 2.4 Change Point Detection (CPD)

- **방법**: CUSUM, Window-based, PELT
- **고도화**: 파라미터 자동 튜닝(Grid Search/Bayesian), 다중 특징 결합, 앙상블 CPD
- **시계열 특징**: acceleration_max, curvature_concentration, strain_surrogate_max

### 2.5 평가 메트릭

- **ROC AUC**: Anomaly rate 변화에 비교적 안정적
- **PR AUC (AUCPR)**: Precision-Recall 균형, 운영 의사결정에 가까움
- **Precision, Recall, F1**: Threshold 기반 이진 분류

### 2.6 개발 단계 (Phase A–B)

- **Phase A**: EXE 배포, Analyze 탭 확장, CPD GUI 통합
- **Phase B**: 합성 데이터 물리 현상 추가, 앙상블, Temporal, 고급 특징, CPD 고도화

---

## 3. Results (결과)

### 3.0 실험별 데이터 규모

| 실험 | Train (데이터셋) | Val | Test | 비고 |
|------|------------------|-----|------|------|
| Goal 1 ML | 812 (normal 749 + crack 63) | 169 | 173 | light_distortion·thick_panel train 포함 |
| Goal 1 CPD | — | — | 80 | crack_frame 메타데이터 |
| Goal 2 ML | 749 (normal 735 + predam 14) | — | 161 | pre_damaged vs normal |
| Phase B 벤치마크 | 자체 생성 소규모 | — | — | 상대 비교용 |

- 특징 행: 데이터셋당 약 61행 (60프레임 + 1 global) → Train 행 ≈ 49,532, Test 행 ≈ 10,553

### 3.1 Precision 99%+ 달성 (최종)

| 지표 | Baseline | 최종(Ensemble) | 개선 |
|------|----------|----------------|------|
| **Precision** | 86~88% | **100%** | +12~14%p |
| **False Positive** | 93~130 | **0** | 100% 감소 |
| **light_distortion 정상 분류** | 0% | **100%** | 100%p |

### 3.2 Confusion Matrix (Ensemble, 최종)

|  | 예측 정상 | 예측 크랙 |
|--|-----------|-----------|
| **실제 정상** | 9,638 (TN) | 0 (FP) |
| **실제 크랙** | 297 (FN) | 557 (TP) |

- **Precision**: 557/(557+0) = **100%**
- **Recall**: 557/854 = 65.2%
- **FP**: 0

### 3.3 Phase B 벤치마크 (Baseline Features)

| 모델 | ROC AUC | PR AUC | Precision | Recall | F1 |
|------|---------|--------|-----------|--------|-----|
| DREAM | 0.913 | 0.953 | 1.000 | 0.672 | 0.804 |
| PatchCore | 0.908 | 0.954 | 0.982 | 0.775 | 0.866 |
| Ensemble | 0.908 | 0.954 | 0.982 | 0.775 | 0.866 |
| Temporal | 0.100 | 0.286 | 0.286 | 1.000 | 0.444 |

### 3.4 고급 특징 효과 (DREAM+Advanced)

| 모델 | ROC AUC | PR AUC | 비고 |
|------|---------|--------|------|
| DREAM | 0.928 | 0.959 | baseline |
| DREAM+Advanced | 1.000 | 1.000 | 과적합 가능성 주의 |

### 3.5 Change Point Detection

- **Goal 1**: CPD 정확도 (frame 30–45 범위)
- **mean_error_frames**: 1.09
- **within_5_frames_pct**: 100.0%

### 3.6 적용된 핵심 조치

1. light_distortion 50개: train 비중 5%로 확대
2. Precision 단일 목표: Recall 제약 제거, 임계값 상향
3. thick_panel train 포함: 경계 케이스 학습
4. light_distortion 증강 다양화: offset/jitter/spike 파라미터 확대
5. 앙상블: DREAM ∧ PatchCore (둘 다 Crack 시에만 Crack)
6. MIN_PRECISION 0.997: 고임계값으로 FP 최소화

---

## 4. Discussion (고찰)

### 4.1 해석

- **Precision 100% 달성**: FP 0으로 오탐 최소화 목표 달성. light_distortion 등 정상 변동에 대한 강건성 확보.
- **Recall 65.2%**: FN 297 존재. 필요 시 Phase 4에서 Recall 개선 검토.
- **DREAM vs PatchCore**: 유사한 성능(ROC AUC ~0.91). Ensemble은 서로 다른 접근 방식 결합으로 FP 감소에 기여.
- **고급 특징**: DREAM+Advanced에서 ROC AUC 1.000, PR AUC 1.000. 합성 데이터에서 과적합 가능성 높음 — 실제 데이터 검증 필수.

### 4.2 Temporal 모델 이슈

- ROC AUC 0.100 수준. Recall 1.000, Precision 0.286 → 모든 것을 anomaly로 예측하는 경향.
- **가능 원인**: 시계열 구조 보존 부족, threshold 설정 문제, 데이터 분할 방식.
- **개선 방향**: 데이터셋 레벨 분할 후 시퀀스 생성, threshold 최적화, 하이퍼파라미터 그리드 서치.

### 4.3 앙상블 다양성

- Ensemble이 단일 모델 대비 큰 향상 없음 (가중치 최적화 필요).
- DREAM과 PatchCore 예측이 유사하여 다양성 부족.
- **개선 방향**: Temporal + DREAM 등 서로 다른 접근 방식 결합, 스태킹 전략 재검토.

### 4.4 제한 사항

- **2D surrogate**: 실제 3D stress/strain과 차이 존재.
- **합성 데이터**: 실제 FPCB 영상 확보 후 재검증 권장.
- **Domain gap**: 실제 데이터와의 domain gap 존재.

### 4.5 결론

- **목표 1 (벤딩 중 크랙)**: Precision 99%+ 달성, FP 0. CPD 정확도 100% (within 5 frames).
- **목표 2 (이미 크랙된 패널)**: DREAM/PatchCore ROC AUC ~0.84, PR AUC 0.57~0.64. 추가 고도화 여지.
- **배포 준비**: Phase A·B 완료, EXE 배포 가능. 실제 데이터 확보 전 단계 개발 완료.

### 4.6 참고 문서

- `docs/PROJECT_GOALS.md`: 프로젝트 목표
- `docs/DEVELOPMENT_ROADMAP_FINAL.md`: 개발 로드맵
- `docs/PHASE_B_INSIGHTS.md`: Phase B 인사이트
- `reports/CRACK_DETECTION_FINAL_REPORT.md`: 최종 개발 보고서
- `reports/DATASET_AND_EXPERIMENT_SPEC.md`: **데이터 분할·규모·실험별 명세**
