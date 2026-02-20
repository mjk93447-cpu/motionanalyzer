# 탐지 성능 개선 전략 (Detection Improvement Strategy)

> **통합 문서**: 최종 성과 및 통합 정리는 [CRACK_DETECTION_FINAL_REPORT.md](./CRACK_DETECTION_FINAL_REPORT.md) 참조.

**작성일**: 2026-02-19 | **상태**: PRECISION_99_DEVELOPMENT_STRATEGY로 통합

---

## 원본 요약 (참고용)

## 1. 추가된 시나리오 요약

| 시나리오 | 설명 | 비율/개수 | 라벨 |
|----------|------|-----------|------|
| **light_distortion** | 정상 + 조명에 의한 테두리선 왜곡 시뮬레이션 | 정상의 1~2% (15개) | 0 (정상) |
| **micro_crack** | 초미세 크랙 패턴 (crack_gain 5.0, 좁은 폭) | 10개 | 1 (크랙) |

---

## 2. Confusion Matrix 기반 평가 결과

### 전체 테스트셋 (frame-level)

- **TN**: 정상으로 정확 분류된 정상 프레임
- **FP**: 정상을 크랙으로 오탐 (light_distortion 포함 시 증가 가능)
- **FN**: 크랙을 정상으로 오탐 (micro_crack 포함 시 증가 가능)
- **TP**: 크랙으로 정확 분류된 크랙 프레임

### Hard Subset (dataset-level)

- **light_distortion**: 정상(조명왜곡) 데이터를 정상으로 정확 분류한 비율
- **micro_crack**: 초미세 크랙 데이터를 크랙으로 정확 분류한 비율

---

## 3. 탐지 개선 전략

### 3.1 light_distortion 대응

| 문제 | 전략 |
|------|------|
| 조명 변화로 인한 edge detection drift | • 주파수 영역 특징(FFT, spectral entropy) 강화<br>• Z-score 정규화로 프레임 간 스케일 변동 완화<br>• 데이터 증강: 조명 시뮬레이션(per-frame offset, spike) 추가 |
| FP 증가 (정상→크랙 오탐) | • 임계값 상향 조정<br>• light_distortion 샘플을 train에 포함하여 정상 분포 확장 |

### 3.2 micro_crack 대응

| 문제 | 전략 |
|------|------|
| 미세 곡률/가속도 신호 | • 곡률 집중도(curvature_concentration) 특징 가중치 상향<br>• 가속도 2차 미분 등 미세 스파이크 민감도 향상<br>• crack_gain, shockwave_amplitude 파라미터 튜닝 |
| FN 증가 (크랙→정상 오탐) | • Recall 우선: 임계값 하향 또는 anomaly score 재보정<br>• micro_crack 샘플을 crack train에 충분히 포함 |

### 3.3 Confusion Matrix 기반 조정

| 지표 | 조치 |
|------|------|
| **FP↑** | 임계값 상향, 정상 분포 학습 강화, light_distortion 증강 |
| **FN↑** | 임계값 하향, crack 특징 추가, micro_crack 증강 |
| **Precision/Recall 트레이드오프** | F1 최적 임계값 또는 운영 비용에 따른 가중 F1 사용 |

---

## 4. 실행 전략 (Precision 99%+ 집중)

| 항목 | 구현 |
|------|------|
| **임계값 선택** | F1-max 대신 Precision ≥ 99%, Recall ≥ 80% 동시 만족하는 구간에서 Recall 최대화; 불가 시 max(Precision) s.t. Recall ≥ 80% |
| **light_distortion 학습** | normal_train에 light_distortion을 우선 포함하여 정상 분포 확장 (FP 감소) |
| **과적합 방지** | DREAM: weight_decay=1e-5, 기존 Dropout(0.1) 유지 |
| **Recall 80~90%** | Precision 99% 달성 시 Recall은 80% 이상 유지; 90% 초과는 불필요 |

**참고**: 합성 데이터 특성상 Precision 99%가 항상 달성되지는 않을 수 있음. light_distortion 등 경계 케이스가 FP를 유발. 현재 전략은 가능한 최대 Precision을 Recall 80% 제약 하에서 선택.

## 5. 구현 변경 사항

- `synthetic.py`: `light_distortion` (per-frame offset, spike), `micro_crack` (crack_gain 5.0, shockwave 1.2) 시나리오 추가
- `generate_ml_dataset.py`: normal_light_distortion 15개, micro_crack 10개 생성
- `analyze_crack_detection.py`: `_select_threshold_precision_priority`, light_distortion in train, hard subset 보고
- `dream.py`: weight_decay 파라미터 추가 (Adam optimizer)

---

## 6. 제한 사항 (FPCB Domain)

- 2D surrogate 모델: 실제 3D stress/strain과의 차이 존재
- light_distortion, micro_crack 파라미터는 provisional; secure-site calibration 필요
- 합성 데이터 domain gap: 실제 FPCB 영상 확보 후 재검증 권장
