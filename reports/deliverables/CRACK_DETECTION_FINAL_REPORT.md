# FPCB 굽힘 공정 크랙 탐지 — 최종 개발 보고서

**문서 버전**: 1.0  
**작성일**: 2026-02-20  
**목적**: Precision 99%+ 달성 개발 성과 통합 정리

---

## 1. 개요

### 1.1 프로젝트 목표

FPCB(Flexible Printed Circuit Board) 굽힘 공정에서 발생하는 구리 배선 크랙을 AI 기반으로 탐지하고, **Precision 99% 이상**을 달성하여 오탐(False Positive)을 최소화하는 것.

### 1.2 최종 성과 요약

| 지표 | Baseline | 최종(Ensemble) | 개선 |
|------|----------|----------------|------|
| **Precision** | 86~88% | **100%** | +12~14%p |
| **False Positive** | 93~130 | **0** | 100% 감소 |
| **light_distortion 정상 분류** | 0% | **100%** | 100%p |

---

## 2. 시나리오 및 데이터셋

### 2.1 합성 데이터 시나리오

| 시나리오 | 설명 | 개수 | 라벨 |
|----------|------|------|------|
| normal | 정상 굽힘 | 1,000 | 0 |
| light_distortion | 정상 + 조명 왜곡 | 50 | 0 |
| crack | 굽힘 중 크랙 | 50 | 1 |
| uv_overcured | UV 과경화 | 30 | 1 |
| micro_crack | 초미세 크랙 | 10 | 1 |
| pre_damaged | 사전 손상 | 20 | 1 |
| thick_panel | 두꺼운 패널 | 20 | 0 |

### 2.2 FP(오탐) 원인 분석

- **light_distortion**: 조명 변화로 테두리선 왜곡 → 정상인데 크랙으로 오탐
- **정상 변동성**: 노이즈, 굽힘 초기/말기 스파이크
- **경계 케이스**: thick_panel 등 정상에 가까운 시나리오

---

## 3. 개발 전략 및 실행

### 3.1 4단계 로드맵

| Phase | 기간 | 초점 | 결과 |
|-------|------|------|------|
| 1 | 1~2주 | 정상 분포 강화 | Precision 99%+ |
| 2 | 2~4주 | 특징·임계값 정교화 | - |
| 3 | 4~8주 | 앙상블 | FP 0 달성 |
| 4 | 8주~ | 운영 최적화 | - |

### 3.2 적용된 핵심 조치

1. **light_distortion 50개**: train 비중 5%로 확대
2. **Precision 단일 목표**: Recall 제약 제거, 임계값 상향
3. **thick_panel train 포함**: 경계 케이스 학습
4. **light_distortion 증강 다양화**: offset/jitter/spike 파라미터 확대
5. **앙상블**: DREAM ∧ PatchCore (둘 다 Crack 시에만 Crack)
6. **MIN_PRECISION 0.997**: 고임계값으로 FP 최소화

---

## 4. 최종 성능

### 4.1 Confusion Matrix (Ensemble, 최종)

|  | 예측 정상 | 예측 크랙 |
|--|-----------|------------|
| **실제 정상** | 9,638 (TN) | 0 (FP) |
| **실제 크랙** | 297 (FN) | 557 (TP) |

- **Precision**: 557/(557+0) = **100%**
- **Recall**: 557/854 = 65.2%
- **FP**: 0

### 4.2 모델별 성능

| 모델 | Precision | FP | Recall |
|------|-----------|-----|--------|
| DREAM | 99.83% | 1 | 67.8% |
| PatchCore | 99.82% | 1 | 65.5% |
| **Ensemble** | **100%** | **0** | 65.2% |

### 4.3 Hard Subset (어려운 케이스)

| 시나리오 | Ensemble | 비고 |
|----------|----------|------|
| light_distortion | 8/8 (100%) | 100% 정상 분류 |
| micro_crack | 2/2 (100%) | 100% 크랙 분류 |

---

## 5. 구현 변경 사항

| 파일 | 변경 |
|------|------|
| `generate_ml_dataset.py` | LIGHT_DISTORTION_COUNT 15→50 |
| `analyze_crack_detection.py` | MIN_PRECISION 0.997, thick_panel train, Ensemble, 앙상블 |
| `synthetic.py` | light_distortion offset/jitter/spike 다양화 |
| `dream.py` | weight_decay 1e-5 |

---

## 6. 제한 사항 및 권장 사항

- **2D surrogate**: 실제 3D stress/strain과 차이 존재
- **합성 데이터**: 실제 FPCB 영상 확보 후 재검증 권장
- **Recall**: 65% 수준. 필요 시 Phase 4에서 Recall 개선 검토

---

## 7. 참조 문서

- `PRECISION_99_DEVELOPMENT_STRATEGY.md`: 전체 전략
- `crack_detection_analysis/`: confusion matrix, insights
