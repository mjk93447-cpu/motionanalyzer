# Precision 99%+ 중장기 개발전략

**작성일**: 2026-02-19  
**목표**: Precision 극대화 (99% 이상), Recall은 후순위  
**원칙**: 전략 완성도 확보 후 단계적 개발

---

## 1. 현황 분석

### 1.1 현재 성능 (Full Dataset, frame-level)

| 모델 | Precision | Recall | FP | FN | TP | TN |
|------|-----------|--------|-----|-----|-----|------|
| DREAM | 86.3% | 95.7% | 130 | 37 | 817 | 9,203 |
| PatchCore | 88.1% | 80.2% | 93 | 169 | 685 | 9,240 |

### 1.2 Precision 99% 달성 조건

- **DREAM**: FP ≤ 8 (현재 130 → 122 감소 필요)
- **PatchCore**: FP ≤ 7 (현재 93 → 86 감소 필요)

### 1.3 FP(False Positive) 근본 원인

| 원인 | 비중(추정) | 특성 |
|------|------------|------|
| **light_distortion** | 높음 | 정상인데 조명 왜곡으로 anomaly score 상승 → 3/3 test 전부 FP |
| **정상 변동성** | 중간 | 노이즈, 굽힘 초기/말기 프레임의 자연스러운 스파이크 |
| **경계 케이스** | 중간 | thick_panel, pre_damage 등 정상에 가까운 시나리오 |
| **모델 과민** | 낮음 | 정상 분포 학습 부족, 임계값 미세 조정 여지 |

---

## 2. 전략 원칙

1. **Precision First**: Recall은 99% 달성 후 필요 시 단계적으로 개선
2. **FP 제거 우선**: FP 원인별로 체계적 제거
3. **과적합 금지**: 정상 분포 과다 fitting 방지, 검증/테스트 분리 유지
4. **단계적 검증**: 각 단계별 Precision 측정 후 다음 단계 진행

---

## 3. 단계별 로드맵

### Phase 1: 정상 분포 강화 (단기, 1~2주)

| # | 과제 | 목표 | 검증 |
|---|------|------|------|
| 1.1 | light_distortion train 비중 확대 | 정상의 5~10% 수준 (50~100 샘플) | light_distortion test FP → 0 |
| 1.2 | light_distortion 증강 다양화 | offset/spike 파라미터 범위 확대 | 정상 분포 커버리지 증가 |
| 1.3 | 정상 train에 경계 케이스 포함 | thick_panel, 노이즈 상위 정상 | FP 감소, Precision 상승 |
| 1.4 | 정상 전용 검증셋 분리 | train/val/test 3-way split | 과적합 조기 감지 |

**예상 효과**: Precision 90~92% 수준

---

### Phase 2: 특징 및 임계값 정교화 (중기, 2~4주)

| # | 과제 | 목표 | 검증 |
|---|------|------|------|
| 2.1 | FP 유발 특징 분석 | 어떤 특징이 정상→크랙 오탐에 기여하는지 | feature importance, SHAP |
| 2.2 | 정상-크랙 분리도 높은 특징 강화 | curvature_concentration, acceleration_std 등 | score 분포 격차 확대 |
| 2.3 | 조명/노이즈 불변 특징 추가 | 주파수 영역, 정규화 통계 | light_distortion에 강건 |
| 2.4 | 임계값 다단계 설계 | 1차: 고임계값(Precision 99% 목표), 2차: 보조 규칙 | 운영 시 FP 최소화 |
| 2.5 | Per-dataset 집계 전략 | 비디오 단위 max/mean score → 1 예측 | frame-level FP 억제 |

**예상 효과**: Precision 94~96% 수준

---

### Phase 3: 모델 및 파이프라인 개선 (중장기, 4~8주)

| # | 과제 | 목표 | 검증 |
|---|------|------|------|
| 3.1 | 2-Stage 파이프라인 | 1단: 고Recall 스크리너 → 2단: 고Precision 분류기 | 2단에서 Precision 99% |
| 3.2 | 앙상블 (DREAM + PatchCore) | 두 모델 모두 Crack 예측 시에만 Crack | FP 상호 필터링 |
| 3.3 | Confidence calibration | Platt scaling, isotonic regression | score→확률 보정, 임계값 안정화 |
| 3.4 | Out-of-distribution 감지 | 정상 분포 밖 샘플 별도 플래그 | 의심 케이스 수동 검토 |
| 3.5 | 실제 데이터 연동(가능 시) | secure-site 샘플 수집 | domain gap 축소, 재검증 |

**예상 효과**: Precision 97~99% 수준

---

### Phase 4: 운영 최적화 및 Recall 검토 (장기, 8주~)

| # | 과제 | 목표 | 검증 |
|---|------|------|------|
| 4.1 | 운영 임계값 고정 | Precision 99% 달성 구간에서 최종 선택 | A/B 테스트 |
| 4.2 | Recall 개선 검토 | Precision 99% 유지 전제 하에 Recall 상향 | FN 분석, 특징/모델 보완 |
| 4.3 | 모니터링 대시보드 | FP/FN 추이, 시나리오별 성능 | 드리프트 감지 |
| 4.4 | 문서화 및 전달 | 파라미터, 임계값, 제한사항 | 운영팀 전달 |

---

## 4. 기술적 세부 전략

### 4.1 light_distortion 대응 (FP 최우선 제거)

```
현재: 15개 (정상의 1.5%), test 3개 전부 FP
목표: train 비중 5~10%, test FP 0

조치:
1. generate_ml_dataset: LIGHT_DISTORTION_COUNT 50~100으로 확대
2. synthetic.py: light_distortion 파라미터 다양화 (offset ±1~5, spike 비율 1~5%)
3. normal_train 구성 시 light_distortion 100% 포함
4. 별도 light_distortion 전용 val set으로 검증
```

### 4.2 임계값 전략 (Precision 극대화)

```
현재: Precision ≥ 99%, Recall ≥ 80% 동시 만족 시도
변경: Precision 99% 단일 목표, Recall 무제약

조치:
1. _select_threshold_precision_priority: min_recall 제거 또는 0으로 설정
2. 임계값 = max{ t : Precision(t) ≥ 0.99 } (가능한 최고 임계값)
3. 달성 불가 시: max Precision 달성, 해당 임계값 기록
```

### 4.3 특징 공학 (FP 유발 특징 억제)

```
후보:
- 제거/감쇠: light_distortion에 민감한 raw 좌표 기반 특징
- 추가: spectral_entropy, fft_dominant_freq (조명 불변)
- 정규화: RobustScaler (이상치에 덜 민감)
- 집계: per-video max 대신 상위 k% 평균 (스파이크 억제)
```

### 4.4 과적합 방지 체크리스트

- [ ] Val set에 light_distortion, micro_crack 포함
- [ ] Early stopping (val loss 기반)
- [ ] 정상 train 크기 제한 (과다 확장 시 교차검증)
- [ ] Test set 사전 고정, 재생성 금지

---

## 5. 성공 기준 및 KPI

| 단계 | Precision 목표 | FP 상한(참고) | Recall |
|------|----------------|---------------|--------|
| Phase 1 완료 | ≥ 90% | ≤ 90 | 제한 없음 |
| Phase 2 완료 | ≥ 95% | ≤ 45 | 제한 없음 |
| Phase 3 완료 | ≥ 99% | ≤ 8 | 제한 없음 |
| Phase 4 | 99% 유지 | ≤ 8 | 필요 시 80%+ 검토 |

---

## 6. 리스크 및 완화

| 리스크 | 완화 |
|--------|------|
| light_distortion 과다 증강 → 정상 분포 왜곡 | val set으로 검증, crack detect 성능 모니터링 |
| 임계값 과상향 → Recall 급락 | Phase 4에서 Recall 개선 별도 검토 |
| 합성 데이터 한계 | 실제 데이터 확보 시 재검증, domain adaptation |
| 99% 미달성 | Phase별 목표 완화(95%→97%→99%), 원인 분석 문서화 |

---

## 7. 개발 시작 순서 (권장)

1. **Phase 1.1**: light_distortion train 비중 확대 (generate_ml_dataset 수정)
2. **Phase 4.2(선행)**: 임계값 전략을 Precision 단일 목표로 변경
3. **Phase 1.4**: 정상 val 분리 및 검증
4. **Phase 1.2**: light_distortion 증강 다양화
5. 이후 Phase 2 → 3 순차 진행

---

## 8. 실행 결과 (2026-02-20)

| Loop | 변경 | DREAM Prec | PatchCore Prec | Ensemble Prec | FP(Ensemble) |
|------|------|------------|----------------|---------------|--------------|
| Baseline | - | 86.3% | 88.1% | - | - |
| 1 | ld 50개, Prec-only, thick_panel | 99.15% | 99.14% | - | - |
| 2 | ld 증강 다양화 | 99.13% | 99.12% | - | - |
| 3 | 앙상블 (both agree) | 99.13% | 99.12% | **99.65%** | 2 |
| 4 | MIN_PRECISION 0.997 | 99.83% | 99.82% | **100%** | **0** |

**결론**: Phase 1~3 전략 적용으로 Precision 99%+ 달성. 앙상블 + 고임계값(0.997)으로 FP=0, light_distortion 100% 정상 분류.

---

## 9. 참조 문서

| 문서 | 용도 |
|------|------|
| [CRACK_DETECTION_FINAL_REPORT.md](./CRACK_DETECTION_FINAL_REPORT.md) | **최종 통합 보고서** (성과, 시나리오, 성능 통합) |
| [FPCB_Crack_Detection_Final_Report.pptx](./FPCB_Crack_Detection_Final_Report.pptx) | **제출용 PPT** (삼성 표준 형식) |
| [DETECTION_IMPROVEMENT_STRATEGY.md](./DETECTION_IMPROVEMENT_STRATEGY.md) | 기존 전략 (참고) |
| [precision_progress_log.md](./precision_progress_log.md) | 루프별 개선 로그 |
| [crack_detection_analysis/](./crack_detection_analysis/) | confusion matrix, insights, 이미지 |
