# Phase B 개발 인사이트 종합 문서

**작성일**: 2026년 2월 18일  
**범위**: Phase B 모든 모델 개발 및 검증 과정에서 얻은 핵심 인사이트

---

## 1. 모델 성능 벤치마크 결과

### 1.1 Baseline Features (21 features)

| 모델 | ROC AUC | PR AUC | Precision | Recall | F1 |
|------|---------|--------|-----------|--------|-----|
| DREAM | 0.913 | 0.953 | 1.000 | 0.672 | 0.804 |
| PatchCore | 0.908 | 0.954 | 0.982 | 0.775 | 0.866 |
| Ensemble | 0.908 | 0.954 | 0.982 | 0.775 | 0.866 |
| Temporal | 0.100 | 0.286 | 0.286 | 1.000 | 0.444 |

**핵심 발견**:
- DREAM과 PatchCore가 유사한 성능 (ROC AUC ~0.91)
- Ensemble이 단일 모델 대비 큰 향상 없음 (가중치 최적화 필요)
- Temporal 모델은 낮은 성능 (데이터 분할 및 시계열 구조 보존 이슈)

### 1.2 Advanced Features (75 features)

| 모델 | ROC AUC | PR AUC | Precision | Recall | F1 |
|------|---------|--------|-----------|--------|-----|
| DREAM+Advanced | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Temporal+Advanced | 0.100 | 0.286 | 0.286 | 1.000 | 0.444 |

**핵심 발견**:
- DREAM+Advanced가 완벽한 성능 (과적합 가능성 높음)
- Temporal 모델은 고급 특징에도 불구하고 성능 향상 없음

---

## 2. 고급 특징 엔지니어링 과적합 분석

### 2.1 특징-레이블 상관관계 분석

**Baseline Features (21개)**:
- 고상관 특징 (|corr| > 0.5): 7개
- 최대 상관계수: 0.657 (`est_max_strain`, `strain_surrogate_max`)

**Advanced Features (75개)**:
- 고상관 특징 (|corr| > 0.5): 19개 (2.7배 증가)
- 최대 상관계수: 0.996 (`acceleration_mean_kurtosis`, `acceleration_mean_skewness`)

### 2.2 과적합 지표

1. **매우 높은 상관관계**: 고급 특징 중 일부가 레이블과 거의 완벽한 상관관계 (0.996)
   - `acceleration_mean_kurtosis`: -0.996
   - `acceleration_mean_skewness`: 0.996
   - `acceleration_mean_spectral_power`: 0.992

2. **특징 수 증가**: Baseline 대비 2.7배 더 많은 고상관 특징

3. **완벽한 성능**: DREAM+Advanced가 ROC AUC 1.000 (합성 데이터에서만)

### 2.3 과적합 원인 분석

**가능한 원인**:
1. **합성 데이터 패턴**: 합성 데이터 생성 로직이 특정 통계적 패턴(왜도, 첨도)을 일관되게 생성
2. **물리 모델 의존성**: 고급 특징들이 합성 데이터의 물리 모델과 직접적으로 연결됨
3. **특징 공선성**: 여러 고급 특징들이 서로 높은 상관관계를 가짐

**검증 필요 사항**:
- 실제 데이터에서의 성능 검증 필수
- 다양한 시드와 시나리오로 합성 데이터 다양성 확보
- 특징 선택(feature selection)을 통한 차원 축소 고려

---

## 3. 앙상블 모델 분석

### 3.1 현재 성능

- Ensemble (DREAM + PatchCore): ROC AUC 0.908
- DREAM 단독: ROC AUC 0.913
- **결과**: Ensemble이 단일 모델보다 약간 낮은 성능

### 3.2 앙상블 효과가 낮은 이유

1. **유사한 성능**: DREAM과 PatchCore가 거의 동일한 성능 (0.913 vs 0.908)
2. **가중치 최적화 한계**: 현재 가중치 최적화가 검증 세트 기준으로 수행되지만, 두 모델의 예측이 유사하여 다양성 부족
3. **예측 상관관계**: 두 모델의 예측이 높은 상관관계를 가질 가능성

### 3.3 개선 방향

1. **다양성 확보**: 서로 다른 접근 방식의 모델 결합 (예: Temporal + DREAM)
2. **스태킹 전략**: 메타 분류기를 사용한 스태킹 앙상블 재검토
3. **가중치 최적화 개선**: 더 다양한 메트릭(F1, PR AUC) 기반 최적화

---

## 4. Temporal 모델 이슈 분석

### 4.1 성능 문제

- ROC AUC: 0.100 (baseline 및 advanced 모두)
- Recall: 1.000, Precision: 0.286
- **해석**: 모든 것을 anomaly로 예측 (threshold가 너무 낮음)

### 4.2 가능한 원인

1. **데이터 분할 문제**: 시계열 구조 보존 부족
   - 해결: 데이터셋 레벨 분할 후 시퀀스 생성 (개선 완료)
   - 여전히 낮은 성능 → 추가 조사 필요

2. **Threshold 설정 문제**: Normal 데이터의 95th percentile이 너무 낮게 설정됨
   - 해결: Threshold 최적화 전략 재검토 필요

3. **시퀀스 길이**: 현재 10 프레임 사용 (문헌 기준 최적)
   - 추가 실험: 다른 시퀀스 길이 테스트

4. **데이터 불균형**: Normal 데이터만으로 학습하지만, 테스트에서 crack 데이터 비율이 높을 수 있음

### 4.3 개선 방향

1. **Threshold 최적화**: Precision-Recall 곡선 기반 최적 threshold 탐색
2. **하이퍼파라미터 튜닝**: 시퀀스 길이, hidden dimension, learning rate 조정
3. **데이터 균형**: 학습 데이터와 테스트 데이터의 클래스 분포 확인
4. **문헌 재검토**: 시계열 이상 감지 모델의 best practices 재확인

---

## 5. 데이터 누설 방지 (Data Leakage Prevention)

### 5.1 정규화 누설 방지

**문제**: 전체 데이터로 정규화 통계 계산 시 crack 데이터가 통계에 영향

**해결**: `normalize_features` 함수에 `fit_df` 파라미터 추가
- Normal 데이터만으로 mean/std 계산
- 계산된 통계를 전체 데이터에 적용

**효과**: 라벨 누설 방지, 더 현실적인 성능 평가

### 5.2 시계열 데이터 분할

**문제**: 시계열 데이터를 랜덤하게 분할하면 미래 정보가 과거 학습에 영향

**해결**: 데이터셋 레벨 분할 후 시퀀스 생성
- 데이터셋을 train/test로 먼저 분할
- 각 데이터셋 내에서 시계열 순서 보존
- 시퀀스는 분할 후에 생성

**문헌 근거**: 
- "Hidden Leaks in Time Series Forecasting" (2025)
- Sliding window cross-validation이 시계열 이상 감지에 효과적

---

## 6. 특징 엔지니어링 인사이트

### 6.1 효과적인 특징

**Baseline Features**:
- `strain_surrogate_max`: 상관계수 0.657
- `stress_surrogate_max`: 상관계수 0.657
- `strain_surrogate_std`: 상관계수 0.623

**Advanced Features** (과적합 주의):
- `acceleration_mean_kurtosis`: 상관계수 -0.996
- `acceleration_mean_skewness`: 상관계수 0.996
- `acceleration_mean_spectral_power`: 상관계수 0.992

### 6.2 특징 선택 전략

1. **과적합 방지**: 모든 고급 특징을 항상 포함하지 말고, 모델 성능에 따라 선택
2. **차원 축소**: 고상관 특징들 간의 공선성 고려, PCA 또는 특징 선택 적용 고려
3. **도메인 지식**: 물리적으로 의미 있는 특징 우선 선택

---

## 7. 평가 메트릭 정책

### 7.1 사용 메트릭

- **ROC AUC**: Anomaly rate 변화에 비교적 안정적 (상대 비교/회귀 테스트)
- **PR AUC (AUCPR)**: Precision-Recall 균형 직접 반영 (실제 운영 의사결정에 가까움)
- **Precision, Recall, F1**: Threshold 기반 이진 분류 성능

### 7.2 메트릭 선택 이유

**문헌 근거**:
- ADBench (Han et al.): 평가 프로토콜의 중요성 강조
- Time-series anomaly metric survey ("Metric Maze", 2023)

**실무 고려사항**:
- False positive 비용이 높은 경우: Precision 중시
- False negative 비용이 높은 경우: Recall 중시
- 균형이 필요한 경우: F1 또는 PR AUC 사용

---

## 8. 다음 단계 권장사항

### 8.1 즉시 개선 가능

1. **Temporal 모델 개선**
   - Threshold 최적화 전략 재검토
   - 하이퍼파라미터 그리드 서치
   - 더 많은 학습 데이터 사용

2. **앙상블 다양성 확보**
   - Temporal + DREAM 앙상블 시도
   - 스태킹 전략 재검토

3. **고급 특징 선택**
   - 특징 중요도 기반 선택
   - 공선성 제거
   - 실제 데이터에서 검증

### 8.2 실제 데이터 확보 후

1. **과적합 검증**: 실제 데이터에서 고급 특징의 성능 확인
2. **모델 캘리브레이션**: 실제 데이터 기반 threshold 재설정
3. **Few-shot Fine-tuning**: 실제 크랙 데이터로 모델 미세조정

---

## 9. 참고 문헌

1. **시계열 데이터 누설**: "Hidden Leaks in Time Series Forecasting" (2025)
2. **시계열 이상 감지 평가**: "Temporal cross-validation impacts multivariate time series subsequence anomaly detection evaluation" (2025)
3. **LSTM 시계열**: "Master lstm time series: windowing, validation, deploy" (Upscend Blog)
4. **평가 메트릭**: ADBench (Han et al.), "Metric Maze" (2023)

---

## 10. 결론

Phase B 개발을 통해 다음을 확인:

1. **DREAM과 PatchCore가 유사한 성능**을 보이며, 합성 데이터에서 효과적
2. **고급 특징 엔지니어링이 성능을 크게 향상**시키지만, 합성 데이터에서 과적합 가능성
3. **앙상블이 단일 모델 대비 큰 향상 없음** - 다양성 확보 필요
4. **Temporal 모델은 추가 개선 필요** - 시계열 구조 보존 및 threshold 최적화
5. **데이터 누설 방지가 중요** - 정규화 및 시계열 분할 시 주의 필요

**실제 데이터 확보 후 재검증이 필수적**이며, 합성 데이터에서의 성능은 참고용으로만 사용해야 함.
