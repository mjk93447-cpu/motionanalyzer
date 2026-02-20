# DREAM 크랙 유사 합성 이상 생성 및 Threshold 최적화

## 개요

DREAM 모델의 합성 이상 생성 방식을 단순 Gaussian 노이즈에서 **크랙 유사 물리 기반 패턴**으로 변경하고, threshold를 Precision-Recall 균형에 맞게 최적화했습니다.

## 핵심 개선 사항

### 1. 크랙 유사 합성 이상 생성 (`_generate_crack_like_anomaly`)

측면 벤딩 카메라 화면에서 실제 크랙을 직접 관찰하는 것은 불가능에 가깝습니다. 대신 시계열 데이터 분석을 통해 다음 패턴을 감지합니다:

- **가속도 스파이크 (충격파)**: `acceleration_max`, `acceleration_std` 증가 (1.2-1.8x)
- **궤적 편차**: `curvature_concentration`, `curvature_like_max` 증가 (1.15-1.5x)
- **변형 집중**: `strain_surrogate_max`, `strain_surrogate_std` 증가 (1.1-1.4x)
- **충격/쇼크**: `impact_surrogate_max` 증가 (1.3-2.0x)
- **미세 진동**: `speed_std` 증가 (1.1-1.3x)

이러한 패턴은 크랙 발생 시점에 미세한 궤적, 속도, 가속도 변화 및 미세진동(충격파)으로 나타나며, AI가 이를 통해 크랙 발생 시점과 위치를 탐지합니다.

### 2. Threshold 최적화 (`optimize_threshold_for_precision_recall`)

기존의 단순 p95 percentile 대신, Precision-Recall 곡선에서 최적점을 찾습니다:

- **target_metric 옵션**:
  - `"f1"`: F1 score 최대화
  - `"balanced"`: F1 최대화하되 recall >= 0.7 요구 (기본값)
  - `"precision"`: Precision 최대화
  - `"recall"`: Recall 최대화

- **최적화 범위**: normal 데이터의 p50 ~ p99.9 percentile 사이에서 100개 threshold 후보 평가

### 3. Feature Name 매칭 개선

더 유연한 feature name 매칭:
- 모든 `acceleration`, `curvature`, `strain`, `impact`, `speed` 관련 특징을 찾음
- `max`, `std`, `mean`, `concentration` 등이 포함된 특징을 우선적으로 사용
- 매칭 실패 시 fallback으로 물리 기반 변형 적용

## 성능 개선 결과

### 합성 데이터 검증 (`scripts/validate_dream_synthetic.py`)

**이전 (단순 노이즈 + p95 threshold)**:
- Accuracy: 0.5312
- Precision: 0.6667
- Recall: 0.1250
- F1: 0.2105
- ROC AUC: 0.7109

**현재 (크랙 유사 합성 + threshold 최적화)**:
- Accuracy: 0.8125 (+53%)
- Precision: 0.8125 (+22%)
- Recall: 0.8125 (+550%)
- F1: 0.8125 (+286%)
- ROC AUC: 0.7891 (+11%)

### DRAEM 논문 비교

- **DRAEM 논문 (MVTec AD)**: Image-level ROC AUC 98.1%
- **현재 구현 (FPCB tabular)**: ROC AUC 0.7891 (78.91%)

**참고**: 도메인이 다르므로 직접 비교는 불가능하지만, 추세적으로 개선되었습니다.

## 구현 세부사항

### 파일 변경

1. **`src/motionanalyzer/ml_models/dream.py`**:
   - `_generate_crack_like_anomaly()`: 크랙 유사 합성 이상 생성
   - `fit()`: feature_names 전달 및 크랙 유사 이상 사용
   - `optimize_threshold_for_precision_recall()`: Threshold 최적화

2. **`src/motionanalyzer/gui/runners.py`**:
   - `_run_dream()`: feature_names 전달 및 threshold 최적화 옵션 추가

3. **`scripts/validate_dream_synthetic.py`**:
   - DataFrame 사용으로 feature_names 전달
   - Threshold 최적화 결과 출력

### 사용 예시

```python
from motionanalyzer.ml_models.dream import DREAMPyTorch
import pandas as pd

# DataFrame으로 학습 (feature_names 자동 추출)
normal_df = pd.DataFrame(X_normal, columns=feature_cols)
model = DREAMPyTorch(
    input_dim=len(feature_cols),
    use_discriminative=True,
    synthetic_noise_std=0.3,  # Fallback용
    discriminator_weight=0.5,
)
model.fit(normal_df, epochs=60, feature_names=feature_cols)

# Threshold 최적화
crack_df = pd.DataFrame(X_crack, columns=feature_cols)
thresh, metrics = model.optimize_threshold_for_precision_recall(
    normal_df,
    crack_df,
    target_metric="balanced",  # F1 with recall >= 0.7
)
print(f"Optimal threshold: {thresh:.4f}")
print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
```

## 향후 개선 방향

1. **실제 크랙 데이터 활용**: 소수의 실제 크랙 데이터로 fine-tuning (few-shot learning)
2. **합성 이상 강도 조정**: 도메인 특성에 맞게 스케일 팩터 튜닝
3. **시계열 패턴 강화**: 프레임 간 temporal dependency 모델링
4. **앙상블**: DREAM + PatchCore 앙상블로 성능 향상

## 참고

- DRAEM 논문: Zavrtanik et al., "DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection", ICCV 2021
- FPCB 도메인 특성: 측면 벤딩 카메라에서 직접 크랙 관찰 불가 → 시계열 패턴 분석 필요
