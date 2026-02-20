# Change Point Detection for FPCB Crack Detection

## 개요

Change Point Detection은 벤딩 진행 중 **크랙이 발생한 정확한 프레임**을 감지하는 기술입니다. 측면 벤딩 카메라에서 직접 크랙을 관찰할 수 없으므로, 시계열 데이터 분석을 통해 미세한 패턴 변화를 감지합니다.

## 구현된 알고리즘

### 1. CUSUM (Cumulative Sum)

**원리**: 누적 편차가 임계값을 초과할 때 변화점을 감지합니다.

**수식**:
- Upper: `C_i^+ = max[0, x_i − (T+K) + C_{i-1}^+]`
- Lower: `C_i^- = max[0, (T−K) − x_i + C_{i-1}^-]`

여기서 T는 타겟 값, K는 민감도 파라미터입니다.

**특징**:
- 실시간 처리 가능 (온라인 알고리즘)
- 간단한 가정만 필요
- 작은 변화도 감지 가능 (≤ 1.5σ)

**참고 문헌**: Page, E.S. (1954). "Continuous Inspection Schemes". Biometrika, 41(1/2), 100-115.

### 2. Window-based Detection

**원리**: 슬라이딩 윈도우로 통계적 변화를 비교합니다.

**방법**:
- 두 윈도우 간 평균/분산 비율 계산
- 임계값을 초과하면 변화점으로 판단

**특징**:
- 통계적 변화 감지에 효과적
- 파라미터 조정이 비교적 쉬움

### 3. PELT (Pruned Exact Linear Time)

**원리**: 동적 프로그래밍으로 최적 분할점을 찾습니다.

**특징**:
- 계산 복잡도: O(CKn) (K는 변화점 수, n은 샘플 수)
- 최적해 보장
- `ruptures` 라이브러리 필요

**참고 문헌**: Killick, R., Fearnhead, P., & Eckley, I.A. (2012). "Optimal detection of changepoints with a linear computational cost". Journal of the American Statistical Association, 107(500), 1590-1598.

## 사용 예시

### 기본 사용

```python
from motionanalyzer.time_series.changepoint import CUSUMDetector, WindowBasedDetector
import numpy as np

# 시계열 신호 준비 (예: acceleration_max)
signal = np.array([...])  # 프레임별 특징 값

# CUSUM 감지기
cusum_detector = CUSUMDetector(threshold=2.0, min_size=5)
result = cusum_detector.detect(signal)
print(f"Change points: {result.change_points}")

# Window-based 감지기
window_detector = WindowBasedDetector(window_size=10, threshold_ratio=1.5, min_size=5)
result = window_detector.detect(signal)
print(f"Change points: {result.change_points}")
```

### FPCB 데이터에 적용

```python
from motionanalyzer.analysis import run_analysis
from motionanalyzer.time_series.changepoint import CUSUMDetector
import pandas as pd

# 데이터셋 분석
output_dir = Path("output")
run_analysis(dataset_path, output_dir, fps=30.0)

# 벡터 로드
vectors = pd.read_csv(output_dir / "vectors.csv")

# 프레임별 특징 추출
frame_features = vectors.groupby("frame")["acceleration"].max().reset_index()
acceleration_signal = frame_features["acceleration"].values

# 변화점 감지
detector = CUSUMDetector(threshold=2.0, min_size=5)
result = detector.detect(acceleration_signal)

# 크랙 발생 프레임 확인
crack_frames = result.change_points
print(f"Crack detected at frames: {crack_frames}")
```

## 파라미터 튜닝 가이드

### CUSUM 파라미터

- **threshold**: 감지 임계값 (기본값: 5.0)
  - 높을수록: False Positive 감소, False Negative 증가
  - 낮을수록: 더 민감하게 감지, False Positive 증가
  - 권장 범위: 1.0 ~ 10.0

- **target**: 타겟 값 (기본값: None = 신호 평균)
  - 신호의 기준값 설정
  - None이면 자동으로 평균 사용

- **sensitivity**: 민감도 파라미터 K (기본값: None = 0.5 * std)
  - 변화 감지 민감도 조절
  - 작을수록: 더 작은 변화도 감지
  - 클수록: 큰 변화만 감지

- **min_size**: 변화점 간 최소 거리 (기본값: 3)
  - 너무 가까운 변화점 필터링
  - 프레임 단위

### Window-based 파라미터

- **window_size**: 슬라이딩 윈도우 크기 (기본값: 10)
  - 작을수록: 빠른 변화 감지, 노이즈에 민감
  - 클수록: 안정적 감지, 느린 변화 감지

- **threshold_ratio**: 변화 비율 임계값 (기본값: 2.0)
  - 평균/분산 비율이 이 값을 초과하면 변화점으로 판단
  - 높을수록: 큰 변화만 감지

- **step_size**: 윈도우 이동 간격 (기본값: 1)
  - 1이면 모든 프레임 검사
  - 크면 계산 속도 향상, 정확도 약간 감소

## 검증 결과

합성 데이터 검증 (`scripts/validate_changepoint_synthetic.py`) 결과:

- **CUSUM**: 크랙 발생 시점 감지 성공 (예상 범위 25-45 프레임 내 감지)
- **Window-based**: 크랙 발생 시점 감지 성공
- **PELT**: `ruptures` 라이브러리 설치 필요 (선택적)

## 향후 개선 방향

1. **파라미터 자동 튜닝**: 검증 세트 기반 최적 파라미터 탐색
2. **다중 특징 결합**: acceleration, curvature, strain 등 여러 특징 동시 분석
3. **앙상블**: 여러 감지기 결과 결합으로 정확도 향상
4. **실제 크랙 데이터 검증**: 현미경 검사로 확인된 크랙 데이터로 검증

## 파일 구조

```
src/motionanalyzer/time_series/
├── __init__.py
└── changepoint.py          # Change Point Detection 구현
```

## 테스트

```bash
# 단위 테스트
python -m pytest tests/test_changepoint.py -v

# 합성 데이터 검증
python scripts/validate_changepoint_synthetic.py
```

## 의존성

- **필수**: numpy, pandas
- **선택적**: `ruptures` (PELT 알고리즘용)
  ```bash
  pip install ruptures
  ```
