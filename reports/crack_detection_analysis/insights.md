# Crack Detection Performance — Insights

## 1. Confusion Matrix Summary

### DREAM

| Metric | Value |
|--------|-------|
| True Negative (TN) | 949 |
| False Positive (FP) | 27 |
| False Negative (FN) | 114 |
| True Positive (TP) | 69 |
| Precision | 0.7188 |
| Recall | 0.3770 |
| F1 | 0.4946 |
| ROC AUC | 0.8945 |

### PatchCore

| Metric | Value |
|--------|-------|
| True Negative (TN) | 946 |
| False Positive (FP) | 30 |
| False Negative (FN) | 113 |
| True Positive (TP) | 70 |
| Precision | 0.7000 |
| Recall | 0.3825 |
| F1 | 0.4947 |
| ROC AUC | 0.9286 |

### Ensemble

| Metric | Value |
|--------|-------|
| True Negative (TN) | 956 |
| False Positive (FP) | 20 |
| False Negative (FN) | 115 |
| True Positive (TP) | 68 |
| Precision | 0.7727 |
| Recall | 0.3716 |
| F1 | 0.5018 |
| ROC AUC | 0.0000 |

## 2. Hard Subset (light_distortion, micro_crack)

### DREAM
- **light_distortion** (정상+조명왜곡): 0/1 정상으로 정확 분류, acc=0.00%
- **micro_crack** (초미세 크랙): 1/1 크랙으로 정확 분류, acc=100.00%

### PatchCore
- **light_distortion** (정상+조명왜곡): 0/1 정상으로 정확 분류, acc=0.00%
- **micro_crack** (초미세 크랙): 1/1 크랙으로 정확 분류, acc=100.00%

### Ensemble
- **light_distortion** (정상+조명왜곡): 0/1 정상으로 정확 분류, acc=0.00%
- **micro_crack** (초미세 크랙): 1/1 크랙으로 정확 분류, acc=100.00%

## 3. Vector Map Interpretation

- **Normal**: Smooth velocity/acceleration arrows, no sudden spikes.
- **Crack**: Shockwave (acceleration spike) and vibration near crack frame.

## 4. Key Insights

- **FP (False Positive)**: Normal 샘플을 크랙으로 오탐 → 과민 반응, 임계값 조정 필요.
- **FN (False Negative)**: 크랙 샘플을 정상으로 오탐 → 위험, Recall 개선 필요.
- **합성 데이터**: 실제 데이터 확보 전 로컬 검증용. 실제 데이터와의 domain gap 존재.

## 5. Detection Improvement Strategy

- **light_distortion 대응**: 조명 변화에 강건한 특징(주파수 영역, 정규화) 강화; 데이터 증강에 조명 시뮬레이션 추가.
- **micro_crack 대응**: 곡률 집중도, 가속도 스파이크 등 미세 신호 민감도 향상; crack_gain/임계값 튜닝.
- **Confusion matrix 기반**: FP↑ → 임계값 상향; FN↑ → Recall 개선(특징 추가, 모델 복잡도 증가).
