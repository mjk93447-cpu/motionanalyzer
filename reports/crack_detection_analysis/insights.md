# Crack Detection — Consolidated Insights

**최종 갱신**: 2026-02-25  
**데이터셋**: DS-260223-ml-fp-20k-60f (ml_dataset_fp_focused)  
**출력**: `analysis.json` (재생성 시 이 파일은 분석 스크립트가 덮어씀. 핵심 지식만 유지)

---

## 1. 현재 성능 (DS-260223-ml-fp-20k-60f 기준)

| 모델 | TN | FP | FN | TP | Precision | Recall | Normal FP Rate |
|------|----|----|----|-----|-----------|--------|----------------|
| DREAM | 800 | 0 | 1 | 50 | 1.0000 | 0.9804 | 0.0000% |
| PatchCore | 800 | 0 | 1 | 50 | 1.0000 | 0.9804 | 0.0000% |
| Ensemble | 800 | 0 | 1 | 50 | 1.0000 | 0.9804 | 0.0000% |

- **Hard subset**: light_distortion 50/50 정상 분류, micro_crack 10/10 크랙 분류
- **임계값**: normal_fp_max=0 (val) 기준

---

## 2. 핵심 지식 (압축)

### 2.1 Domain Gap

- **합성 crack 비율**: ~5–20% (학습/평가)
- **실제 공정 crack 비율**: ~0.01% (만분의 일)
- **영향**: 합성에서 Precision 100%여도 실제 0.01% 환경에서는 FP 1개가 precision에 큰 영향. 실제 데이터 확보 후 재검증 필수.

### 2.2 FP(오탐) 원인

- light_distortion: 조명 변화 → 정상인데 크랙으로 오탐
- 정상 변동성: 노이즈, 굽힘 초기/말기 스파이크
- 경계 케이스: thick_panel 등

### 2.3 적용된 전략

1. **fp_focused 데이터셋**: Normal 94%, Crack 6% — normal 오탐에 초점
2. **앙상블**: DREAM ∧ PatchCore (둘 다 Crack 시에만 Crack)
3. **임계값**: normal_fp_max=0, MIN_PRECISION 0.997
4. **light_distortion**: train 비중 확대, 증강 다양화

### 2.4 Vector Map 해석

- **Normal**: 부드러운 속도/가속도 화살, 급격한 스파이크 없음
- **Crack**: 크랙 프레임 근처 충격파(가속도 스파이크), 진동

### 2.5 개선 방향

- **FP↑**: 임계값 상향
- **FN↑**: Recall 개선 (특징 추가, crack_gain/임계값 튜닝)
- **light_distortion**: 주파수 영역·정규화 강화, 조명 시뮬레이션 증강
- **micro_crack**: 곡률 집중도, 가속도 스파이크 민감도 향상

---

## 3. 데이터셋 선택

| 용도 | 권장 |
|------|------|
| 학습·검증·논문 메인 | DS-260223-ml-fp-20k-60f |
| 100k 교차·일반화 | DS-260220-ml-100k-100k-60f |
| 목표 2(pre_damage) | DS-260220-ml-100k-100k-60f |

---

## 4. 재생성

```powershell
python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --normal-fp-max 0
```
