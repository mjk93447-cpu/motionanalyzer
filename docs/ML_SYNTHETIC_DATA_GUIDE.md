# ML 학습용 합성 데이터 사용 설명서

**작성일**: 2026년 2월 19일  
**데이터 규격**: [docs/SYNTHETIC_DATA_SPEC.md](SYNTHETIC_DATA_SPEC.md)  
**목표**: 벤딩 중 크랙 감지의 **Precision-Recall 최대화**

---

## 1. 핵심 목표

- **목표 1 (최우선)**: 벤딩 중 크랙 감지 — DREAM·PatchCore로 **Precision-Recall 최대화**
- 국소 감지(충격파·진동·벌어짐) + 전체 패턴 감지(과경화 징후·크랙 후 물성) 복합 활용

---

## 2. 데이터 생성

### 2.1 전체 데이터셋 (1100건)

```powershell
python scripts/generate_ml_dataset.py
```

- 1000 normal
- 80 crack_in_bending (목표 1: 50 crack + 30 uv_overcured)
- 20 pre_damaged (목표 2)

### 2.2 소규모 테스트 (115건)

```powershell
python scripts/generate_ml_dataset.py --small
```

### 2.3 합성 규칙 변경 후 재생성

1. `docs/SYNTHETIC_DATA_SPEC.md` 또는 `scripts/generate_ml_dataset.py` 수정
2. `python scripts/generate_ml_dataset.py` 재실행

---

## 3. ML 학습 시나리오

### 3.1 목표 1: 벤딩 중 크랙 (최우선)

**데이터**: normal (label=0) vs crack_in_bending (label=1)

**특징 (국소+전체)**:
- **국소**: acceleration_max, curvature_concentration, spectral_entropy, shockwave, vibration
- **전체**: 통계 특징, 고급 특징(왜도·첨도), 주파수 영역

**모델**: DREAM, PatchCore, CPD, Temporal (LSTM/GRU)

**평가**: **Precision, Recall, F1, PR AUC** 최대화

```powershell
python scripts/evaluate_goal1_ml.py      # DREAM/PatchCore, PR
python scripts/evaluate_goal1_cpd.py     # CPD 시점 정확도
```

### 3.2 목표 2: 이미 크랙된 패널

**데이터**: normal (label=0) vs pre_damaged (label=1)

**모델**: DREAM, PatchCore

```powershell
python scripts/evaluate_goal2_ml.py
```

---

## 4. 목표 달성도 측정

| 스크립트 | 출력 | 용도 |
|----------|------|------|
| `evaluate_goal1_ml.py` | `goal1_ml_evaluation.json` | **PR 최대화** (DREAM/PatchCore) |
| `evaluate_goal1_cpd.py` | `goal1_cpd_evaluation.json` | CPD 시점 정확도 |
| `evaluate_goal2_ml.py` | `goal2_ml_evaluation.json` | 목표 2 ML |
| `evaluate_goals_summary.py` | `goal_achievement_summary.md` | 종합 요약 |

---

## 5. 개발-테스트-평가 루프

1. **개발**: 합성 규칙·특징·모델 수정
2. **테스트**: `generate_ml_dataset.py` → `evaluate_goal1_ml.py`
3. **평가**: Precision, Recall, F1, PR AUC 확인
4. **전략 검토**: 결과 해석 → 수정 → 1로
