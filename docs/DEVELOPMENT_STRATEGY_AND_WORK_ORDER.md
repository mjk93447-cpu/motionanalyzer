# 중장기 개발 전략 및 상세 작업 순서

**작성일**: 2026년 2월 19일  
**기준**: [docs/PROJECT_GOALS.md](PROJECT_GOALS.md)

---

## 1. 핵심 원칙

- **통합 NG**: 벤딩 중 크랙이든, 이미 크랙된 패널 투입이든, 벤딩 후 크랙이 있으면 모두 NG
- **목표 1 최우선**: 벤딩 중 크랙 감지에 집중하여 우수한 결과 확보
- **최종 지표**: 벤딩 중 크랙 감지의 **Precision-Recall Score 최대화**

---

## 2. 목표 1 감지 전략 (하이라이트)

### 2.1 국소적 감지 (Local)

크랙 발생 직전·직후의 미세한 물리적 현상:

| 대상 | 특징 |
|------|------|
| 충격파 | acceleration_max, shockwave_amplitude |
| 진동 | FFT, spectral_entropy, vibration_frequency |
| 크랙 부위 벌어짐 | strain_surrogate, curvature_concentration |
| 속도 변화 | velocity, acceleration 시계열 |

### 2.2 전체 패턴 감지 (Global)

크랙 징후·크랙 후 물성 변화에 따른 벡터 패턴:

| 대상 | 특징 |
|------|------|
| 크랙 징후 (과경화 등) | uv_delay_ratio, uv_snap_gain |
| 크랙 후 물성 변화 | bend_angle, curvature_concentration, 통계 특징 |
| 전체 시퀀스 이상 | DREAM/PatchCore 이상 점수 |

### 2.3 DREAM·PatchCore 적극 활용

- **목표 1**에 DREAM·PatchCore를 적용하여 normal vs crack_in_bending 구분
- 국소+전체 특징을 입력으로 사용
- **Precision-Recall 최대화**를 최종 목표로 설정

---

## 3. 목표 연계 요약

| 우선순위 | 목표 | 특성 | 합성 데이터 | ML 평가 |
|----------|------|------|-------------|---------|
| **1 (최우선)** | 벤딩 중 크랙 | 국소+전체 복합 | crack, uv_overcured (80건) | **PR 최대화**, CPD, DREAM, PatchCore |
| 2 | 이미 크랙된 패널 | 전체 패턴 | pre_damage (20건) | DREAM/PatchCore ROC, PR |

---

## 4. 작업 순서 (Phase별)

### Phase 1: 합성 데이터 체계화 (완료)

| # | 작업 | 산출물 |
|---|------|--------|
| 1.1 | 합성 규칙·태그 스키마 | `docs/SYNTHETIC_DATA_SPEC.md` |
| 1.2 | ML용 데이터셋 (1000/80/20) | `scripts/generate_ml_dataset.py`, manifest |

### Phase 2: 목표 1 평가·고도화 (핵심)

| # | 작업 | 산출물 | 검증 |
|---|------|--------|------|
| 2.1 | CPD 정확도 | `scripts/evaluate_goal1_cpd.py` | crack_frame vs detected |
| 2.2 | **DREAM·PatchCore 목표 1 평가** | `scripts/evaluate_goal1_ml.py` | **Precision, Recall, F1, PR AUC** |
| 2.3 | 국소 특징 강화 | synthetic, feature extraction | 충격파·진동·곡률 집중 |
| 2.4 | 전체 패턴 특징 | FeatureExtractionConfig | 과경화 징후, 크랙 후 물성 |
| 2.5 | 목표 1 종합 리포트 | `reports/goal1_evaluation.json` | PR 최대화 추적 |

### Phase 3: 목표 2 평가 (보조)

| # | 작업 | 산출물 |
|---|------|--------|
| 3.1 | DREAM/PatchCore 목표 2 | `scripts/evaluate_goal2_ml.py` |

### Phase 4: 반복 개선 루프 (지속)

| # | 작업 | 설명 |
|---|------|------|
| 4.1 | **개발** | 합성 규칙·특징·모델 수정 |
| 4.2 | **테스트** | 데이터 재생성, 평가 스크립트 실행 |
| 4.3 | **평가·검증** | PR, F1, CPD 정확도 확인 |
| 4.4 | **전략 검토·수정** | 결과 해석 → 전략 조정 → 4.1로 |

---

## 5. 합성 데이터 규격

| 클래스 | 개수 | 목표 | 시나리오 |
|--------|------|------|----------|
| normal | 1000 | - | normal |
| crack_in_bending | 80 | **목표 1** | crack (50), uv_overcured (30) |
| pre_damaged_panel | 20 | 목표 2 | pre_damage (20) |

---

## 6. 평가 메트릭

### 목표 1 (벤딩 중 크랙) — 최우선

| 메트릭 | 설명 |
|--------|------|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × P × R / (P + R) |
| **PR AUC** | Precision-Recall 곡선 아래 면적 |
| CPD | detected_frame vs crack_frame 오차 |

### 목표 2

- DREAM/PatchCore: ROC AUC, PR AUC

---

## 7. 실행 명령어

```powershell
# 데이터 생성
python scripts/generate_ml_dataset.py
python scripts/generate_ml_dataset.py --small   # 소규모 테스트

# 목표 1 평가 (핵심)
python scripts/evaluate_goal1_cpd.py            # CPD 정확도
python scripts/evaluate_goal1_ml.py             # DREAM/PatchCore, PR (신규/확장)

# 목표 2 평가
python scripts/evaluate_goal2_ml.py
python scripts/evaluate_goal2_ml.py --small

# 목표 달성 요약
python scripts/evaluate_goals_summary.py
```

---

## 8. 인간 개발자 개입 요청 지점

- 합성 규칙 변경: `docs/SYNTHETIC_DATA_SPEC.md` 검토
- 평가 결과 해석: `reports/goal_*_evaluation.json` 검토
- PR 최대화를 위한 임계값·하이퍼파라미터 결정

---

## 9. GPU · Windows exe 활용 (오피스 PC, Cursor 불가)

**환경**: 오피스 PC (RTX 2070 Super), 보안 네트워크(Cursor 불가) → **Windows exe 내에서 모든 ML 수행**.

| 문서 | 용도 |
|------|------|
| [docs/GPU_REMOTE_STRATEGY.md](GPU_REMOTE_STRATEGY.md) | RTX 2070 Super 호환성, exe 내 ML 전략 |
| [docs/GPU_TODO_LIST.md](GPU_TODO_LIST.md) | device 자동 감지, GUI 확장, exe 빌드 TODO |

**핵심**:
- **RTX 2070 Super**: CUDA 7.5, 8GB VRAM — DREAM·Temporal 등 현재 모델 규모에 충분
- **exe 내 ML**: 합성 데이터 생성, DREAM/PatchCore 학습, 목표 평가를 GUI에서 실행
- **PyInstaller**: PyTorch+CUDA 번들, 대상 PC에 NVIDIA 드라이버 필요

---

## 10. 통합 개발 계획 (UNIFIED_DEVELOPMENT_PLAN.md)

**전체 흐름**: Phase 1 (시나리오·합성·모델 검증) → Phase 2 (GUI·시각화) → Phase 3 (exe·GitHub Actions) → Phase 4 (오피스 PC → 리포트 이메일 → Cursor 검토)

| 문서 | 용도 |
|------|------|
| [docs/UNIFIED_DEVELOPMENT_PLAN.md](UNIFIED_DEVELOPMENT_PLAN.md) | 통합 개발 계획, Phase별 체크리스트 |

---

## 11. 현재 상태 및 다음 단계 (2026-02-19)

### 완료

- Phase 1: 합성 데이터 (1000/80/20 + thick_panel 20)
- Goal 1 CPD, Goal 1 ML, Goal 2 ML
- DREAM device 자동 감지 (CUDA 사용 시 GPU)
- GUI "Synthetic & Goals" 탭 (합성 생성, 목표 평가)
- 통합 개발 계획 문서

### 다음 작업

1. **Phase 3**: exe 빌드·GitHub Actions 아티팩트 배포 확인
2. **Phase 4**: 오피스 PC 워크플로우 가이드, 리포트 이메일 절차
3. 개발 → 테스트 → 평가 → 전략 검토 루프 반복
