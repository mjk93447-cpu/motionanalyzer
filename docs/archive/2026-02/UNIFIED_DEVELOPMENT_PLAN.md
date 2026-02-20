# 통합 개발 계획 (Unified Development Plan)

**작성일**: 2026년 2월 19일  
**기준**: [PROJECT_GOALS.md](PROJECT_GOALS.md), [GPU_REMOTE_STRATEGY.md](GPU_REMOTE_STRATEGY.md)

---

## 1. 전체 흐름 개요

### 전략 우선순위 (2026-02-19)

**실제 데이터 확보 전**: 로컬 합성데이터로 크랙검출 성능 검증 (confusion matrix, 벡터맵, 인사이트)  
**실제 데이터 확보 후**: GPU·exe·오피스 PC 워크플로우 개발

```
[Phase 1] 로컬: 시나리오·합성데이터·모델 검증 (GPU 없이)
    ↓
[Phase 2] GUI·시각화 고도화
    ↓
[Phase 3] Windows exe 구축 + GitHub Actions 아티팩트 배포
    ↓
[Phase 4] 오피스 PC (RTX 2070 Super) 설치 → 분석 → 리포트 이메일 → Cursor Composer 검토
    ↓
[GPU 개발] 실제 데이터 확보 시 GPU·CUDA 활용 본격화
```

---

## 2. Phase 1: 시나리오·합성데이터·모델 검증 (GPU 없이)

**목표**: 다양한 시나리오와 충분한 합성데이터로 모델별 성능 검증. CPU만으로 완료.

### 2.1 시나리오 확장

| 시나리오 | 현재 | 확장 | 목표 |
|----------|------|------|------|
| normal | 1000 | 1000 | 정상 기준선 |
| crack | 50 | 50 | 목표 1 (국소) |
| uv_overcured | 30 | 30 | 목표 1 (과경화) |
| pre_damage | 20 | 20 | 목표 2 |
| **thick_panel** | 0 | **20** | 변형 시나리오 (정상에 가까움) |

**총 1120건** (기존 1100 + thick_panel 20)

### 2.2 합성 데이터 규모

| 클래스 | 개수 | 시나리오 |
|--------|------|----------|
| normal | 1000 | normal |
| crack_in_bending | 80 | crack (50), uv_overcured (30) |
| pre_damaged | 20 | pre_damage |
| thick_panel | 20 | thick_panel (경계 케이스) |

### 2.3 모델별 성능 검증 (CPU)

| # | 작업 | 산출물 | 검증 |
|---|------|--------|------|
| 1.1 | generate_ml_dataset에 thick_panel 추가 | `scripts/generate_ml_dataset.py` | manifest에 thick_panel 20 |
| 1.2 | evaluate_goal1_cpd.py | CPD 정확도 | crack_frame vs detected |
| 1.3 | evaluate_goal1_ml.py | DREAM·PatchCore PR | Precision, Recall, F1 |
| 1.4 | evaluate_goal2_ml.py | 목표 2 ML | ROC, PR AUC |
| 1.5 | evaluate_goals_summary.py | 종합 리포트 | goal_achievement_summary.md |
| 1.6 | 모델별 벤치마크 스크립트 | `scripts/benchmark_models.py` | DREAM vs PatchCore vs Temporal 비교 |

### 2.4 산출물

- `data/synthetic/ml_dataset/` (1120건)
- `reports/goal1_cpd_evaluation.json`
- `reports/goal1_ml_evaluation.json`
- `reports/goal2_ml_evaluation.json`
- `reports/goal_achievement_summary.md`
- `reports/model_benchmark.json`

---

## 3. Phase 2: GUI·시각화 고도화

**목표**: 모든 분석 프로세스 결과 시각화, GUI 기능 최대화.

### 3.1 GUI 기능 확장

| # | 작업 | 설명 |
|---|------|------|
| 2.1 | **합성 데이터 생성** 탭 | 전체/소규모 생성, 진행률, 완료 알림 |
| 2.2 | **목표 평가** 탭 | Goal 1/2 평가 실행, Precision·Recall·F1 표시 |
| 2.3 | **리포트 뷰어** | goal_achievement_summary.md, JSON 요약 표시 |
| 2.4 | CPD 결과 시각화 강화 | 감지 시점 마커, 프레임별 점수 |
| 2.5 | 모델 벤치마크 결과 표시 | DREAM vs PatchCore 비교 차트 |

### 3.2 시각화 개선

| # | 작업 | 설명 |
|---|------|------|
| 2.6 | 벡터 맵 SI 단위 일관성 | mm/px 스케일 입력 시 미터 좌표 |
| 2.7 | 이상 점수 시계열 플롯 | 프레임별 anomaly score |
| 2.8 | PR/ROC 곡선 표시 (선택) | 평가 결과 시각화 |

---

## 4. Phase 3: Windows exe + GitHub Actions

**목표**: 합성데이터 생성·ML·AI 분석을 모두 수행하는 exe 구축, 아티팩트 배포.

### 4.1 exe 포함 기능

| 기능 | 포함 |
|------|------|
| 합성 데이터 생성 | ✅ |
| DREAM·PatchCore·Temporal 학습 | ✅ |
| CPD (CUSUM/Window/PELT) | ✅ |
| 목표 1/2 평가 | ✅ |
| 리포트 생성 | ✅ |
| GPU 자동 감지 (RTX 2070 Super) | ✅ |

### 4.2 코드 수정

| # | 작업 | 파일 |
|---|------|------|
| 3.1 | DREAM device 자동 감지 | `dream.py` |
| 3.2 | runners에 device 전달 | `runners.py` |
| 3.3 | GUI에 합성·평가 탭 | `desktop_gui.py` |
| 3.4 | PyInstaller spec (ML+CUDA) | `motionanalyzer_gui.spec` 또는 `build_exe.ps1` |

### 4.3 GitHub Actions

| # | 작업 | 설명 |
|---|------|------|
| 3.5 | 아티팩트 업로드 확인 | `motionanalyzer-gui.exe`, `motionanalyzer-gui-ml.exe` |
| 3.6 | retention-days 30 유지 | 기존 설정 |
| 3.7 | 빌드 시 ML 의존성 포함 | `pip install -e ".[ml]"` |

---

## 5. Phase 4: 오피스 PC 워크플로우

**목표**: exe 설치 → 분석 → 리포트 이메일 → Cursor Composer 검토.

### 5.1 오피스 PC (RTX 2070 Super)

| 단계 | 작업 |
|------|------|
| 1 | GitHub Actions 아티팩트에서 exe 다운로드 |
| 2 | 오피스 PC에 설치 (또는 portable 실행) |
| 3 | 합성 데이터 생성 (또는 기존 데이터 사용) |
| 4 | 목표 1/2 평가 실행 |
| 5 | `reports/` 폴더 내 리포트 파일 확인 |
| 6 | 리포트를 이메일로 전송 |

### 5.2 Cursor Composer 1.5 검토

| 입력 | 출력 |
|------|------|
| 이메일로 받은 리포트 (goal_achievement_summary.md, JSON) | 추가 개발 항목 검토 |
| PR, Recall, F1 수치 | 개선 방향 제안 |
| CPD mean_error | 시점 추정 정교화 여부 |

---

## 6. 작업 순서 (실행 체크리스트)

### Phase 1 (GPU 없이)

- [x] 1.1 generate_ml_dataset에 thick_panel 20건 추가
- [x] 1.2 SYNTHETIC_DATA_SPEC 업데이트
- [x] 1.3 manifest 스키마에 thick_panel 반영
- [ ] 1.4 benchmark_models.py 작성 (선택)
- [x] 1.5 전체 파이프라인 실행·검증

### Phase 2 (GUI·시각화)

- [x] 2.1 합성 데이터 생성 탭
- [x] 2.2 목표 평가 탭
- [x] 2.3 리포트 뷰어 (Summary 버튼)
- [ ] 2.4 CPD 시각화 강화 (선택)

### Phase 3 (exe·CI)

- [x] 3.1 DREAM device 자동 감지
- [x] 3.2 GUI 통합 (Synthetic & Goals 탭)
- [x] 3.3 build_exe.ps1 ML 버전 확인
- [ ] 3.4 GitHub Actions 아티팩트 배포 확인 (push 시 자동)

### Phase 4 (오피스 PC)

- [x] 4.1 exe 다운로드·설치 가이드 (OFFICE_PC_WORKFLOW.md)
- [x] 4.2 리포트 이메일 전송 절차
- [x] 4.3 Cursor Composer 검토 템플릿

---

## 7. 예상 소요 시간

| Phase | 예상 |
|-------|------|
| Phase 1 | 1–2일 |
| Phase 2 | 2–3일 |
| Phase 3 | 1일 |
| Phase 4 | 수동 (오피스 PC) + 검토 |

---

## 8. 문서 연계

- `DEVELOPMENT_STRATEGY_AND_WORK_ORDER.md`: 본 계획의 상세 참조
- `GPU_TODO_LIST.md`: Phase 3 코드 수정 상세
- `SYNTHETIC_DATA_SPEC.md`: 시나리오·규모 정의
