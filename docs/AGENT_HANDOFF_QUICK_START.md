# 에이전트 핸드오프 · 즉시 실행 가이드

**작성일**: 2026년 2월 25일  
**목적**: 새 에이전트가 캐시 없이 프로젝트를 즉시 이어받아 실행할 수 있도록 정리

---

## 1. 로드맵 현황 (최종 수정: 2026-02-20)

**파일**: `docs/DEVELOPMENT_ROADMAP_FINAL.md`

### 진행률: 약 85%

| Phase | 상태 | 비고 |
|-------|------|------|
| **Phase A.1** | ✅ 완료 | EXE 빌드, Analyze 탭 확장 (Physics/DREAM/PatchCore/Ensemble) |
| **Phase A.2** | ✅ 완료 | Change Point Detection GUI (Time Series Analysis 탭) |
| **Phase B.1** | ✅ 완료 | 합성 데이터 물리 현상 (충격파, 미세 진동) |
| **Phase B.2** | ✅ 완료 | 앙상블 (DREAM+PatchCore) |
| **Phase B.3** | ✅ 완료 | Temporal Modeling (LSTM/GRU) |
| **Phase B.4** | ✅ 완료 | 고급 특징 엔지니어링 |
| **Phase B.5** | ✅ 완료 | CPD 고도화 (자동 튜닝, 다중 특징, 앙상블) |
| **Phase C** | ⏸ 대기 | 실제 데이터 확보 후 Few-shot, Contrastive Learning |
| **Phase D** | 📋 신규 | 사용자 시나리오 기반 고도화 (스케일, 배치 분석 등) |

### 다음 우선 작업 (목표 1 우선)

1. **Temporal 모델 개선** — 벤치마크 ROC AUC 0.25, 시계열 구조 보존/CPD 연계
2. **CPD 정확도** — 크랙 발생 시점(프레임) 정확도 향상
3. **충격파·진동 감지 강화**
4. **고급 특징 과적합 관리** — DREAM+Advanced ROC 1.0 (합성 데이터 과적합 주의)

---

## 2. 환경 세팅 (한 번에)

### 2.1 GPU/ML 파이프라인용 (권장)

```powershell
cd c:\motionanalyzer
.\scripts\setup_gpu_env.ps1
```

**수행 내용**:
- `.venv-gpu` 가상환경 생성
- `pip install -e ".[ml]"` + PyTorch (CUDA 12.1)
- Jupyter, ipykernel, joblib
- Jupyter 커널 `motionanalyzer-gpu` 등록

### 2.2 기본 개발용 (GUI, 테스트)

```powershell
cd c:\motionanalyzer
.\scripts\bootstrap.ps1
```

**수행 내용**:
- `.venv` 생성, dev 의존성, pre-commit

### 2.3 GPU 검증

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 3. 실행 도구 목록

### 3.1 파이프라인 (PowerShell)

| 스크립트 | 용도 |
|----------|------|
| `.\scripts\run_full_pipeline.ps1` | 100k 데이터 생성 → ML 분석 → 논문 리포트 (일괄) |
| `.\scripts\run_gui.ps1` | GUI 실행 (기본 `.venv`, `.venv-gpu` 자동 fallback) |
| `.\scripts\run_gui.ps1 -ML` | GUI 실행 (DREAM/PatchCore용 `.venv-gpu` 명시) |
| `.\scripts\build_exe.ps1` | EXE 빌드 (경량) |
| `.\scripts\build_exe.ps1 -IncludeML` | EXE 빌드 (ML 포함) |
| `.\scripts\setup_gpu_env.ps1` | GPU 환경 일괄 세팅 |
| `.\scripts\bootstrap.ps1` | 기본 개발 환경 세팅 |

### 3.2 파이프라인 (Bash, Linux/WSL)

| 스크립트 | 용도 |
|----------|------|
| `./scripts/run_ml_pipeline_gpu.sh` | 합성 데이터 → Goal1/Goal2 ML 평가 → 요약 |

### 3.3 Python 스크립트 (핵심)

| 스크립트 | 용도 |
|----------|------|
| `python scripts/generate_ml_dataset.py --scale 100k --workers 4` | 100k 합성 데이터 생성 |
| `python scripts/analyze_crack_detection.py` | DREAM + PatchCore 분석 |
| `python scripts/generate_final_report_docx.py` | 논문 리포트 재생성 |
| `python scripts/benchmark_phase_b_comprehensive.py` | Phase B 종합 벤치마크 |
| `python scripts/evaluate_synthetic_dataset_quality.py` | QA 게이트: 합성 데이터 품질 |
| `python scripts/validate_enhanced_dream.py` | QA 게이트: DREAM 정밀 검증 |
| `python scripts/evaluate_goal1_ml.py` | 목표 1 ML 평가 |
| `python scripts/evaluate_goal2_ml.py` | 목표 2 ML 평가 |
| `python scripts/evaluate_goals_summary.py` | 목표 달성 요약 |

### 3.4 Jupyter 노트북

| 노트북 | 용도 | 커널 |
|--------|------|------|
| `notebooks/00_setup_gpu_env.ipynb` | GPU 환경 검증 | Python (motionanalyzer GPU) |
| `notebooks/04_full_pipeline_100k.ipynb` | 100k 전체 파이프라인 | Python (motionanalyzer GPU) |

---

## 4. 즉시 실행 시나리오

### 시나리오 A: 전체 파이프라인 (100k)

```powershell
cd c:\motionanalyzer
.\scripts\setup_gpu_env.ps1   # 최초 1회
.\scripts\run_full_pipeline.ps1
```

**예상 소요**: ~1~2시간 (데이터 생성 45~60분 + 분석 15~30분 + DREAM/PatchCore 5~15분)

### 시나리오 B: QA 게이트 (개발 단위 시작 전)

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python scripts/evaluate_synthetic_dataset_quality.py
python scripts/validate_enhanced_dream.py
```

### 시나리오 C: Phase B 벤치마크

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python scripts/benchmark_phase_b_comprehensive.py
```

**출력**: `reports/phase_b_benchmark_results.json`

### 시나리오 D: GUI 실행 (ML 포함)

```powershell
.\.venv-gpu\Scripts\Activate.ps1
motionanalyzer gui
# 또는
.\.venv-gpu\Scripts\python.exe -m motionanalyzer.cli gui
```

### 시나리오 E: Jupyter 100k 파이프라인

1. Cursor에서 `notebooks/04_full_pipeline_100k.ipynb` 열기
2. 커널: **Python (motionanalyzer GPU)**
3. Run All

---

## 5. 디렉터리 구조 요약

```
c:\motionanalyzer\
├── docs/                    # 문서 (INDEX.md → PROJECT_GOALS → ROADMAP)
├── src/motionanalyzer/      # 소스 코드
├── scripts/                 # PowerShell, Python, Shell 스크립트
├── notebooks/               # Jupyter (00_setup, 04_full_pipeline)
├── data/synthetic/          # 합성 데이터 (DS-260220-ml-100k-100k-60f, DS-260223-ml-fp-20k-60f)
├── reports/                 # 분석 결과, 리포트
│   ├── crack_detection_analysis/
│   ├── deliverables/
│   └── phase_b_benchmark_results.json
├── .venv-gpu/               # GPU/ML 가상환경 (setup_gpu_env.ps1)
├── .venv/                   # 기본 개발 환경 (bootstrap.ps1)
├── pyproject.toml
├── requirements-gpu.txt
└── AGENTS.md                # AI 에이전트 계약
```

---

## 6. 핵심 문서 읽기 순서

1. `docs/INDEX.md`
2. `docs/PROJECT_GOALS.md`
3. `docs/DEVELOPMENT_ROADMAP_FINAL.md`
4. `docs/PHASE_B_INSIGHTS.md`
5. `docs/PIPELINE_SETUP_COMPLETE.md`
6. `reports/INDEX.md`

---

## 7. 의존성 요약

| 항목 | 파일 |
|------|------|
| 기본 | `pyproject.toml` (dependencies) |
| ML | `pyproject.toml` [ml] + `requirements-gpu.txt` |
| GPU | PyTorch CUDA 12.1 (`--index-url https://download.pytorch.org/whl/cu121`) |

---

## 8. Skills 및 캐싱

| Skill | 용도 |
|-------|------|
| `ai-coding-accelerator` | 코딩 (shell, compaction) |
| `agent-performance` | 검증, 캐싱, 컴팩션 |

**캐시 과다 시**: `scripts/cursor-speed-optimization/RUN_ALL_OPTIMIZATIONS.ps1` (Cursor 종료 후 실행)

## 9. 알려진 이슈 및 인사이트

- **Temporal 모델**: 벤치마크 ROC AUC 0.25 — 시계열 분할/정렬 개선 필요
- **DREAM+Advanced**: ROC 1.0 — 합성 데이터 과적합 가능성, 실제 데이터 검증 필요
- **앙상블**: 단일 모델 대비 큰 향상 없음 — 다양성 확보 필요
- **라벨 누설 방지**: ML 입력에 `crack_risk_*` 사용 금지

---

## 10. 검증 (객관적 지표)

```powershell
.\scripts\verify_agent_handoff.ps1 -OutputJson reports/agent_verification.json
# Quick 모드 (ML/GPU 검증 생략): -Quick
```

**목표**: 점수 80/100 이상. **검증 결과**: 100/100 (2026-02-25)  
**상세**: `docs/AGENT_VERIFICATION_AND_CACHING.md`

**성능평가 (상위 4%)**: `python scripts/evaluate_agent_setup.py -o reports/agent_setup_evaluation.json`  
**반복 개선**: `docs/AGENT_REFINEMENT_LOOP.md`

**시나리오 준비도**: `python scripts/evaluate_scenario_readiness.py -o reports/scenario_readiness.json`  
**시나리오별 원클릭**: `run_scenario_S01.ps1` ~ `run_scenario_S15.ps1` | `docs/ROADMAP_SCENARIOS_AND_READINESS.md`

## 11. 체크리스트 (새 에이전트 시작 시)

- [ ] `.\scripts\setup_gpu_env.ps1` 실행
- [ ] `.\scripts\verify_agent_handoff.ps1` 실행 (점수 80 이상 확인)
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 확인
- [ ] `docs/DEVELOPMENT_ROADMAP_FINAL.md` 최신 상태 확인
- [ ] 다음 작업: Temporal 개선 또는 CPD 정확도 (로드맵 "다음 단계" 참조)
