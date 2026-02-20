# GPU · Windows exe 활용 — TODO 리스트

**기준**: [docs/GPU_REMOTE_STRATEGY.md](GPU_REMOTE_STRATEGY.md)  
**환경**: 오피스 PC (RTX 2070 Super), 보안 네트워크(Cursor 불가) → **Windows exe 내 모든 ML 수행**

---

## Phase A: GPU device 자동 감지 (코드 수정)

| ID | 작업 | 파일/위치 | 상태 |
|----|------|-----------|------|
| A.1 | DREAM device 자동 감지 | `src/motionanalyzer/ml_models/dream.py` | ⬜ |
| A.2 | runners.py DREAM에 device 전달 | `src/motionanalyzer/gui/runners.py` | ⬜ |
| A.3 | evaluate_goal1_ml.py `--device` 옵션 | `scripts/evaluate_goal1_ml.py` | ⬜ |
| A.4 | evaluate_goal2_ml.py `--device` 옵션 | `scripts/evaluate_goal2_ml.py` | ⬜ |

### A.1 상세

- `DREAMPyTorch.__init__`: `device` 기본값을 `"cuda" if torch.cuda.is_available() else "cpu"`로 변경
- exe 실행 시 RTX 2070 Super에서 자동으로 CUDA 사용

---

## Phase B: GUI에 exe 내 ML 기능 추가

| ID | 작업 | 설명 | 상태 |
|----|------|------|------|
| B.1 | **합성 데이터 생성** 탭/메뉴 | `generate_ml_dataset.py` 로직을 GUI에서 실행 | ⬜ |
| B.2 | **목표 평가** 탭/메뉴 | `evaluate_goal1_ml.py`, `evaluate_goal2_ml.py` 로직을 GUI에서 실행 | ⬜ |
| B.3 | 목표 달성 요약 표시 | `evaluate_goals_summary.py` 결과를 GUI에 표시 | ⬜ |

### B.1 상세

- `desktop_gui.py`에 "합성 데이터" 또는 "ML 데이터셋" 메뉴/탭 추가
- 출력 경로: `data/synthetic/ml_dataset/` (기본값)
- 옵션: 전체(1100) / 소규모(115) 선택

### B.2 상세

- "목표 1 평가 (DREAM/PatchCore)", "목표 2 평가" 버튼
- 실행 후 `reports/goal1_ml_evaluation.json`, `goal2_ml_evaluation.json` 생성
- 결과(Precision, Recall, F1, PR AUC)를 GUI에 표시

---

## Phase C: PyInstaller exe 빌드 (PyTorch+CUDA)

| ID | 작업 | 설명 | 상태 |
|----|------|------|------|
| C.1 | PyInstaller spec 작성 | `--hidden-import` torch, CUDA 관련 모듈 | ⬜ |
| C.2 | PyTorch CUDA 빌드 | `cu118` 또는 `cu121` 휠 사용 | ⬜ |
| C.3 | exe 테스트 (RTX 2070 Super PC) | `torch.cuda.is_available()` 확인용 | ⬜ |

### C.1 상세

- `motionanalyzer_gui.spec` 또는 `pyinstaller` 명령에 `--hidden-import=torch` 등 포함
- 대상 PC: NVIDIA 드라이버 설치 (CUDA runtime 포함)

---

## Phase D: 선택적 개선 (VM/병렬화)

| ID | 작업 | 설명 | 상태 |
|----|------|------|------|
| D.1 | generate_ml_dataset.py `--workers N` | CPU 병렬화 (exe 내에서도 가능) | ⬜ |
| D.2 | prepare_training_data 병렬 옵션 | 특징 추출 가속 | ⬜ |

---

## Phase E: 검증

| ID | 작업 | 검증 방법 | 상태 |
|----|------|-----------|------|
| E.1 | RTX 2070 Super에서 DREAM GPU 동작 | exe 실행, ML & Optimization에서 Train, nvidia-smi 확인 | ⬜ |
| E.2 | exe 내 합성 데이터 생성 | GUI에서 합성 데이터 생성, manifest 확인 | ⬜ |
| E.3 | exe 내 목표 평가 | GUI에서 목표 평가 실행, reports/ 확인 | ⬜ |

---

## 우선순위

1. **A.1, A.2** — exe 내 DREAM GPU 사용 (즉시 효과)
2. **B.1, B.2, B.3** — exe 내 전체 ML 파이프라인
3. **C.1, C.2, C.3** — exe 빌드·배포
4. **D.1** — 합성 데이터 병렬화 (선택)
