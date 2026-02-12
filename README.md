# motionanalyzer

Time-series motion analyzer for physics-based pattern analysis.

## 프로젝트 목표

비디오 프레임에서 추출한 점좌표 시계열을 기반으로, 이동경로/속도/가속도 벡터를 계산하고 비교 가능한 형태로 시각화하는 분석 도구를 구축합니다.

## 입력/출력 명세

### Input 1

하나의 비디오에서 추출된 프레임별 점좌표 텍스트 파일 묶음.
파일명 순서가 프레임 순서를 의미합니다.

각 txt 파일 포맷:

```txt
# x,y,index
1097,1087,1
1096,1086,2
1095,1086,3
1094,1086,4
...
```

- txt 파일당 약 200~300개 라인
- 동일한 `index`는 동일 포인트 트래킹 대상

### Input 2

`fps` (초당 프레임 수). 속도/가속도 벡터 계산 시 사용.

### Output 1

같은 `index` 기준으로 프레임별:
- 위치 벡터
- 속도 벡터
- 가속도 벡터

를 표준화된 txt 또는 csv로 내보낸 결과물.

### Output 2

Output 1 결과를 시각화/비교 분석하는 GUI.
- 패턴 해석 보조
- 여러 비디오 결과 동시 비교
- 차이점/특이점 탐색
- 이미지 추출 기능

## 개발환경 (사전 셋업 완료)

현재 저장소에 아래 사전 준비를 반영했습니다.

- Python 패키지 구조: `src/motionanalyzer`
- 개발 도구: `ruff`, `mypy`, `pytest`, `pre-commit`
- Cursor 규칙: `.cursor/rules/motionanalyzer.mdc`
- MCP 설정 예시: `.cursor/mcp.json`
- 부트스트랩 스크립트: `scripts/bootstrap.ps1`
- 표준 디렉토리 생성 CLI: `motionanalyzer init-dirs`

## 빠른 시작 (Windows PowerShell)

### 1) 자동 부트스트랩

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

### 2) 수동 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
motionanalyzer init-dirs
```

## 개발용 명령어

```powershell
pytest
ruff check .
ruff format .
mypy src
motionanalyzer doctor
motionanalyzer gen-synthetic
motionanalyzer analyze-bundle
motionanalyzer compare-runs
motionanalyzer gui
motionanalyzer run-synthetic-suite
motionanalyzer internal-realdata-run
```

## 권장 디렉토리 구조

```txt
mpj/
  data/
    raw/          # 원본 프레임 txt 묶음
    processed/    # 전처리 결과
  exports/
    vectors/      # 벡터 계산 결과
    plots/        # 그래프/이미지 출력
  logs/
  scripts/
  src/motionanalyzer/
  tests/
```

## 보안망 테스트 전략

실제 데이터 외부 반출 금지 정책을 전제로 아래 2단계 검증을 사용합니다.

1. 외부 개발환경: 합성 데이터로 기능 검증
2. 사내 보안환경: exe로 실제 데이터 평가

상세 절차는 `docs/SECURE_TEST_POLICY.md`를 따릅니다.

### 합성 데이터 테스트 (외부)

```powershell
motionanalyzer gen-synthetic --output-dir data/synthetic/session_001 --frames 120 --points-per-frame 220 --fps 30
pytest
```

### Windows exe 빌드 (외부/GitHub Actions 또는 로컬)

```powershell
.\scripts\build_exe.ps1
```

생성 파일:

- `dist/motionanalyzer-cli.exe` (CLI)
- `dist/motionanalyzer-gui.exe` (오프라인 로컬 GUI 런처)

GitHub에서는 `.github/workflows/build-windows-exe.yml`가 위 두 exe를 아티팩트로 생성합니다.

## FPCB 벤딩 합성데이터 고도화

FPCB side-view 벤딩(직선 -> 호 -> U형)을 반영한 물리 모델 기반 생성기를 제공합니다.

- 모델 문서: `docs/SYNTHETIC_MODEL_FPCB.md`
- 시나리오: `normal`, `crack`, `pre_damage`, `thick_panel`, `uv_overcured`
- 생성 명령:

```powershell
motionanalyzer gen-synthetic --scenario normal --output-dir data/synthetic/normal_case
motionanalyzer gen-synthetic --scenario crack --output-dir data/synthetic/crack_case
```

- 선검증 명령:

```powershell
motionanalyzer validate-synthetic --input-dir data/synthetic/normal_case --scenario normal
motionanalyzer validate-synthetic --input-dir data/synthetic/crack_case --scenario crack
```

## 물리/화학 지식베이스와 AI 활용

개발 중 고급 도메인 지식을 반복 활용하기 위해 아래 문서를 기준으로 사용합니다.

- 지식베이스: `docs/KNOWLEDGE_FPCB_PHYSICS_CHEMISTRY.md`
- AI 구현 플레이북: `docs/AI_MODELING_PLAYBOOK.md`
- Cursor 규칙:
  - `.cursor/rules/fpcb-domain-knowledge.mdc`
  - `.cursor/rules/fpcb-modeling-checklist.mdc`

## 사내 실데이터 분석 전 셋업

사내 실데이터 투입 전 순차 준비 절차:

- 문서: `docs/INTERNAL_REALDATA_SETUP.md`
- 설정 템플릿: `configs/internal_eval.template.json`
- 자동 스크립트:

```powershell
.\scripts\prepare_internal_eval.ps1 -InputDir "data/raw/session_real_001"
```

## GUI 및 분석 기능

- GUI 실행:

```powershell
.\scripts\run_gui.ps1
# 또는 빌드된 GUI exe 실행
.\dist\motionanalyzer-gui.exe --host 127.0.0.1 --port 8501
```

GUI 기본 설계 기준:
- 해상도 프로파일: 1920x1080 (Full HD)
- 화면 폭: 최대 1920px 고정 레이아웃
- 오프라인 로컬 실행: Streamlit 서버를 로컬에서만 기동

- CLI 분석:

```powershell
motionanalyzer analyze-bundle --input-dir data/synthetic/normal_case --output-dir exports/vectors/normal_case
motionanalyzer compare-runs --base-summary exports/vectors/normal_case/summary.json --candidate-summary exports/vectors/crack_case/summary.json
```

## 사내망 실환경 테스트 시나리오

- 시나리오 문서: `docs/INTERNAL_REALDATA_TEST_SCENARIOS.md`
- 통합 실행 런북: `docs/PHASE2_EXECUTION_RUNBOOK.md`
- 합성데이터 내부 기능 테스트 스크립트:

```powershell
.\scripts\run_internal_synthetic_test.ps1 -Scenario normal -SessionName internal_sim_001
motionanalyzer run-synthetic-suite --output-root reports/synthetic_suite
```

## EXE 실데이터 전체 실행 (TXT 로그)

사내망에서 exe 단독 실행으로 preflight + 분석 + 비교(+선택)를 수행하고
단일 txt 로그를 남길 수 있습니다.

```powershell
.\scripts\run_internal_realdata.ps1 `
  -ExePath ".\dist\motionanalyzer-cli.exe" `
  -InputDir "data/raw/session_real_001" `
  -OutputDir "exports/vectors/real_session_001" `
  -BaselineSummary "exports/vectors/baseline/summary.json" `
  -RunLogTxt "internal_eval/logs/internal_run_latest.txt"
```

생성 핵심 결과:
- `internal_eval/logs/internal_run_latest.txt`
- `reports/preflight/internal_preflight_latest.json`
- `exports/vectors/real_session_001/vectors.txt`
- `exports/vectors/real_session_001/summary.txt`
