# EXE 로컬 테스트 가이드

로컬에서 motionanalyzer EXE를 테스트하기 위한 세팅과 순서별 가이드입니다.

---

## 사전 요구사항

- **Windows 10/11**
- **Python 3.11+** (가상환경 권장)
- **PowerShell** (관리자 권한 불필요)

---

## 1단계: 프로젝트 준비

### 1.1 저장소 클론 및 가상환경

```powershell
# 프로젝트 디렉터리로 이동
cd C:\path\to\motionanalyzer

# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 패키지 설치
python -m pip install --upgrade pip
python -m pip install -e ".[dev,build]"
```

### 1.2 원클릭 준비 스크립트 (권장)

```powershell
.\scripts\prepare_exe_test.ps1
```

다음을 수행합니다:

- 합성 데이터셋 생성 (`data\synthetic\examples\`)
- GUI EXE 빌드 (`dist\motionanalyzer-gui.exe`)
- CLI EXE 빌드 (`dist\motionanalyzer-cli.exe`)
- 출력 디렉터리 생성 (`exports\`, `reports\`)

**옵션:**

```powershell
# 빌드 생략 (데이터만 준비)
.\scripts\prepare_exe_test.ps1 -SkipBuild

# 데이터 생성 생략 (기존 데이터 사용)
.\scripts\prepare_exe_test.ps1 -SkipData
```

---

## 2단계: 수동 준비 (스크립트 없이)

### 2.1 합성 데이터셋 생성

```powershell
python scripts\generate_example_datasets.py
```

생성 경로: `data\synthetic\examples\`

| 시나리오       | 설명       |
|----------------|------------|
| `normal`       | 정상 공정  |
| `crack`        | 크랙 발생  |
| `pre_damage`   | 사전 손상  |
| `thick_panel`  | 두꺼운 패널 |
| `uv_overcured` | UV 과경화  |

### 2.2 GUI EXE 빌드

```powershell
.\scripts\build_exe.ps1
```

출력: `dist\motionanalyzer-gui.exe`

ML 버전:

```powershell
.\scripts\build_exe.ps1 -IncludeML
```

출력: `dist\motionanalyzer-gui-ml.exe`

### 2.3 CLI EXE 빌드

```powershell
.\scripts\build_cli_exe.ps1
```

출력: `dist\motionanalyzer-cli.exe`

---

## 3단계: CLI EXE 테스트

### 3.1 CLI 배치 테스트 (자동)

```powershell
.\scripts\run_exe_synthetic_analysis.ps1
```

실행 내용:

1. normal 합성 데이터 분석
2. crack 합성 데이터 분석
3. normal vs crack 비교
4. FPCB 파이프라인 (generate → analyze → plot)

### 3.2 CLI 개별 명령 테스트

```powershell
# 1. 단일 분석
.\dist\motionanalyzer-cli.exe analyze-bundle `
    --input-dir "data\synthetic\examples\normal" `
    --output-dir "exports\vectors\exe_normal"

# 2. 결과 비교
.\dist\motionanalyzer-cli.exe compare-runs `
    --base-summary "exports\vectors\exe_normal\summary.json" `
    --candidate-summary "exports\vectors\exe_crack\summary.json"

# 3. FPCB 파이프라인
.\dist\motionanalyzer-cli.exe run-fpcb-pipeline `
    --data-dir "data\synthetic\fpcb_high_fidelity" `
    --export-vectors-dir "exports\vectors\fpcb_high_fidelity" `
    --plots-dir "exports\plots"
```

### 3.3 CLI 결과 확인

| 경로 | 파일 |
|------|------|
| `exports\vectors\exe_normal\` | vectors.csv, summary.json, vector_map.png |
| `exports\vectors\exe_crack\` | vectors.csv, summary.json, vector_map.png |
| `exports\plots\` | fpcb_metrics.png |
| `reports\compare\` | latest_compare.txt |

---

## 4단계: GUI EXE 테스트

### 4.1 GUI 실행

```powershell
.\dist\motionanalyzer-gui.exe
```

또는 `dist\motionanalyzer-gui.exe`를 더블클릭하여 실행합니다.

### 4.2 Analyze 탭 테스트

1. **Input Dir**: `data\synthetic\examples\normal` (또는 Browse로 선택)
2. **Output Dir**: `exports\vectors\gui_test`
3. **FPS**: `30`
4. **Run Analysis** 클릭
5. 벡터맵, 요약이 정상 출력되는지 확인

### 4.3 Compare 탭 테스트

1. **Base Summary**: `exports\vectors\exe_normal\summary.json`
2. **Candidate Summary**: `exports\vectors\exe_crack\summary.json`
3. **Compare** 클릭
4. delta 값이 표시되는지 확인

---

## 5단계: EXE 검증 스크립트

```powershell
.\scripts\test_build_exe.ps1
```

- EXE 존재 여부 확인
- (선택) 실행 테스트 (GUI 창이 열림)

실행 테스트 생략:

```powershell
.\scripts\test_build_exe.ps1 -SkipExecutionTest
```

---

## 트러블슈팅

### 한글 경로 문제

프로젝트 경로에 한글이 있으면 `Set-Location` 오류가 날 수 있습니다.

- **해결 1**: 프로젝트를 영문 경로로 복사 (예: `C:\projects\motionanalyzer`)
- **해결 2**: 프로젝트 루트에서 직접 스크립트 실행

### EXE가 없음

```powershell
# CLI EXE만 빌드
.\scripts\build_cli_exe.ps1

# GUI EXE만 빌드
.\scripts\build_exe.ps1
```

### 합성 데이터 없음

```powershell
python scripts\generate_example_datasets.py
```

### Python을 찾을 수 없음

```powershell
# 시스템 Python 사용
.\scripts\prepare_exe_test.ps1 -PythonExe "python"

# 또는 가상환경 활성화 후
.\.venv\Scripts\Activate.ps1
.\scripts\prepare_exe_test.ps1
```

---

## 요약 체크리스트

- [ ] 가상환경 생성 및 패키지 설치
- [ ] `.\scripts\prepare_exe_test.ps1` 실행
- [ ] `.\scripts\run_exe_synthetic_analysis.ps1`로 CLI 테스트
- [ ] `.\dist\motionanalyzer-gui.exe`로 GUI 테스트
- [ ] `exports\` 아래 결과 파일 확인
