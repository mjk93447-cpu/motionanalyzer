# MotionAnalyzer 사용자 가이드

**버전**: 0.2.0  
**최종 업데이트**: 2026년 2월 19일

### 프로젝트 목표 (우선순위)

1. **벤딩 중 크랙 감지** (최우선): 과경화·과도한 궤적·빠른 속도로 인한 크랙 — 시계열·국소적 감지 (속도 변화, 충격파, 진동, 길이 변화)
2. **이미 크랙된 패널 감지**: 손상된 FPCB의 미묘한 물성·구조 차이 — 전체적 패턴, AI 모델로 정교하게 구분

상세: [docs/PROJECT_GOALS.md](PROJECT_GOALS.md)

---

## 목차

1. [시작하기](#시작하기)
2. [GUI 사용법](#gui-사용법)
3. [CLI 사용법](#cli-사용법)
4. [합성 데이터 생성](#합성-데이터-생성)
5. [데이터 분석](#데이터-분석)
6. [ML 모델 사용](#ml-모델-사용)
7. [문제 해결](#문제-해결)

---

## 시작하기

### 시스템 요구사항

- **OS**: Windows 10/11
- **Python**: 3.11 이상 (개발용)
- **메모리**: 최소 4GB RAM (ML 기능 사용 시 8GB 권장)
- **디스크**: 최소 500MB 여유 공간 (ML 포함 EXE 사용 시 2GB)

### 설치 방법

#### 방법 1: EXE 파일 사용 (권장)

1. GitHub Releases에서 EXE 파일 다운로드
   - `motionanalyzer-gui.exe`: 경량 버전 (ML 기능 없음)
   - `motionanalyzer-gui-ml.exe`: ML 포함 버전 (DREAM/PatchCore 사용 가능)

2. EXE 파일을 원하는 위치에 저장

3. 더블클릭하여 실행

#### 방법 2: Python 패키지 설치 (개발용)

```powershell
# 1. 저장소 클론
git clone <repository-url>
cd motionanalyzer

# 2. 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 패키지 설치
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 4. 표준 디렉토리 생성
motionanalyzer init-dirs
```

---

## GUI 사용법

### GUI 실행

**EXE 사용**:
```powershell
.\motionanalyzer-gui.exe
# 또는 ML 포함 버전
.\motionanalyzer-gui-ml.exe
```

**Python 패키지 사용**:
```powershell
motionanalyzer gui
# 또는
.\scripts\run_gui.ps1
```

### 주요 탭

#### 1. Analyze 탭

**기능**: 데이터셋 분석 및 벡터 계산

**사용 방법**:
1. "Input directory"에서 분석할 데이터셋 폴더 선택
2. "Output directory"에서 결과 저장 위치 지정
3. "Analysis mode" 선택:
   - **Physics**: 물리 기반 분석 (기본)
   - **DREAM**: DREAM 모델 사용 (ML 포함 버전만)
   - **PatchCore**: PatchCore 모델 사용 (ML 포함 버전만)
   - **Ensemble**: 앙상블 모델 사용 (ML 포함 버전만)
   - **Temporal**: Temporal 모델 사용 (ML 포함 버전만)
4. "Run Analysis" 버튼 클릭

**결과 파일**:
- `vectors.csv`: 프레임별 벡터 데이터
- `vectors.txt`: 텍스트 형식 벡터 데이터
- `summary.json`: 요약 통계
- `vector_map.png`: 벡터 시각화

#### 2. Compare 탭

**기능**: 두 분석 결과 비교

**사용 방법**:
1. "Base summary"에서 기준 데이터셋의 `summary.json` 선택
2. "Candidate summary"에서 비교할 데이터셋의 `summary.json` 선택
3. "Compare" 버튼 클릭

**결과**: 두 데이터셋 간의 차이점 표시

#### 3. Crack Model Tuning 탭

**기능**: 크랙 탐지 모델 파라미터 조정

**사용 방법**:
1. 파라미터 값 입력
2. "Save Parameters" 버튼으로 저장
3. Analyze 탭에서 Physics 모드로 분석 시 저장된 파라미터 사용

#### 4. ML & Optimization 탭

**기능**: ML 모델 학습 및 최적화

**사용 방법**:
1. "Normal datasets"에서 정상 데이터셋 폴더들 선택
2. "Crack datasets"에서 크랙 데이터셋 폴더들 선택
3. "Model type" 선택 및 옵션 설정
4. "Train Model" 버튼 클릭

**결과**: 학습된 모델이 `%APPDATA%/motionanalyzer/models/`에 저장됨

#### 5. Time Series Analysis 탭

**기능**: Change Point Detection (변화점 탐지)

**사용 방법**:
1. "Dataset path"에서 분석할 데이터셋 선택
2. "Detection Method" 선택 (CUSUM/Window-based/PELT)
3. "Time Series Feature" 선택
4. 옵션 설정 (다중 특징, 자동 튜닝, 앙상블)
5. "Detect Change Points" 버튼 클릭

**결과**: 변화점 프레임 번호 및 시각화

### Help 메뉴

GUI 상단의 **Help** 메뉴에서:
- **Quick Start Guide**: 빠른 시작 가이드
- **User Guide**: 사용자 가이드 문서 열기
- **About**: 프로그램 정보

---

## CLI 사용법

### 기본 명령어

```powershell
# 도움말
motionanalyzer --help

# 환경 점검
motionanalyzer doctor

# 합성 데이터 생성
motionanalyzer gen-synthetic --scenario normal --output-dir data/synthetic/normal_case

# 데이터 분석
motionanalyzer analyze-bundle --input-dir data/synthetic/normal_case --output-dir exports/vectors/normal_case

# 결과 비교
motionanalyzer compare-runs --base-summary exports/vectors/normal_case/summary.json --candidate-summary exports/vectors/crack_case/summary.json
```

### 예제 데이터셋 사용

```powershell
# 예제 데이터셋 생성
python scripts/generate_example_datasets.py

# 예제 데이터 분석
motionanalyzer analyze-bundle `
  --input-dir data/synthetic/examples/normal `
  --output-dir exports/vectors/example_normal
```

---

## 합성 데이터 생성

### 시나리오 종류

1. **normal**: 정상 공정
2. **crack**: 크랙 발생
3. **pre_damage**: 사전 손상
4. **thick_panel**: 두꺼운 패널
5. **uv_overcured**: UV 과경화

### 생성 예제

```powershell
# 정상 케이스
motionanalyzer gen-synthetic `
  --scenario normal `
  --output-dir data/synthetic/normal_case `
  --frames 120 `
  --points-per-frame 230 `
  --fps 30

# 크랙 케이스
motionanalyzer gen-synthetic `
  --scenario crack `
  --output-dir data/synthetic/crack_case `
  --frames 120 `
  --points-per-frame 230 `
  --fps 30
```

---

## 데이터 분석

### 입력 데이터 형식

데이터셋 폴더 구조:
```
dataset/
  frame_00000.txt
  frame_00001.txt
  frame_00002.txt
  ...
  fps.txt
```

각 `frame_*.txt` 파일 형식:
```
# x,y,index
1097,1087,1
1096,1086,2
1095,1086,3
...
```

`fps.txt` 파일:
```
30.0
```

### 출력 파일

- **vectors.csv**: 프레임별 벡터 데이터 (CSV 형식)
- **vectors.txt**: 프레임별 벡터 데이터 (텍스트 형식)
- **summary.json**: 요약 통계 (JSON 형식)
- **summary.txt**: 요약 통계 (텍스트 형식)
- **vector_map.png**: 벡터 시각화 이미지

---

## ML 모델 사용

### 모델 학습

**GUI 사용**:
1. ML & Optimization 탭에서 데이터셋 선택
2. 모델 타입 선택
3. "Train Model" 클릭

**모델 저장 위치**: `%APPDATA%/motionanalyzer/models/`

### 모델 사용

학습된 모델은 Analyze 탭에서 "Analysis mode"로 선택하여 사용할 수 있습니다.

---

## 문제 해결

### EXE 실행 오류

**문제**: EXE 파일이 실행되지 않음

**해결**:
1. Windows Defender 또는 안티바이러스에서 예외 추가
2. 관리자 권한으로 실행 시도
3. Python 패키지 버전으로 대체 사용

### ML 모델 로딩 실패

**문제**: "ML dependencies not installed" 오류

**해결**:
1. ML 포함 EXE (`motionanalyzer-gui-ml.exe`) 사용
2. 또는 Python 패키지에서 ML 의존성 설치:
   ```powershell
   pip install -e ".[ml]"
   ```

### 메모리 부족 오류

**문제**: 대용량 데이터셋 분석 시 메모리 부족

**해결**:
1. 데이터셋을 작은 단위로 분할
2. 프레임 수 줄이기
3. 포인트 수 줄이기

---

## 추가 리소스

- **개발 로드맵**: `docs/DEVELOPMENT_ROADMAP_FINAL.md`
- **Phase B 인사이트**: `docs/PHASE_B_INSIGHTS.md`
- **합성 데이터 모델**: `docs/SYNTHETIC_MODEL_FPCB.md`
- **Change Point Detection**: `docs/CHANGEPOINT_DETECTION.md`

---

## 피드백 및 지원

문제가 발생하거나 개선 사항이 있으면 GitHub Issues에 등록해주세요.
