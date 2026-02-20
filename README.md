# motionanalyzer

**Physics-based time-series motion analyzer for FPCB bending analysis**

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 프로젝트 개요

`motionanalyzer`는 비디오 프레임에서 추출한 점좌표 시계열을 기반으로, 이동경로/속도/가속도 벡터를 계산하고 ML 기반 이상 감지를 수행하는 오프라인 Windows GUI 애플리케이션입니다.

### 🎯 최종 목표 (상세: [docs/PROJECT_GOALS.md](docs/PROJECT_GOALS.md))

| 우선순위 | 목표 | 특성 | 핵심 감지 대상 |
|----------|------|------|----------------|
| **1 (최우선)** | **벤딩 중 크랙 감지** | 시계열·국소적 | 크랙 직전/직후 속도 변화, 충격파, 진동, 미세 길이 변화(벌어짐) |
| **2** | **이미 크랙된 패널 감지** | 전체적 패턴 | 손상으로 인한 미묘한 물성·구조 차이, 다른 벤딩 궤적 |

**목표 1 원인**: 과경화, 과도한 벤딩 궤적, 너무 빠른 벤딩 속도  
**목표 2**: 손상 부위가 매우 작아 차이가 미세 → 다양한 AI 모델로 정교하게 구분

**현재 상태**: ✅ **v0.2.0 배포 준비 완료** (실제 데이터 확보 전 단계 개발 완료)

---

## ✨ 주요 기능

### 1. 벡터 분석 엔진
- **위치/속도/가속도 벡터 계산**: 프레임별 점좌표에서 물리량 계산
- **곡률 분석**: 곡률 집중도 및 변화율 분석
- **시각화**: 벡터 맵, 시계열 플롯, 히스토그램
- **단위 표준화**: 픽셀 단위를 SI 단위로 변환 (px/s, px/s² to m/s,km/s²)

### 2. 합성 데이터 생성기
- **5개 시나리오**: `normal`, `crack`, `pre_damage`, `thick_panel`, `uv_overcured`
- **물리 기반 모델**: FPCB 벤딩 시뮬레이션 (직선 → 호 → U형)
- **고급 패턴**: 충격파(shockwave), 미세 진동(micro-vibration) 포함
- **예제 데이터셋**: 빠른 테스트를 위한 사전 생성 데이터 제공

### 3. ML 기반 이상 감지 모델

#### DREAM (DRAEM 전략)
- **전략**: Deep Reconstruction Error-based Anomaly Model
- **성능**: ROC AUC 0.913, PR AUC 0.953 (baseline features)
- **특징**: 정상 데이터만으로 학습, reconstruction error 기반 이상 점수

#### PatchCore
- **전략**: Memory-bank 기반 이상 감지
- **성능**: ROC AUC 0.908, PR AUC 0.954 (baseline features)
- **특징**: Coreset 기반 효율적인 메모리 사용

#### Ensemble 모델
- **전략**: DREAM + PatchCore 결합
- **방법**: 가중 평균, 최대값, 스태킹
- **현재 성능**: ROC AUC 0.908 (단일 모델 대비 향상 필요)

#### Temporal 모델
- **전략**: LSTM/GRU 기반 시계열 이상 감지
- **특징**: 슬라이딩 윈도우, reconstruction error 기반
- **상태**: 성능 개선 필요 (ROC AUC 0.100, 실제 데이터 재평가 필요)

### 4. Change Point Detection (CPD)
- **알고리즘**: CUSUM, Window-based, PELT
- **고급 기능**:
  - 파라미터 자동 튜닝 (Grid Search, Bayesian Optimization)
  - 다중 특징 결합 (union, intersection, majority)
  - 앙상블 CPD (여러 방법 결합)
- **용도**: 크랙 발생 시점 정확한 탐지

### 5. 고급 특징 엔지니어링
- **통계 특징**: skewness, kurtosis, autocorrelation
- **시간 도메인**: frame-to-frame 변화율
- **주파수 도메인**: FFT 기반 dominant frequency, spectral power, spectral entropy
- **주의**: 합성 데이터에서 과적합 가능성 있음 (실제 데이터 검증 필요)

### 6. GUI 애플리케이션 (Tkinter 기반)
- **5개 탭**: Analyze, Compare, Crack Model Tuning, ML & Optimization, Time Series Analysis
- **Help 메뉴**: Quick Start Guide, User Guide, About
- **오프라인 실행**: 인터넷 연결 불필요
- **EXE 빌드**: 경량 버전 및 ML 포함 버전 제공

---

## 🚀 빠른 시작

### 설치 방법

#### 방법 1: EXE 파일 사용 (권장, 일반 사용자)

1. **GitHub Releases에서 다운로드**:
   - `motionanalyzer-gui.exe`: 경량 버전 (ML 기능 없음, ~50-100MB)
   - `motionanalyzer-gui-ml.exe`: ML 포함 버전 (DREAM/PatchCore 사용 가능, ~200-500MB)

2. **실행**: EXE 파일을 더블클릭하여 실행

#### 방법 2: Python 패키지 설치 (개발자용)

```powershell
# 1. 저장소 클론
git clone https://github.com/mjk93447-cpu/motionanalyzer.git
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

### 기본 사용법

#### GUI 실행

```powershell
# EXE 사용
.\motionanalyzer-gui.exe
# 또는 ML 포함 버전
.\motionanalyzer-gui-ml.exe

# Python 패키지 사용
motionanalyzer gui
```

#### 예제 데이터셋으로 시작하기

```powershell
# 예제 데이터셋 생성 (5개 시나리오)
python scripts/generate_example_datasets.py

# 예제 데이터 분석
motionanalyzer analyze-bundle `
  --input-dir data/synthetic/examples/normal `
  --output-dir exports/vectors/example_normal
```

#### 합성 데이터 생성 및 분석

```powershell
# 합성 데이터 생성
motionanalyzer gen-synthetic `
  --scenario normal `
  --output-dir data/synthetic/normal_case `
  --frames 120 `
  --points-per-frame 230 `
  --fps 30

# 데이터 분석
motionanalyzer analyze-bundle `
  --input-dir data/synthetic/normal_case `
  --output-dir exports/vectors/normal_case

# 결과 비교
motionanalyzer compare-runs `
  --base-summary exports/vectors/normal_case/summary.json `
  --candidate-summary exports/vectors/crack_case/summary.json
```

---

## 📖 상세 사용 가이드

### GUI 주요 탭

#### 1. Analyze 탭
- **기능**: 데이터셋 분석 및 벡터 계산
- **분석 모드**: Physics, DREAM, PatchCore, Ensemble, Temporal
- **출력**: `vectors.csv`, `summary.json`, `vector_map.png`

#### 2. Compare 탭
- **기능**: 두 분석 결과 비교
- **입력**: 두 개의 `summary.json` 파일
- **출력**: 차이점 시각화 및 통계

#### 3. Crack Model Tuning 탭
- **기능**: 크랙 탐지 모델 파라미터 조정
- **저장**: `%APPDATA%/motionanalyzer/configs/`

#### 4. ML & Optimization 탭
- **기능**: ML 모델 학습 및 최적화
- **모델 타입**: DREAM, PatchCore, Ensemble, Temporal
- **저장 위치**: `%APPDATA%/motionanalyzer/models/`

#### 5. Time Series Analysis 탭
- **기능**: Change Point Detection
- **방법**: CUSUM, Window-based, PELT
- **고급 옵션**: 자동 튜닝, 다중 특징, 앙상블

### CLI 명령어

```powershell
# 환경 점검
motionanalyzer doctor

# 합성 데이터 생성
motionanalyzer gen-synthetic --scenario crack --output-dir data/synthetic/crack_case

# 데이터 분석
motionanalyzer analyze-bundle --input-dir <input> --output-dir <output>

# 결과 비교
motionanalyzer compare-runs --base-summary <base> --candidate-summary <candidate>

# GUI 실행
motionanalyzer gui
```

---

## 🛠️ 개발자 가이드

### 개발 환경 설정

```powershell
# 자동 부트스트랩
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1

# 또는 수동 설치
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
motionanalyzer init-dirs
```

### 개발 명령어

```powershell
# 테스트 실행
pytest

# 코드 스타일 검사
ruff check .
ruff format .

# 타입 검사
mypy src

# EXE 빌드 (로컬)
.\scripts\build_exe.ps1              # 경량 버전
.\scripts\build_exe.ps1 -IncludeML  # ML 포함 버전
.\scripts\build_cli_exe.ps1         # CLI 버전 (analyze-bundle 등)

# EXE 로컬 테스트 (합성 데이터 + GUI/CLI)
.\scripts\prepare_exe_test.ps1      # 원클릭 준비 (데이터 + 빌드)
.\scripts\run_exe_synthetic_analysis.ps1  # CLI 배치 테스트
# 상세 가이드: [docs/EXE_LOCAL_TEST_GUIDE.md](docs/EXE_LOCAL_TEST_GUIDE.md)
```

### 프로젝트 구조

```
motionanalyzer/
├── src/motionanalyzer/          # 소스 코드
│   ├── analysis.py              # 벡터 분석 엔진
│   ├── synthetic.py             # 합성 데이터 생성
│   ├── desktop_gui.py           # GUI 애플리케이션
│   ├── cli.py                   # CLI 인터페이스
│   ├── ml_models/               # ML 모델들
│   │   ├── dream.py             # DREAM 모델
│   │   ├── patchcore.py         # PatchCore 모델
│   │   ├── hybrid.py            # Ensemble 모델
│   │   └── dream_temporal.py    # Temporal 모델
│   ├── time_series/             # 시계열 분석
│   │   ├── changepoint.py       # CPD 알고리즘
│   │   └── changepoint_optimizer.py  # CPD 최적화
│   └── gui/                     # GUI 모듈
│       └── runners.py            # 모델 실행기
├── tests/                       # 테스트 코드 (68개 테스트)
├── scripts/                     # 유틸리티 스크립트
│   ├── build_exe.ps1            # EXE 빌드
│   ├── generate_example_datasets.py  # 예제 데이터 생성
│   └── benchmark_phase_b_comprehensive.py  # 벤치마크
├── docs/                        # 문서
│   ├── USER_GUIDE.md            # 사용자 가이드
│   ├── DEVELOPMENT_ROADMAP_FINAL.md  # 개발 로드맵
│   ├── PHASE_B_INSIGHTS.md      # 개발 인사이트
│   └── PROJECT_READINESS_ASSESSMENT.md  # 완성도 평가
└── .github/workflows/           # CI/CD
    └── build-windows-exe.yml     # EXE 빌드 워크플로우
```

---

## 📊 프로젝트 현황

### 개발 진행률: **85%** (v0.2.0)

#### ✅ 완료된 기능 (Phase A & B)

- ✅ **Phase A.1**: EXE 빌드 및 Analyze 탭 확장
- ✅ **Phase A.2**: Change Point Detection GUI 통합
- ✅ **Phase B.1**: 합성 데이터 물리 현상 추가 (충격파, 진동)
- ✅ **Phase B.2**: 앙상블 모델 구현
- ✅ **Phase B.3**: Temporal 모델 구현
- ✅ **Phase B.4**: 고급 특징 엔지니어링
- ✅ **Phase B.5**: Change Point Detection 고도화

#### ⚠️ 개선 필요 사항

- **Temporal 모델**: 성능 개선 필요 (ROC AUC 0.100)
- **앙상블 모델**: 단일 모델 대비 향상 없음 (가중치 최적화 필요)
- **고급 특징**: 합성 데이터에서 과적합 가능성 (실제 데이터 검증 필요)

### 모델 성능 벤치마크

| 모델 | ROC AUC | PR AUC | 상태 |
|------|---------|--------|------|
| DREAM (baseline) | 0.913 | 0.953 | ✅ 양호 |
| PatchCore (baseline) | 0.908 | 0.954 | ✅ 양호 |
| Ensemble | 0.908 | 0.954 | ⚠️ 개선 필요 |
| Temporal | 0.100 | 0.286 | ❌ 개선 필요 |
| DREAM+Advanced | 1.000 | 1.000 | ⚠️ 과적합 의심 |

**참고**: 모든 벤치마크는 합성 데이터 기반입니다. 실제 데이터에서의 성능 검증이 필요합니다.

---

## 🗺️ 개발 로드맵 및 다음 단계

### 즉시 시작 가능한 작업

1. **실제 데이터 확보 및 검증**
   - 실제 크랙 데이터 수집
   - 합성 데이터와 실제 데이터 성능 비교
   - 모델 재학습 및 튜닝

2. **Temporal 모델 개선**
   - 데이터셋 분할 방식 개선
   - 시계열 구조 보존 강화
   - 하이퍼파라미터 튜닝

3. **앙상블 다양성 확보**
   - 다양한 모델 추가 고려
   - 가중치 최적화 알고리즘 개선
   - 메타 학습 기법 적용

### 중장기 개발 계획

자세한 내용은 [`docs/DEVELOPMENT_ROADMAP_FINAL.md`](docs/DEVELOPMENT_ROADMAP_FINAL.md) 참조.

---

## ⚠️ 중요 주의사항 및 인사이트

### 1. 라벨 누설 방지

**문제**: Physics 모델의 `crack_risk_*` 특징을 ML 모델 입력에 사용하면 성능이 과대평가됨

**해결책**:
- ML 모델 학습/추론 시 `crack_risk_*` 특징 제외
- 정규화 시 `fit_df` 파라미터로 정상 데이터만 사용하여 통계 계산
- 시계열 데이터 분할 시 데이터셋 레벨에서 분할 (프레임 레벨 분할 금지)

### 2. 합성 데이터 한계

**과적합 위험**:
- 고급 특징이 합성 데이터 패턴에 과적합 가능성 높음
- DREAM+Advanced가 ROC AUC 1.000 (합성 데이터에서만)
- 실제 데이터에서 성능 저하 가능

**대응 방안**:
- 실제 데이터 확보 후 즉시 재검증
- 특징 선택(feature selection)을 통한 차원 축소
- 다양한 시드와 시나리오로 합성 데이터 다양성 확보

### 3. 평가 메트릭 선택

**권장 메트릭**:
- **ROC AUC**: anomaly rate 변화에 비교적 안정적 (상대 비교용)
- **PR AUC (AUCPR)**: precision-recall 균형 반영 (실제 운영 의사결정에 가까움)

**참고**: ADBench(Han et al.), time-series anomaly metric survey("Metric Maze", 2023)

### 4. 데이터 분할 전략

**시계열 데이터**:
- 데이터셋 레벨에서 분할 (train/test)
- 프레임 레벨 분할 금지 (데이터 누설)
- Temporal 모델은 슬라이딩 윈도우 구성 전에 분할

**일반 ML 모델**:
- 정상 데이터로만 정규화 통계 계산 (`fit_df` 파라미터)
- 크랙 데이터는 정규화만 적용 (통계 계산 제외)

### 5. CI/CD 환경 고려사항

- Windows 환경에서 `DISPLAY` 환경 변수 없음 → `platform.system()`으로 OS 감지
- `webbrowser` 모듈은 CI 환경에서 실패 가능 → graceful error handling 필요
- 테스트는 모든 모드(ensemble, temporal 포함)를 검증해야 함

자세한 인사이트는 [`docs/PHASE_B_INSIGHTS.md`](docs/PHASE_B_INSIGHTS.md) 참조.

---

## 📚 문서

### 사용자 문서
- **[사용자 가이드](docs/USER_GUIDE.md)**: GUI/CLI 사용법, 예제 시나리오
- **[CHANGELOG](CHANGELOG.md)**: 버전별 변경 이력
- **[버전 관리 정책](docs/VERSION_POLICY.md)**: 시맨틱 버저닝 정책

### 개발자 문서
- **[개발 로드맵](docs/DEVELOPMENT_ROADMAP_FINAL.md)**: 중장기 개발 전략 및 진행 상황
- **[Phase B 인사이트](docs/PHASE_B_INSIGHTS.md)**: 개발 과정에서 얻은 핵심 인사이트
- **[프로젝트 완성도 평가](docs/PROJECT_READINESS_ASSESSMENT.md)**: 배포 준비 상태 평가
- **[Change Point Detection 가이드](docs/CHANGEPOINT_DETECTION.md)**: CPD 알고리즘 상세 설명

### 기술 문서
- **[FPCB 합성 데이터 모델](docs/SYNTHETIC_MODEL_FPCB.md)**: 물리 기반 시뮬레이션 모델
- **[GUI 아키텍처](docs/GUI_ARCHITECTURE.md)**: GUI 설계 및 구조
- **[크랙 탐지 로드맵](docs/CRACK_DETECTION_ROADMAP.md)**: 크랙 탐지 시스템 개발 계획

---

## 🔧 빌드 및 배포

### EXE 빌드

#### 로컬 빌드

```powershell
# 경량 버전 (ML 기능 없음)
.\scripts\build_exe.ps1

# ML 포함 버전
.\scripts\build_exe.ps1 -IncludeML
```

#### GitHub Actions 빌드

- **워크플로우**: `.github/workflows/build-windows-exe.yml`
- **트리거**: `main` 브랜치 푸시 또는 `v*` 태그 푸시
- **아티팩트**: `motionanalyzer-gui.exe`, `motionanalyzer-gui-ml.exe`
- **보존 기간**: 30일

### 배포 체크리스트

자세한 내용은 [`docs/PRE_RELEASE_CHECKLIST.md`](docs/PRE_RELEASE_CHECKLIST.md) 참조.

---

## 🧪 테스트

### 테스트 실행

```powershell
# 전체 테스트
pytest

# 특정 테스트 파일
pytest tests/test_dream_draem.py -v

# 커버리지 포함
pytest --cov=src/motionanalyzer --cov-report=html
```

### 테스트 커버리지

- **총 테스트 수**: 68개
- **주요 테스트 영역**:
  - 벡터 분석 (`test_analysis.py`)
  - 합성 데이터 (`test_synthetic.py`)
  - Change Point Detection (`test_changepoint.py`)
  - ML 모델들 (`test_dream_draem.py`, `test_patchcore.py`)
  - 고급 특징 (`test_advanced_features.py`)
  - 정규화 (`test_normalize_features.py`)

---

## 🤝 기여 가이드

### 개발 규칙

1. **코드 스타일**: Ruff, MyPy 준수
2. **타입 힌팅**: 주요 함수에 타입 힌팅 필수
3. **테스트**: 새 기능 추가 시 테스트 작성
4. **문서화**: 주요 함수/클래스에 docstring 작성

### 커밋 메시지 규칙

- `feat:`: 새 기능 추가
- `fix:`: 버그 수정
- `docs:`: 문서 수정
- `test:`: 테스트 추가/수정
- `refactor:`: 코드 리팩토링
- `ci:`: CI/CD 설정 변경

---

## 📝 라이선스

MIT License

---

## 🙏 감사의 말

- DRAEM (Zavrtanik et al., ICCV 2021) 논문 참고
- PatchCore (Roth et al., CVPR 2022) 논문 참고
- ADBench (Han et al.) 벤치마크 프레임워크 참고

---

## 📞 지원 및 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/mjk93447-cpu/motionanalyzer/issues)
- **문서**: [`docs/`](docs/) 디렉토리 참조

---

**마지막 업데이트**: 2026년 2월 18일  
**현재 버전**: v0.2.0  
**프로젝트 상태**: ✅ 배포 준비 완료 (실제 데이터 확보 전 단계)
