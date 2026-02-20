# Release Notes v0.2.0

**릴리즈 버전**: v0.2.0  
**릴리즈 날짜**: 2026-02-18

---

## 요약

이 릴리즈는 실제 데이터 확보 전 단계에서 사용성 및 모델 완성도 테스트를 위한 모든 기능을 포함합니다. ML 기반 이상 감지 모델, Change Point Detection 고도화, 고급 특징 엔지니어링, 그리고 배포 준비가 완료되었습니다.

---

## 새로운 기능

### 주요 기능

- **ML 기반 이상 감지 모델**:
  - DREAM 모델 (DRAEM 전략 기반)
  - PatchCore 모델 (Memory-bank 기반)
  - Ensemble 모델 (DREAM+PatchCore 결합)
  - Temporal 모델 (LSTM/GRU 기반 시계열 이상 감지)

- **Change Point Detection 고도화**:
  - 파라미터 자동 튜닝 (Grid Search, Bayesian Optimization)
  - 다중 특징 결합 (union, intersection, majority)
  - 앙상블 CPD (여러 방법 결합)

- **고급 특징 엔지니어링**:
  - 통계 특징 (skewness, kurtosis, autocorrelation)
  - 시간 도메인 특징 (frame-to-frame 변화율)
  - 주파수 도메인 특징 (FFT 기반 dominant frequency, spectral power, spectral entropy)

- **GUI 개선**:
  - Help 메뉴 추가 (Quick Start Guide, User Guide, About)
  - Time Series Analysis 탭 완성
  - ML & Optimization 탭 완성

- **예제 데이터셋**:
  - 5개 시나리오 예제 데이터셋 생성 스크립트
  - 빠른 테스트 및 학습용 데이터 제공

### 개선 사항

- 정규화 함수에 `fit_df` 파라미터 추가로 라벨 누설 방지
- Temporal 모델 벤치마크에서 시계열 구조 보존 개선
- GUI에 ML 모델 로딩 시 graceful error handling 추가
- GitHub Actions 워크플로우에 ML 포함/미포함 두 가지 EXE 빌드 추가

---

## 변경 사항

### 추가됨 (Added)

- `scripts/generate_example_datasets.py`: 예제 합성 데이터셋 생성 스크립트
- `scripts/test_build_exe.ps1`: EXE 빌드 테스트 및 검증 스크립트
- `docs/USER_GUIDE.md`: 사용자 가이드 문서
- `docs/PHASE_B_INSIGHTS.md`: Phase B 개발 인사이트 종합 문서
- `docs/PROJECT_READINESS_ASSESSMENT.md`: 프로젝트 완성도 평가 문서
- `docs/VERSION_POLICY.md`: 버전 관리 정책 문서
- `docs/PRE_RELEASE_CHECKLIST.md`: 배포 체크리스트
- `docs/FINAL_SUMMARY_PRE_RELEASE.md`: 최종 요약 문서
- `CHANGELOG.md`: 변경 이력 문서
- GUI Help 메뉴 (Quick Start Guide, User Guide, About)

### 변경됨 (Changed)

- `normalize_features` 함수: `fit_df` 파라미터 추가로 라벨 누설 방지
- `.github/workflows/build-windows-exe.yml`: ML 포함/미포함 두 가지 EXE 빌드
- `README.md`: 새로운 문서 링크 추가

### 수정됨 (Fixed)

- DREAM 모델 예측 시 reconstruction error와 discriminator 출력 결합 수정
- Temporal 모델 벤치마크에서 시계열 구조 보존 개선 (데이터셋 레벨 분할)

---

## 사용 방법

### 설치

**EXE 파일**:
1. GitHub Releases에서 다운로드:
   - `motionanalyzer-gui.exe`: 경량 버전 (ML 기능 없음)
   - `motionanalyzer-gui-ml.exe`: ML 포함 버전 (DREAM/PatchCore 사용 가능)
2. 더블클릭하여 실행

**Python 패키지**:
```powershell
pip install motionanalyzer==0.2.0
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

### 주요 변경사항 적용 방법

- **ML 모델 사용**: GUI의 "ML & Optimization" 탭에서 모델 학습 후 "Analyze" 탭에서 사용
- **Change Point Detection**: GUI의 "Time Series Analysis" 탭에서 자동 튜닝 및 앙상블 옵션 사용
- **고급 특징**: ML 모델 학습 시 자동으로 포함됨

---

## 알려진 이슈

- Temporal 모델의 성능이 합성 데이터에서 낮음 (ROC AUC ~0.10-0.25). 실제 데이터에서 재평가 필요
- 고급 특징이 합성 데이터에 과적합 가능성 있음. 실제 데이터에서 검증 필요

---

## 마이그레이션 가이드

### 이전 버전에서 업그레이드

- **v0.1.x → v0.2.0**: 주요 변경사항 없음. 기존 데이터셋과 호환됨
- **ML 모델 사용**: 처음 사용 시 GUI의 "ML & Optimization" 탭에서 모델 학습 필요
- **정규화**: 새로운 `fit_df` 파라미터 사용 권장 (라벨 누설 방지)

---

## 기여자

- motionanalyzer contributors

---

## 다운로드

- **경량 EXE**: [motionanalyzer-gui.exe](https://github.com/mjk93447-cpu/motionanalyzer/releases/download/v0.2.0/motionanalyzer-gui.exe)
- **ML 포함 EXE**: [motionanalyzer-gui-ml.exe](https://github.com/mjk93447-cpu/motionanalyzer/releases/download/v0.2.0/motionanalyzer-gui-ml.exe)
- **소스 코드**: [v0.2.0](https://github.com/mjk93447-cpu/motionanalyzer/tree/v0.2.0)

---

## 참고 문서

- [사용자 가이드](docs/USER_GUIDE.md)
- [개발 로드맵](docs/DEVELOPMENT_ROADMAP_FINAL.md)
- [Phase B 인사이트](docs/PHASE_B_INSIGHTS.md)
- [프로젝트 완성도 평가](docs/PROJECT_READINESS_ASSESSMENT.md)
- [CHANGELOG](CHANGELOG.md)
