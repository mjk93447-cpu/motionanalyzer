# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 10k/100k 스케일 합성 데이터셋 생성 옵션 및 병렬 생성 워크플로우 (`scripts/generate_ml_dataset.py`)
- GPU/Jupyter 실행 체계 및 운영 문서/노트북 추가 (`docs/PIPELINE_SETUP_COMPLETE.md`, `notebooks/*`)
- Cursor 프로젝트 스킬/룰 및 유지보수 스크립트 추가 (`.cursor/skills/*`, `.cursor/rules/*`, `scripts/cursor-speed-optimization/*`)
- Phase B.5: Change Point Detection 고도화 (파라미터 자동 튜닝, 다중 특징 결합, 앙상블 CPD)
- Phase B.4: 고급 특징 엔지니어링 (통계/시간/주파수 도메인 특징)
- Phase B.3: Temporal Modeling (LSTM/GRU 기반 시계열 이상 감지)
- Phase B.2: 앙상블 모델 구현 (DREAM+PatchCore 결합)
- Phase B.1: 합성 데이터 물리 현상 추가 (충격파, 미세 진동)
- Phase A.2: Change Point Detection GUI 통합 (Time Series Analysis 탭)
- Phase A.1: EXE 빌드 스크립트 ML 포함/미포함 옵션화
- GUI Help 메뉴 추가 (Quick Start Guide, User Guide, About)
- 예제 합성 데이터셋 생성 스크립트 (`scripts/generate_example_datasets.py`)
- EXE 빌드 테스트 스크립트 (`scripts/test_build_exe.ps1`)

### Changed
- 리포트 생성기가 `analysis.json` 기반으로 테이블 메트릭을 동적으로 반영하도록 개선 (`scripts/generate_final_report_docx.py`)
- 크랙 분석 임계값 선정 로직을 validation 우선 방식으로 개선 (`scripts/analyze_crack_detection.py`)
- 합성 생성기에 재현 가능한 노이즈 모드(gaussian/outlier/temporal_drift/scale_jitter/mixed) 추가 (`src/motionanalyzer/synthetic.py`)
- 버전 상수 정합화: 패키지 버전을 `0.2.0`으로 통일 (`src/motionanalyzer/__init__.py`)
- 정규화 함수에 `fit_df` 파라미터 추가로 라벨 누설 방지
- GUI에 ML 모델 로딩 시 graceful error handling 추가
- GitHub Actions 워크플로우에 ML 포함/미포함 두 가지 EXE 빌드 추가

### Fixed
- 보고서 요약 문서 오염 텍스트 자동 복구 및 검증 완료 (`reports/goal_achievement_summary.md`)
- Git 추적 잡음 방지: GPU 가상환경 및 임시 산출물 무시 규칙 보강 (`.gitignore`)
- DREAM 모델 예측 시 reconstruction error와 discriminator 출력 결합 수정
- Temporal 모델 벤치마크에서 시계열 구조 보존 개선

### Documentation
- `docs/PHASE_B_INSIGHTS.md`: Phase B 개발 인사이트 종합 문서 추가
- `docs/PROJECT_READINESS_ASSESSMENT.md`: 프로젝트 완성도 평가 문서 추가
- `docs/USER_GUIDE.md`: 사용자 가이드 작성
- `docs/VERSION_POLICY.md`: 버전 관리 정책 문서화
- `docs/PRE_RELEASE_CHECKLIST.md`: 배포 체크리스트 작성
- `docs/FINAL_SUMMARY_PRE_RELEASE.md`: 최종 요약 문서 작성
- `CHANGELOG.md`: 변경 이력 정리

## [0.2.0] - 2026-02-18

### Added
- 오프라인 Windows GUI (Tkinter 기반)
- 합성 데이터 생성기 (FPCB 벤딩 시뮬레이션)
- 벡터 분석 엔진 (위치/속도/가속도 계산)
- DREAM 이상 감지 모델 (DRAEM 전략)
- PatchCore 이상 감지 모델
- Change Point Detection (CUSUM, Window-based, PELT)
- CLI 인터페이스 (Typer 기반)

### Documentation
- README.md 작성
- 개발 로드맵 문서 작성
- 물리/화학 지식베이스 문서 작성

## [0.1.0] - 2026-01-XX

### Added
- 초기 프로젝트 구조
- 기본 분석 기능
