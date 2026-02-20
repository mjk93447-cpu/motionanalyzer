# Phase-2 Execution Runbook

## Sequence

1. 외부 환경에서 합성데이터 기반 기능 테스트
2. GUI 동작 점검
3. exe 빌드 및 전달
4. 사내망 실데이터 preflight
5. 사내망 실데이터 분석 및 비교
6. 비식별 로그 공유

## 1) Synthetic Feature Suite

```powershell
motionanalyzer run-synthetic-suite --output-root reports/synthetic_suite
```

Expected:

- `reports/synthetic_suite/synthetic_feature_suite_report.json` 생성
- 각 시나리오에 대해 `synthetic_validation_passed=true` 확인
- `delta_vs_normal`에 시나리오별 차이값 존재

## 2) GUI Validation

```powershell
.\scripts\run_gui.ps1
```

Expected:

- Analyze 탭에서 `vectors.csv`, `summary.json` 생성
- Compare 탭에서 두 summary 비교 delta 표시

## 3) EXE Build

```powershell
.\scripts\build_exe.ps1
```

Expected:

- `dist/motionanalyzer-cli.exe` 생성

## 4) Internal Preflight

```powershell
.\scripts\prepare_internal_eval.ps1 -InputDir "data/raw/session_real_001"
```

Expected:

- preflight pass
- `reports/preflight/internal_preflight_latest.json` 생성

## 5) Internal Analysis

```powershell
motionanalyzer analyze-bundle --input-dir data/raw/session_real_001 --output-dir exports/vectors/real_session_001
motionanalyzer compare-runs --base-summary exports/vectors/baseline/summary.json --candidate-summary exports/vectors/real_session_001/summary.json
```

Expected:

- 분석 산출물 3종 생성
- 기준 대비 delta 확인 가능

## 6) Share Redacted Log

- `internal_eval/logs/result_template.csv`에 실행 요약 기록
- 원본 좌표/이미지/내부 경로는 공유 금지
