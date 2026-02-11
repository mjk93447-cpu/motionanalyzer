# Internal Real-Data Test Scenarios

## Scope

사내망에서 `motionanalyzer-cli.exe`를 사용해 실데이터를 평가할 때,
확인할 기능/로그/예상 결과를 표준 시나리오로 정의한다.

## Pre-conditions

- `prepare-internal` 및 `preflight-realdata` 완료
- 대상 데이터: `frame_*.txt`, `fps.txt`
- 비식별 로그 템플릿: `internal_eval/logs/result_template.csv`

## Features to Verify

1. 입력 데이터 유효성 점검(preflight)
2. 벡터 분석 실행(analyze-bundle)
3. 요약 리포트 생성(summary.json)
4. 기준 런과 비교(compare-runs)
5. 공유용 비식별 로그 작성

## Scenario Matrix

### S1. Baseline Healthy Lot

- 목적: 정상 공정에서 기준 지표 확보
- 입력: 정상 샘플 다수 프레임
- 실행:
  - `motionanalyzer preflight-realdata --input-dir <BASELINE_DIR>`
  - `motionanalyzer analyze-bundle --input-dir <BASELINE_DIR> --output-dir <OUT_BASELINE>`
- 확인 로그:
  - preflight report: fail 없음
  - summary: 프레임 수/인덱스 수/평균 속도/평균 가속도
- 예상 결과:
  - preflight passed
  - 벡터 산출 정상 (`vectors.csv`, `vectors.txt`, `summary.json`)

### S2. Suspected Crack Behavior

- 목적: 국부 힌지형 변형 징후 검출 가능성 확인
- 입력: crack 의심 lot
- 실행:
  - baseline 분석 후 candidate 분석
  - `motionanalyzer compare-runs --base-summary <OUT_BASELINE>/summary.json --candidate-summary <OUT_CRACK>/summary.json`
- 확인 로그:
  - `delta_max_curvature_like` 증가 여부
  - `delta_mean_acceleration` 증가 여부
- 예상 결과:
  - baseline 대비 곡률 surrogate 상승 경향
  - 극단 프레임에서 가속도 피크 상승

### S3. Thick/Over-Cured Candidate

- 목적: 과두께/과경화에 의한 "덜 접힘" 패턴 확인
- 입력: stiff 의심 lot
- 실행:
  - baseline과 동일 절차로 분석/비교
- 확인 로그:
  - 분석 summary에서 속도/가속도 분포 변화
  - 필요 시 최종 frame 형상 비교(내부 시각화)
- 예상 결과:
  - baseline 대비 동적 응답 둔화(평균 속도/가속도 하락 가능)
  - 특정 구간 snap-like 이상이 있으면 가속도 피크 상승 가능

### S4. Data Integrity Failure (Negative)

- 목적: 데이터 이상 감지 체계 확인
- 입력: 누락 frame 또는 포맷오류 데이터
- 실행:
  - `motionanalyzer preflight-realdata --input-dir <BROKEN_DIR>`
- 예상 결과:
  - preflight failed
  - report에 누락/포맷/index mismatch 오류 명시

## Logging Template (Required)

`internal_eval/logs/result_template.csv`에 아래 항목 기록:

- `run_id`, `build_version`, `scenario_tag`
- `frame_count`, `fps`, `elapsed_ms`
- `validation_passed`, `failure_signature`
- `notes_redacted`

## Single TXT Log Output (EXE)

권장 실행은 `internal-realdata-run`으로 단일 실행 로그를 남긴다.

- command log: `internal_eval/logs/internal_run_latest.txt`
- analysis text summary: `exports/vectors/<run>/summary.txt`
- vector text export: `exports/vectors/<run>/vectors.txt`
- compare text: `reports/compare/internal_vs_baseline.txt` (baseline 존재 시)

## Pass/Fail Guideline

- **Pass**
  - preflight 통과
  - 분석 산출물 3종 생성 (`vectors.csv`, `vectors.txt`, `summary.json`)
  - 시나리오별 기대 변화(증가/감소 경향)가 관측됨
- **Fail**
  - preflight 실패
  - 분석 중 예외
  - 비교 지표가 무의미(데이터 누락/프레임수 과소 등)

## Security Note

- 외부 공유는 `summary.json` 요약값 + `result_template.csv` 기반 비식별 로그로 제한
- 원본 좌표 전체/원본 이미지/내부 경로는 외부 반출 금지
