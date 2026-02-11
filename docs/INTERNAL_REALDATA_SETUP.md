# Internal Real-Data Setup (Sequential)

## 목적

사내 실데이터 분석 전, 필수 준비를 누락 없이 순차 수행한다.

## Step 0. exe 준비

1. GitHub Actions에서 `motionanalyzer-cli.exe` 아티팩트 다운로드
2. 사내 PC 작업 폴더에 배치

## Step 1. 폴더/템플릿 초기화

```powershell
.\.venv\Scripts\motionanalyzer prepare-internal
```

생성 항목:

- `internal_eval/inbox`
- `internal_eval/outbox`
- `internal_eval/logs/result_template.csv`
- `reports/preflight/`

## Step 2. 실데이터 사전점검 (Preflight)

```powershell
.\.venv\Scripts\motionanalyzer preflight-realdata `
  --input-dir data/raw/session_real_001 `
  --report-path reports/preflight/internal_preflight_latest.json
```

점검 항목:

- `fps.txt` 존재 및 숫자 형식
- `frame_*.txt` 연속성(누락 프레임)
- 헤더/행 포맷(`# x,y,index`)
- 프레임별 포인트 수 범위
- 인덱스 집합 일관성

## Step 3. 실패 시 조치

- 프레임 누락: 파일 수집/복사 재확인
- 포맷 오류: 추출기 포맷을 `x,y,index` 정수로 통일
- 인덱스 불일치: 트래킹 파이프라인 재수출

## Step 4. 로그 공유 원칙

- `internal_eval/logs/result_template.csv` 기준으로 비식별 요약 로그만 공유
- 원본 좌표, 원본 이미지, 내부 경로는 외부 공유 금지

## Step 5. EXE 단일 실행 (권장)

exe로 preflight + 분석 + 비교를 한 번에 수행:

```powershell
.\dist\motionanalyzer-cli.exe internal-realdata-run `
  --input-dir data/raw/session_real_001 `
  --output-dir exports/vectors/real_session_001 `
  --baseline-summary exports/vectors/baseline/summary.json `
  --run-log-txt internal_eval/logs/internal_run_latest.txt
```

`--baseline-summary`를 비워 비교를 생략할 수 있음:

```powershell
.\dist\motionanalyzer-cli.exe internal-realdata-run `
  --input-dir data/raw/session_real_001 `
  --output-dir exports/vectors/real_session_001 `
  --baseline-summary "" `
  --run-log-txt internal_eval/logs/internal_run_latest.txt
```

## 원클릭 스크립트

```powershell
.\scripts\prepare_internal_eval.ps1 -InputDir "data/raw/session_real_001"
```
