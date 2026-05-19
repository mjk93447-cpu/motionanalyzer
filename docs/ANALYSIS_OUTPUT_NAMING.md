# 분석 결과 저장·명명 규칙 (Analysis Output Naming)

**목적**: 반복 실험 시 분석 결과를 구분하고, 잘못된 결과 참조를 방지한다.

---

## 1. Run ID 체계

모든 분석 실행은 **run_id** 하나로 식별한다.

| 구성요소 | 설명 | 예시 |
|----------|------|------|
| **dataset_slug** | base_dir 마지막 폴더명 (데이터셋 구분) | `ml_fp_focused_20k_60f`, `ml_100k_60f` |
| **train_limit** | `--max-train` 값. 없으면 `full` | `100`, `2000`, `full` |
| **timestamp** | 실행 시각 `YYYYMMDD_HHMM` (UTC 권장) | `20260226_0447` |

**형식**: `{dataset_slug}_{train_limit}_{timestamp}`

**예**:
- `fp_focused_2000_20260226_0447` — fp_focused, max-train 2000, 2026-02-26 04:47
- `fp_focused_full_20260226_1200` — fp_focused, 전체 학습, 12:00
- `ml_100k_60f_10000_20260226_1500` — 100k 데이터, max-eval 10000

---

## 2. 파일 명명 규칙

### 2.1 Baseline 분석 (analyze_crack_detection.py)

| 파일 패턴 | 용도 |
|-----------|------|
| `analysis_{run_id}.json` | **고유 보관** — 실행마다 새 파일, 덮어쓰지 않음 |
| `analysis.json` | **최신 실행 결과** — 파이프라인/스크립트가 참조하는 기본 파일 (위 파일의 복사본) |

동일 디렉터리: `reports/crack_detection_analysis/`

- 매 실행 시 `analysis_{run_id}.json` 저장 후, 그 내용을 `analysis.json`에 복사한다.
- 기존 스크립트·Queue는 `analysis.json`만 읽어도 되며, 실험 이력은 `analysis_*.json`으로 구분해 참조할 수 있다.

### 2.2 100k Inference (evaluate_100k_inference_only.py)

| 파일 패턴 | 용도 |
|-----------|------|
| `analysis_100k_{run_id}.json` | **고유 보관** — run_id에 eval 데이터셋·max_eval·timestamp 포함 |
| `analysis_100k_inference.json` | **최신 100k 결과** — 파이프라인 기본 참조 (위 파일의 복사본) |

run_id 예: `100k_v2_10000_20260226_1600` (eval_base 슬러그, max_eval, timestamp)

### 2.3 기타 보고서·인사이트

| 파일 | 용도 |
|------|------|
| `insights_canonical.md` | **인사이트만** (수치·표 없음). Vector Map 해석, Key Insights, Detection Strategy. 실행과 무관하게 유지. |
| `insights.md` | 최신 실행 요약 + `analysis.json`·`insights_canonical.md` 로의 링크만 포함. |
| `archive/insights_{run_id}.md` | 실행별 전체 표·CM·리포트. 과거 결과 격리. |
| `archive/README.md` | 아카이브 규칙 및 격리 대상 설명. |
| `loop1_threshold_comparison.json` | Loop 1 단계별 결과 (필요 시 run_id 접미사 확장 가능) |

---

## 3. JSON 내부 메타데이터 (run_meta)

각 분석 JSON에는 **run_meta** 필드를 넣어, 파일만 보고도 어떤 실행인지 알 수 있게 한다.

```json
{
  "run_meta": {
    "run_id": "fp_focused_2000_20260226_0447",
    "script": "analyze_crack_detection.py",
    "base_dir": "data/synthetic/ml_fp_focused_20k_60f",
    "max_train": 2000,
    "test_base_dir": null,
    "dataset_level_eval": true,
    "normal_fp_max": 0,
    "timestamp_iso": "2026-02-26T04:47:00Z"
  },
  "n_test": 548,
  "models": { ... }
}
```

- **참조 시**: 수치만 쓰지 말고, 논문/보고서에는 `run_id` 또는 `timestamp_iso`를 함께 적어 출처를 명확히 한다.

---

## 4. 사용 규칙 (정리)

1. **실험할 때**: `analysis.json`을 직접 수정하지 말고, 새로 실행해 `analysis_{run_id}.json`이 생성되도록 한다.
2. **비교할 때**: 비교 대상마다 `analysis_{run_id}.json` 경로를 명시해 사용한다.
3. **파이프라인(Queue 등)**: 계속 `analysis.json`(및 필요 시 `analysis_100k_inference.json`)만 읽어도 되며, “최신 baseline/100k 결과”로 동작한다.
4. **보고·논문**: 인용한 수치가 어느 실행인지 구분할 수 있도록 run_id 또는 run_meta.timestamp_iso를 문서에 남긴다.

---

## 5. 관련 스크립트

| 스크립트 | 출력 JSON (고유) | 출력 JSON (최신) |
|----------|------------------|------------------|
| `analyze_crack_detection.py` | `analysis_{run_id}.json` | `analysis.json` |
| `evaluate_100k_inference_only.py` | `analysis_100k_{run_id}.json` | `analysis_100k_inference.json` |

---

## 6. 기존 JSON과의 호환

- `run_meta`가 없는 예전 `analysis.json` / `analysis_100k_inference.json`도 그대로 유효하다. 스크립트는 `n_test`, `models` 등 기존 필드만 읽으면 된다.
- 새로 저장되는 파일만 `run_meta`와 `analysis_{run_id}.json` 규칙을 따른다.

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-02-26
