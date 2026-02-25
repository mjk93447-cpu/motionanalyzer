# 연구 완성 및 수준 높은 논문 제작 마스터 계획

**목적**: 로드맵 부족 부분 보완, 상황분석→개선개발→테스트→평가분석→추가개발 루프 3회 이상 실행, 연구 완성 후 논문 제작을 계획적으로 진행.

---

## Part 1. 로드맵 부족 부분 분석

### 1.1 현재 달성 상태

| 항목 | 상태 | 비고 |
|------|:----:|------|
| Normal FP Rate ≤ 0.1% | ✅ | fp_focused 기준 FP=0, 0.00% |
| Phase 1.1 (normal-fp-max=0) | ✅ | 구현·검증 완료 |
| Phase 1.2 (percentile 99.9) | ✅ 구현 | ⚠️ 실행·검증 미완 |
| Phase 2a (threshold-margin) | ✅ 구현 | ⚠️ 실행·검증 미완 |
| Phase 2 (improvement loop) | ✅ 구현 | ⚠️ 6 step 전부 실행·비교 미완 |
| 100k 교차 검증 | 🔄 | 스크립트 있음, 결과 미산출 |
| Phase 3 (Class weight 등) | 🔄 | 미구현 |
| Ablation study | ❌ | 없음 |
| 논문용 체계적 실험 로그 | ❌ | 없음 |

### 1.2 부족한 부분

| 부족 항목 | 내용 |
|-----------|------|
| **대규모 검증** | fp_focused(851 test)만 검증, 100k(5k test)에서 일반화·재현성 미확인 |
| **임계값 전략 비교** | normal-fp-max vs percentile vs margin 조합에 대한 정량 비교 없음 |
| **개발 루프 미정형** | 상황분석→개선→테스트→평가→추가개발의 반복 구조가 명시되지 않음 |
| **연구 완료 정의** | 논문 작성 가능한 “완료” 기준이 없음 |
| **논문 제작 순서** | 실험 완료 → 데이터 고정 → 논문 작성 순서와 체크리스트 부재 |

---

## Part 2. 개발 루프 정의 (3회 이상)

### 2.1 루프 구조 (Iterative Refinement, 문헌 기반)

**문헌**: Self-Refine (NeurIPS 2023), Verifier-Guided Refinement (arXiv 2504.01931), NeurIPS Reproducibility (arXiv 2003.12206)

```
[상황분석] → [성능개선개발] → [테스트] → [평가분석] → [추가개발]
     ↑                                                      |
     +------------------------------------------------------+
```

**핵심 원칙** (문헌 요약):
- **Gap → Action**: 상황분석에서 간극 정의, 개선 개발로 직접 해소
- **평가 기준 명시**: FP, Recall, Precision 등 메트릭 사전 정의
- **과잉 보정 방지**: 반복 시 diminishing returns 주의
- **재현성**: 스크립트·설정·시드 고정, 대규모 검증(100k)으로 일반화 확인

### 2.2 루프 1: fp_focused 임계값 전략 검증

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **상황분석** | 현재: normal-fp-max=0만 검증. percentile, margin 미검증 | gap 정의 |
| **개선개발** | (이미 구현됨) percentile 99.9, margin 0.1/0.2 적용 | - |
| **테스트** | `run_normal_fp_improvement_loop.py --max-train 2000` 6 step 실행 | analysis.json ×6 |
| **평가분석** | FP, Recall, Precision 비교표 작성 | `reports/loop1_threshold_comparison.json` |
| **추가개발** | 최적 설정 선정, 100k 평가용으로 고정 | config/args 고정 |

**실행**:
```powershell
python scripts/run_normal_fp_improvement_loop.py --base-dir data/synthetic/ml_dataset_fp_focused --max-train 2000
```

### 2.3 루프 2: 100k 교차 검증

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **상황분석** | fp_focused만 검증됨. 100k에서 일반화 불명 | gap 정의 |
| **개선개발** | (이미 구현) inference-only 100k 평가 | - |
| **테스트** | `evaluate_100k_inference_only.py` 실행 | analysis_100k_inference.json |
| **평가분석** | DREAM/PatchCore/Ensemble 100k CM 비교, TN+FP+FN+TP=100k 확인 | insights_100k.md |
| **추가개발** | 100k 결과가 허용 범위 밖이면 임계값·설정 재검토 | 필요 시 루프 1로 회귀 |

**실행**:
```powershell
python scripts/evaluate_100k_inference_only.py
```

### 2.4 루프 3: Ablation 및 견고성

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **상황분석** | 앙상블 기여도, light_distortion/micro_crack 하위 메트릭 미문서화 | gap 정의 |
| **개선개발** | ablation 스크립트: DREAM only, PatchCore only, Ensemble 비교 | - |
| **테스트** | fp_focused + 100k 양쪽에서 DREAM/PatchCore/Ensemble 각각 실행 | ablation_results.json |
| **평가분석** | Ensemble이 DREAM·PatchCore 단독보다 FP 감소·Precision 향상 확인 | ablation_summary.md |
| **추가개발** | 논문용 표·그림 데이터 고정 | - |

### 2.5 루프 4 (선택): Phase 3 학습 개선

| 단계 | 작업 | 산출물 |
|------|------|--------|
| **상황분석** | FP=0 달성했으나 Recall·FN 개선 여지 | gap 정의 |
| **개선개발** | class_weight, FP 가중 손실 등 Phase 3 항목 구현 | dream.py, analyze_crack_detection.py |
| **테스트** | 개선된 모델로 fp_focused·100k 재평가 | - |
| **평가분석** | Recall·FN 변화, FP 유지 여부 확인 | - |
| **추가개발** | 필요 시 반복 | - |

---

## Part 3. 연구 완료 기준 (명확화)

### 3.1 필수 충족 항목

| # | 기준 | 검증 방법 |
|---|------|-----------|
| 1 | Normal FP Rate ≤ 0.1% (fp_focused) | analysis.json `normal_fp_rate` |
| 2 | **100k 교차 검증 완료** | analysis_100k_inference.json 존재 (논문 신뢰성) |
| 3 | 임계값 전략 비교표 | loop1 또는 수동 실험 결과 문서화 |
| 4 | DREAM / PatchCore / Ensemble 비교 | confusion matrix 3종 |
| 5 | hard subset (light_distortion, micro_crack) 결과 | hard_subset_metrics |

### 3.2 논문 작성 가능 조건

- [ ] 위 1~5 충족
- [ ] 최종 수치·표·그림 고정 (더 이상 실험 변경 없음)
- [ ] 레거시 산출물 아카이빙 완료 (`archive_legacy_deliverables.ps1`)

### 3.3 논문 완성도 평가 기준

**문헌**: IMRaD, QuOCCA, NeurIPS Reproducibility → `docs/PAPER_QUALITY_CRITERIA.md` 참조.

---

## Part 4. 논문 제작 순서 (연구 완료 후)

| 순서 | 작업 | 스크립트/명령 |
|------|------|---------------|
| 1 | 레거시 아카이빙 | `.\scripts\archive_legacy_deliverables.ps1` |
| 2 | 최신 분석 확보 | `analyze_crack_detection.py` (fp_focused) |
| 3 | 100k 평가 확보 | `evaluate_100k_inference_only.py` |
| 4 | Paper Banana 도표 (선택) | `paperbanana generate -i docs/paperbanana_inputs/fpcb_methodology.txt ...` |
| 5 | Word 논문 생성 | `generate_final_report_docx.py` |
| 6 | PDF 생성 | docx2pdf 또는 Word 수동 |
| 7 | PPT 생성 | `generate_final_report_ppt.py` |

---

## Part 5. 실행 체크리스트 (3회 이상 루프)

### 루프 1: 임계값 전략 검증

```powershell
# 1) 상황분석: 현재 baseline 확인
#    reports/crack_detection_analysis/analysis.json 확인 (FP=0, Recall 98.04%)

# 2) 테스트: 6 step improvement loop
python scripts/run_normal_fp_improvement_loop.py --base-dir data/synthetic/ml_dataset_fp_focused --max-train 2000

# 3) 평가분석: 각 step 결과 수동 기록 또는 스크립트로 자동 기록
#    → reports/loop1_threshold_comparison.json (수동 생성 권장)
```

### 루프 2: 100k 교차 검증

```powershell
# 1) 테스트
python scripts/evaluate_100k_inference_only.py

# 2) 평가분석: analysis_100k_inference.json 확인
#    TN+FP+FN+TP ≈ 100k, DREAM/PatchCore/Ensemble CM 비교
```

### 루프 3: Ablation (선택, 수동 비교 가능)

- DREAM only, PatchCore only, Ensemble 결과는 이미 analysis.json에 포함
- 별도 ablation_results.json 생성은 선택

### 논문 제작 (연구 완료 후)

```powershell
.\scripts\archive_legacy_deliverables.ps1
.\scripts\run_paper_pipeline.ps1
# 또는 -SkipAnalysis -SkipPaperBanana 등으로 Word/PPT만
```

---

## Part 6. 문서·스크립트 연계

| 문서 | 용도 |
|------|------|
| `crack_detection_analysis/insights.md` | 성능, 전략, domain gap, 데이터셋 선택 |
| `DEVELOPMENT_PROGRESS.md` | 단계별 진행 상태 |
| `PAPER_WRITING_AND_PAPERBANANA_PLAN.md` | 논문 작성·Paper Banana 사용법 |
| `REFERENCE_PAPERS_FOR_PAPER_WRITING.md` | 참고 논문·참고문헌 |
| `PAPER_QUALITY_CRITERIA.md` | 논문 완성도 평가 기준 (IMRaD·QuOCCA·재현성) |
| **RESEARCH_AND_PAPER_MASTER_PLAN.md** (본 문서) | 루프·완료 기준·논문 순서 통합 |

---

## Part 7. 루프 결과 기록 템플릿

### loop1_threshold_comparison.json (예시)

```json
{
  "base_dir": "data/synthetic/ml_dataset_fp_focused",
  "steps": [
    {"label": "normal-fp-max=0", "fp": 0, "recall": 0.98, "precision": 1.0},
    {"label": "normal-fp-max=0 +margin=0.1", "fp": 0, "recall": 0.97, "precision": 1.0},
    {"label": "percentile=99.9", "fp": 1, "recall": 0.96, "precision": 0.99}
  ]
}
```

### loop2_100k_summary (insights_100k.md에 반영)

- n_test, n_normal, n_crack
- DREAM/PatchCore/Ensemble 각 TN, FP, FN, TP, Precision, Recall, Normal FP Rate

---

**문서 버전**: 1.0 | **최종 갱신**: 2026-02-23
