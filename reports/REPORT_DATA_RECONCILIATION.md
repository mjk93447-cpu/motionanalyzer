# 보고서–실제 데이터셋 관계 분석 및 정합성 검토

**작성일**: 2026년 2월 24일  
**목적**: 결과 보고서와 실제 데이터셋 간 관계를 분석하고, 실제 데이터 기반의 명확한 결과 제공

---

## 1. 요약

| 구분 | 상태 | 비고 |
|------|------|------|
| **실제 데이터셋** | ✅ 존재 (`data/synthetic/ml_dataset`) | `--small` 스케일로 생성됨 |
| **analysis.json** | ⚠️ 이전 실행 결과 | n_test=1159, 현재 workspace 데이터와 **scale 불일치** |
| **CRACK_DETECTION_FINAL_REPORT** | ⚠️ 다른 실행/목표 기준 | Precision 100%, FP 0 — 실제 analysis.json과 **불일치** |
| **DATASET_AND_EXPERIMENT_SPEC** | 설계 기반 | 실제 데이터 없이 **코드/설계** 기준 작성 |

---

## 2. 실제 데이터셋 현황 (2026-02-24 생성)

### 2.1 생성 스케일

```
Scale: small (--small 옵션)
총 125개 데이터셋
```

| 시나리오 | 개수 | goal | label |
|----------|------|------|-------|
| normal | 100 | normal | 0 |
| light_distortion | 3 | normal | 0 |
| crack_in_bending | 10 | goal1 | 1 |
| micro_crack | 2 | goal1 | 1 |
| pre_damaged | 5 | goal2 | 1 |
| thick_panel | 5 | variant | 0 |

### 2.2 Split 분포 (manifest)

| split | 개수 |
|-------|------|
| train | 86 |
| val | 16 |
| test | 23 |

### 2.3 Goal 1 ML용 실제 사용 데이터

| 구분 | Normal (label=0) | Crack (label=1, goal1) | 합계 |
|------|------------------|-------------------------|------|
| **Train** | 75 | 8 | **83** |
| **Val** | 15 | 1 | **16** |
| **Test** | 18 | 3 | **21** |

- **Hard subset (test)**:
  - light_distortion: **1개**
  - micro_crack: **1개**

### 2.4 특징 행 수 (행 단위)

- `include_per_frame=True`, `include_global_stats=True`, `include_per_point=False` → **데이터셋당 61행**
- **Train 행**: 83 × 61 = **5,063**
- **Test 행**: 21 × 61 = **1,281**

---

## 3. 보고서별 정합성 분석

### 3.1 `reports/crack_detection_analysis/analysis.json`

| 항목 | 값 | 해석 |
|------|-----|------|
| n_test | 1,159 | 행 단위 (61행/데이터셋) → **약 19개 데이터셋** |
| n_normal | 976 | 976/61 ≈ 16 normal 데이터셋 |
| n_crack | 183 | 183/61 ≈ 3 crack 데이터셋 |

**결론**: `analysis.json`은 **이전 실행** (소규모 ~19 test 데이터셋) 결과. 현재 workspace small 데이터(21 test)와 **scale 유사**하나, 동일 시드/구성은 아님.

**실제 결과 (analysis.json 기준)**:

| 모델 | Precision | Recall | FP | FN | TP | TN |
|------|-----------|--------|-----|-----|-----|-----|
| DREAM | 0.7188 | 0.3770 | 27 | 114 | 69 | 949 |
| PatchCore | 0.7000 | 0.3825 | 30 | 113 | 70 | 946 |
| Ensemble | 0.7727 | 0.3716 | 20 | 115 | 68 | 956 |

- Hard subset: light_distortion n=1, micro_crack n=1 (둘 다 0% 정상 분류, 100% 크랙 분류)

### 3.2 `reports/CRACK_DETECTION_FINAL_REPORT.md`

| 항목 | 보고서 주장 | 실제 (analysis.json) |
|------|-------------|------------------------|
| Precision (Ensemble) | **100%** | **77.3%** |
| FP | **0** | **20** |
| Recall | 65.2% | 37.2% |
| light_distortion 정상 분류 | 8/8 (100%) | 0/1 (0%) |
| micro_crack 크랙 분류 | 2/2 (100%) | 1/1 (100%) |

**결론**: `CRACK_DETECTION_FINAL_REPORT`는 **다른 실행/스케일** (예: 10k scale, full dataset) 또는 **목표/향후 성과** 기준으로 작성된 것으로 보임. 현재 workspace의 `analysis.json` 결과와 **일치하지 않음**.

### 3.3 `reports/DATASET_AND_EXPERIMENT_SPEC.md`

**기준**: `generate_ml_dataset.py`의 **default scale** 설계 (총 1,180개)

**실제 workspace**: `--small` scale (총 125개)

| 구분 | DATASET_AND_EXPERIMENT_SPEC | 실제 (small) |
|------|-----------------------------|--------------|
| Train 데이터셋 | 812 | 83 |
| Test 데이터셋 | 173 | 21 |
| light_distortion (test) | 8 | 1 |
| micro_crack (test) | 2 | 1 |

---

## 4. 실제 데이터 기반 명확한 결과

### 4.1 현재 workspace에서 검증 가능한 결과

- **분석 스크립트**: `analyze_crack_detection.py` 실행 시 PyTorch 필요
- **현재**: ML 의존성 없어 **재실행 불가** → `analysis.json`이 **가장 최근 실행 결과**로 간주

### 4.2 analysis.json 기반 (실제 실행 결과)

| 지표 | DREAM | PatchCore | Ensemble |
|------|-------|-----------|----------|
| **Precision** | 71.9% | 70.0% | **77.3%** |
| **Recall** | 37.7% | 38.3% | 37.2% |
| **F1** | 0.495 | 0.495 | 0.502 |
| **ROC AUC** | 0.895 | 0.929 | 0.000* |
| **FP** | 27 | 30 | 20 |
| **FN** | 114 | 113 | 115 |

\* Ensemble은 ROC AUC 미적용 (both_agree 전략)

- **Test 규모**: ~19 데이터셋 (1,159 행)
- **Hard subset**: light_distortion 1개 → 0% 정상 분류, micro_crack 1개 → 100% 크랙 분류

### 4.3 보고서와의 차이

| 보고서 | Precision | FP | 데이터 |
|--------|-----------|-----|--------|
| CRACK_DETECTION_FINAL_REPORT | 100% | 0 | 10k/다른 scale 추정 |
| **analysis.json (실제)** | **77.3%** | **20** | ~19 test 데이터셋 |

---

## 5. 권장 사항

1. **데이터 생성**: `python scripts/generate_ml_dataset.py` (default scale)로 전체 데이터셋 생성 후 재분석
2. **보고서 정합**: `CRACK_DETECTION_FINAL_REPORT`에 **실행/스케일/날짜** 명시, 또는 `analysis.json` 기준으로 수정
3. **DATASET_AND_EXPERIMENT_SPEC**: 실제 사용 스케일(small/default/10k)을 명시하고, 실행 시 manifest 기반으로 검증
4. **재현성**: `pip install -e '.[ml]'` 후 `python scripts/analyze_crack_detection.py` 실행 시, 현재 manifest와 동일한 결과 재현 가능

---

## 6. 참조

- `data/synthetic/ml_dataset/manifest.json`: 실제 데이터 manifest
- `scripts/analyze_crack_detection.py`: Goal 1 ML 평가, train/val/test 구분
- `reports/crack_detection_analysis/analysis.json`: 실제 실행 결과
