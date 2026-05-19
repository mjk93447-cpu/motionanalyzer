# 데이터셋 폴더·파일 구조 및 명명 규칙

**목적**: 폴더명·파일명만으로 용도·규모·생성 방식을 구분하여, 잘못된 데이터 사용을 방지한다.

---

## 1. 루트 구조

```
data/
  synthetic/                    # 합성 데이터 전용
    ml_fp_focused_20k_60f/      # FP 개선·Baseline 학습용 (~20k, 60f)
    ml_100k_60f/                # 100k 교차 검증·평가용
    ml_default_1k_60f/          # 기본 소규모 (테스트용)
    ml_10k_60f/                 # 10k 규모
    ml_pretrain_balanced_3k_60f/  # GUI/릴리스 pretrain (~3k, 7:2:1, NG 균형)
    README.md                   # 본 구조 요약 및 마이그레이션
  raw/                          # 실데이터 ingest 전/후 (비디오·번들 계층)
    <video_id>/<bundle_id>/frame_*.txt, fps.txt
  real/                         # 실데이터 ML manifest
    ml_<dataset_id>/manifest.json
```

---

## 2. 폴더명 규칙 (dataset_id)

형식: **`ml_{purpose}_{count}_60f`**

| 요소 | 설명 | 예시 |
|------|------|------|
| **ml** | 생성 스크립트: `generate_ml_dataset.py` | 고정 |
| **purpose** | 목적/스케일. fp_focused \| 100k \| default \| 10k | fp_focused, 100k |
| **count** | 전체 시퀀스 수 (천 단위). 1k \| 10k \| 20k \| 100k | 20k, 100k |
| **60f** | 시퀀스당 프레임 수 (고정) | 60f |

**표준 dataset_id 매핑**

| --scale | dataset_id (폴더명) | 용도 |
|---------|---------------------|------|
| fp_focused | ml_fp_focused_20k_60f | Baseline 분석, Loop 1, 논문 학습·평가 |
| 100k | ml_100k_60f | 100k inference-only 평가 |
| hard_10k | ml_hard_10k_60f | 복잡성 증가 상정·Ensemble 필요성 검증 (inference-only) |
| pretrain_balanced | ml_pretrain_balanced_3k_60f | 균형 pretrain (7:2:1, P~0.9/R~0.7 목표) |
| default | ml_default_1k_60f | 빠른 테스트 |
| 10k | ml_10k_60f | 중간 규모 실험 |

- **폴더명만 봐도** 어떤 스케일·용도인지 구분 가능해야 하며, `ml_dataset_fp_focused` 같은 모호한 이름은 사용하지 않는다.

---

## 3. 폴더 내부 구조 (공통)

각 dataset_id 폴더 아래:

```
ml_fp_focused_20k_60f/
  manifest.json           # 필수. entries, splits, total_count, dataset_id
  normal/
    normal_0001/ ... normal_NNNN/
    normal_ld_0001/ ...   # light_distortion (label=0)
  crack_in_bending/
    crack_0001/ ...       # goal1
    micro_0001/ ...       # micro_crack
  pre_damaged/
    predam_0001/ ...
  thick_panel/
    thick_0001/ ...
  over_bending/ under_bending/ jig_vibration/  (해당 시 있을 때)
```

- **시나리오별 서브폴더명**은 스크립트·manifest와 일치시키고, 임의 변경 금지.

---

## 4. manifest.json 필수 필드

- **dataset_id**: 위 폴더명과 동일 (예: `ml_fp_focused_20k_60f`). 다른 데이터셋과 혼동 방지.
- **scale**: 생성 시 사용한 스케일 (`fp_focused` \| `100k` \| `default` \| `10k`).
- **total_count**, **splits**, **entries** (기존과 동일).

---

## 5. 구 데이터 경로와의 대응 (마이그레이션)

| 이전 경로 | 표준 경로 (권장) |
|-----------|-------------------|
| data/synthetic/ml_dataset_fp_focused | data/synthetic/ml_fp_focused_20k_60f |
| data/synthetic/ml_dataset_100k_v2 | data/synthetic/ml_100k_60f |
| data/synthetic/ml_dataset | data/synthetic/ml_default_1k_60f |

- 기존 데이터를 쓰는 경우: 위 표준 경로로 **폴더 이름만 변경**하면 됨. 내부 구조·파일명은 그대로 두어도 됨.
- 새로 생성할 때는 `generate_ml_dataset.py` 기본값이 표준 경로를 쓰므로, 별도 `--output-dir` 없이 실행하면 표준 구조로 생성된다.

---

## 6. 참조 문서

- 데이터셋 **코드/코드명** 규칙: `docs/DATASET_NAMING_RULE.md`
- 분석 결과 명명: `docs/ANALYSIS_OUTPUT_NAMING.md`

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-02-26
