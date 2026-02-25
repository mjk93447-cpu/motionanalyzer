# Bending 영상 데이터셋 인벤토리

**분석일**: 2026년 2월 25일  
**범위**: `data/synthetic/` — DS-260220-ml-100k-100k-60f, DS-260223-ml-fp-20k-60f만 유지 (연구 혼동 방지)  
**명명 규칙**: `docs/DATASET_NAMING_RULE.md`

---

## 요약 표

| 코드 | 폴더 경로 | 생성 시점 | 전체 수 | 프레임 | 시나리오 구성 | 용도 |
|------|-----------|-----------|---------|--------|---------------|------|
| **DS-260220-ml-100k-100k-60f** | `data/synthetic/ml_dataset_100k_v2` | 2026-02-20 | 100,000 | 60 | normal 75k, crack_in_bending 16k, pre_damaged 4k, thick_panel 5k | ML 학습·평가 (100k 스케일) |
| **DS-260223-ml-fp-20k-60f** | `data/synthetic/ml_dataset_fp_focused` | 2026-02-23 | 20,000 | 60 | normal 16k, thick_panel 2.8k, crack_in_bending 1.05k, pre_damaged 150 | FP 개선·Precision 중심 |

---

## 상세 분석

### DS-260220-ml-100k-100k-60f: ml_dataset_100k_v2

| 항목 | 내용 |
|------|------|
| **코드** | DS-260220-ml-100k-100k-60f (생성일-스크립트-목적-총수-프레임) |
| **경로** | `data/synthetic/ml_dataset_100k_v2` |
| **생성** | 2026-02-20 20:12 |
| **생성 스크립트** | `scripts/generate_ml_dataset.py` |
| **생성 명령** | `--scale 100k --output-dir data/synthetic/ml_dataset_100k_v2` |
| **전체 수** | 100,000 시퀀스 |
| **프레임/시퀀스** | 60 (고정) |
| **FPS** | 30 |
| **구조** | `normal/`, `crack_in_bending/`, `pre_damaged/`, `thick_panel/` |
| **Split** | train 90k, val 5k, test 5k |
| **manifest** | `manifest.json` |

---

### DS-260223-ml-fp-20k-60f: ml_dataset_fp_focused

| 항목 | 내용 |
|------|------|
| **코드** | DS-260223-ml-fp-20k-60f (생성일-스크립트-목적-총수-프레임) |
| **경로** | `data/synthetic/ml_dataset_fp_focused` |
| **생성** | 2026-02-23 09:00 |
| **생성 스크립트** | `scripts/generate_ml_dataset.py` |
| **생성 명령** | `--scale fp_focused --output-dir data/synthetic/ml_dataset_fp_focused` |
| **전체 수** | ~20,000 시퀀스 |
| **프레임/시퀀스** | 60 |
| **구조** | normal 16k, thick_panel 2.8k, crack_in_bending 1.05k, pre_damaged 150 |
| **용도** | False Positive 개선, Precision 0.99 목표 |
| **manifest** | `manifest.json` |

---

## 폴더 구조

```
data/synthetic/
├── ml_dataset_100k_v2/          [DS-260220-ml-100k-100k-60f] 100k, 60f
└── ml_dataset_fp_focused/       [DS-260223-ml-fp-20k-60f] 20k, 60f
```

---

## 총계

| 구분 | 수치 |
|------|------|
| **활성 데이터셋** | 2개 |
| **총 시퀀스 수** | ~120,000 |
| **프레임 수** | 60 |
| **FPS** | 30 |
