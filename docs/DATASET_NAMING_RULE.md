# 데이터셋 코드 명명 규칙

**목적**: 데이터셋별로 한 번에 찾을 수 있도록, 코드에서 생성방식·구성·규모를 알 수 있게 함.

---

## 1. 코드 형식

```
DS-{YYMMDD}-{GEN}-{PURPOSE}-{COUNT}k-{F}f
```

| 요소 | 설명 | 예시 |
|------|------|------|
| **YYMMDD** | 생성일 (년월일) | 260220 = 2026-02-20 |
| **GEN** | 생성 스크립트 약어 | ml = generate_ml_dataset.py |
| **PURPOSE** | 목적/스케일 | 100k, fp, ex |
| **COUNT** | 전체 시퀀스 수 (천 단위) | 100k, 20k |
| **F** | 시퀀스당 프레임 수 | 60f |

---

## 2. GEN(생성 스크립트) 매핑

| GEN | 스크립트 | 설명 |
|-----|----------|------|
| ml | `scripts/generate_ml_dataset.py` | ML 학습·평가용 합성 데이터 |
| ex | `scripts/generate_example_datasets.py` | 예제·튜토리얼용 소량 |
| fpcb | `scripts/prepare_fpcb_test_suite.py` | FPCB 테스트 시나리오 |

---

## 3. PURPOSE(목적) 예시

| PURPOSE | 설명 |
|---------|------|
| 100k | 100k 스케일 ML 학습·평가 |
| fp | FP 개선, Precision 중심 (fp_focused) |
| ex | 예제 (5개 시나리오) |

---

## 4. 코드 해석 예시

| 코드 | 해석 |
|------|------|
| **DS-260220-ml-100k-100k-60f** | 2026-02-20 생성, generate_ml_dataset.py, 100k 목적, 100k 시퀀스, 60프레임 |
| **DS-260223-ml-fp-20k-60f** | 2026-02-23 생성, generate_ml_dataset.py, fp 목적, 20k 시퀀스, 60프레임 |

---

## 5. 상세 메타데이터 (코드 외)

구성비·설명 등은 코드에 직접 넣지 않고, `reports/BENDING_DATASETS_INVENTORY.md` 또는 `manifest.json`에 기록.

| 항목 | 저장 위치 |
|------|-----------|
| 시나리오별 구성비 | BENDING_DATASETS_INVENTORY.md, manifest.json |
| 생성 명령 | BENDING_DATASETS_INVENTORY.md |
| 폴더 경로 | BENDING_DATASETS_INVENTORY.md, mcp.json 등 |
