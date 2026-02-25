# PaperBanana / Nano-Banana Figure Generation Guide

논문 이미지 및 도표 품질 개선을 위한 PaperBanana 활용 가이드입니다.

---

## 1. PaperBanana 개요

- **웹사이트**: https://paper-banana.org/ko
- **용도**: 방법론 다이어그램, 통계 차트, 시스템 아키텍처, 플로우 차트
- **특징**: 텍스트 설명 → 출판 수준 학술 일러스트 자동 생성
- **GitHub**: https://github.com/dwzhu-pku/PaperBanana (웹 데모, 코드 곧 공개 예정)

---

## 2. FPCB 논문용 추천 프롬프트

### 방법론 다이어그램
```
FPCB bending crack detection pipeline: 
1) Input: contour trajectory frames from bending process
2) Feature extraction: velocity, acceleration, curvature, strain surrogate per frame
3) Two anomaly detectors in parallel: DREAM (reconstruction + discriminator) and PatchCore (memory bank + k-NN)
4) Ensemble: logical AND - predict crack only when both models agree
5) Output: binary label (normal / crack)
```

### 시스템 아키텍처
```
Two-branch anomaly detection architecture:
Left branch: DREAM model with autoencoder reconstruction and discriminative head
Right branch: PatchCore with normal feature memory bank and distance-based scoring
Both branches feed into AND gate for final crack prediction
```

### 플로우 차트
```
Flow: Synthetic data generation (physics-informed) 
-> Feature extraction (per-frame + global stats) 
-> DREAM training on normal 
-> PatchCore memory bank 
-> Threshold selection on validation 
-> Ensemble evaluation on test set
```

---

## 3. Nano-Banana Pro (이미지 합성)

- **웹사이트**: https://nanobanana.org/ko
- **용도**: 참조 이미지 기반 편집, 4K 출력, 캐릭터/객체 일관성
- **활용**: 벡터맵/시각화 스타일 변환, 포스터/발표 자료용 고해상도 이미지

---

## 4. 로컬 도표 개선 (자동)

본 프로젝트의 `scripts/analyze_crack_detection.py`는 이미 다음과 같이 개선되었습니다:

- **Confusion matrix**: DPI 300, 가독성 좋은 색상(진한 셀에 흰 글자)
- **Insights summary**: DPI 300, 깔끔한 배경
- **재생성**: `python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_100k_v2`

---

## 5. 사용 순서

**웹 (수동):**
1. paper-banana.org에서 위 프롬프트 입력
2. 스타일 선택 (Academic, NeurIPS 등)
3. 생성된 이미지 다운로드
4. `reports/deliverables/figures/` 또는 논문 figures 폴더에 저장
5. Word 보고서에 삽입

**CLI/자동 (권장):** PyPI 패키지로 로컬 생성 가능. 설치·API 키·실행 방법은 [PAPER_WRITING_AND_PAPERBANANA_PLAN.md](PAPER_WRITING_AND_PAPERBANANA_PLAN.md) 참고.

```powershell
pip install paperbanana
paperbanana setup
paperbanana generate -i docs/paperbanana_inputs/fpcb_methodology.txt -c "FPCB crack detection pipeline" -o reports/deliverables/figures/fig_methodology.png
```
