# DREAM = DRAEM 레퍼런스

## 논문

- **제목**: DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection  
- **저자**: Vitjan Zavrtanik, Matej Kristan, Danijel Skočaj  
- **학회**: ICCV 2021, pp. 8330–8339  
- **arXiv**: [2108.07610](https://arxiv.org/abs/2108.07610)  
- **코드**: [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)

## 전략 요약

- **목표**: 정상 데이터만으로 학습하여 표면 이상(이 프로젝트에서는 FPCB 벤딩 구리배선 크랙)을 감지.
- **두 가지 구성요소**  
  1. **Reconstructive**: 이상이 포함된 입력을 넣었을 때 **정상(이상 제거) 복원**을 출력하도록 학습.  
  2. **Discriminative**: (입력, 복원)의 공동 표현 위에 **정상 vs 이상** 결정 경계를 학습.
- **학습 데이터**: **정상 이미지만** 사용. 이상은 **합성 시뮬레이션**(정상 + 합성 이상)으로 생성.
- **손실**: 재구성 손실(L2, SSIM 등) + 판별 손실(마스크/이상 여부에 대한 Focal/BCE).
- **추론**: 판별기 출력(및 필요 시 재구성 오차와 결합)을 이상 점수로 사용.

## 논문 보고 수치 (MVTec AD)

- 이미지 수준 ROC AUC: **98.1%**  
- 픽셀 수준 ROC AUC: 97.5%  
- 픽셀 수준 AP: 68.9%

(우리 도메인은 FPCB 테이블/시계열이므로 위 수치와 **직접 비교하지 않고**, 합성/실측 데이터로 정확도·리콜·AUC를 별도 보고.)

## 합성 데이터 검증

- **스크립트**: `scripts/validate_dream_synthetic.py`  
- **실행**: `pip install -e ".[ml]"` 후 `python scripts/validate_dream_synthetic.py`  
- **출력**: Accuracy, Precision, Recall, F1, ROC AUC.  
- **해석**: 합성 normal/crack에서 AUC가 논문 대비 높게 나오면(예: 0.9 이상) 정상/이상 구분이 잘 되는 것으로 해석. 논문 98.1%와는 도메인·데이터가 다르므로 수치 직접 비교는 하지 않는다.

## FPCB 탭ular 적용

- **입력**: 프레임/샘플별 특징 벡터(정상 데이터만으로 추출).
- **합성 이상**: 정상 샘플에 노이즈 또는 시나리오 기반 변형(예: crack 시나리오 특징)을 가해 “이상” 샘플 생성.
- **재구성**: AE는 정상만 또는 (합성 이상 입력 → 정상 타깃) 쌍으로 학습해 “정상 복원”을 학습.
- **판별**: (입력, 재구성)을 concat → MLP → P(이상). 정상 (x, x_recon) → 0, 합성 이상 (x_aug, x_recon) → 1로 학습.
- **추론**: DREAM 모드에서는 재구성 오차와 판별기 출력을 결합한 이상 점수 사용.
