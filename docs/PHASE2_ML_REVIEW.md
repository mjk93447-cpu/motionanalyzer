# Phase 2 ML 모델 리뷰: 레퍼런스 검증 및 벤치마크 비교

## 1. 목적

- DREAM, PatchCore 등 Phase 2 AI 모델의 **논문·벤치마크 조사**
- **구현이 최신 연구 성과를 올바르게 레퍼런스하는지 검증**
- **합성 데이터 기반 성능 측정** 후 문헌 벤치마크와 비교 (도메인 차이 명시)

---

## 2. PatchCore

### 2.1 논문 및 벤치마크

- **논문**: Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). **Towards Total Recall in Industrial Anomaly Detection.** *CVPR 2022*.
- **arXiv**: [2106.08265](https://arxiv.org/abs/2106.08265)
- **코드**: [amazon-research/patchcore-inspection](https://github.com/amazon-research/patchcore-inspection)

**방법 요약**:
- **입력**: ImageNet 사전학습 CNN(Wide ResNet 등)의 중간 특징 맵에서 **패치 수준** 특징 추출
- **메모리 뱅크**: 정상 이미지의 모든 패치 특징을 풀링한 뱅크 M
- **Coreset**: **Greedy** 서브샘플링 — 이미 선택된 벡터와 **거리가 최대인** 패치를 반복 선택 (K-center 근사), 뱅크 크기 1–10% 수준으로 유지
- **이상 점수**: 테스트 패치와 메모리 뱅크 **최근접 이웃(1-NN)** 의 유클리드 거리. 이미지 수준 점수는 **패치별 점수의 최댓값**
- **벤치마크**: MVTec AD **이미지 수준 AUROC 최대 99.6%**, 이전 SOTA 대비 오류 약 50% 감소

### 2.2 현재 구현과의 대응

| 항목 | 논문 (PatchCore) | 본 프로젝트 구현 |
|------|-------------------|------------------|
| 특징 | CNN 패치 특징 (ImageNet 백본) | **테이블 특징** (strain, curvature, crack_risk 등) |
| 메모리 뱅크 | 정상 패치 특징 풀 | 정상 샘플 특징 벡터 |
| Coreset | **Greedy** (최대 거리 순 선택) | **랜덤** 샘플링 (`coreset_size` 고정) |
| 거리 | 1-NN 유클리드 | **k-NN 평균** 유클리드 (기본 k=1이면 1-NN과 동일) |
| 이미지/샘플 수준 | max over patches | 샘플당 1점수 (이미 샘플 단위) |

**결론**:  
구현은 **PatchCore의 아이디어**(정상 메모리 뱅크 + NN 거리 기반 이상 점수)를 **테이블/시계열 도메인**에 적용한 **PatchCore-inspired** 버전이다.  
- **올바른 레퍼런스**: 논문을 인용하고, “테이블 특징 + 랜덤 coreset” 등 **차이점을 문서에 명시**하는 것이 적절함.  
- **추가 개선 권장**: Coreset을 greedy(최대 거리)로 바꾸면 논문과 더 가깝고, 일부 설정에서 성능 향상 가능.

---

## 3. DREAM = DRAEM (Zavrtanik et al.)

### 3.1 논문

- **DREAM**은 **DRAEM** 전략을 따름: Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). *DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection.* ICCV 2021. arXiv:2108.07610.
- **1저자**: Vitjan Zavrtanik. 코드: https://github.com/VitjanZ/DRAEM
- **전략**: 정상만 학습; 합성 이상(정상+노이즈 등)으로 (1) 재구성 서브넷(이상 입력 → 정상 복원), (2) 판별 서브넷(입력‖재구성 → P(이상)) 동시 학습. 추론 시 판별기 출력(및 재구성 오차)으로 이상 점수.

### 3.2 논문 보고 수치 (MVTec AD)

- 이미지 수준 ROC AUC: **98.1%** (리콜·정확도 추세 참고용).

### 3.3 현재 구현과의 대응

| 항목 | DRAEM 논문 | 본 프로젝트 (DREAMPyTorch) |
|------|------------|----------------------------|
| 재구성 | 이미지 → 정상 복원 | 테이블 AE: 입력 → 재구성 |
| 합성 이상 | 정상 이미지 + 패치/텍스처 | 정상 벡터 + Gaussian 노이즈 |
| 판별 | (재구성, 입력) → 마스크 | (입력‖재구성) → P(이상) |
| 학습 | L2 + SSIM + Focal(마스크) | MSE + λ·BCE(판별) |

**결론**: DREAM 모드는 DRAEM 논문 전략에 맞춰 **재구성 + 판별**을 구현했으며, 정상만 + 합성 이상으로 학습. 합성 검증은 `scripts/validate_dream_synthetic.py`로 정확도·리콜·AUC 확인. 소수 실제 이상 활용 전략은 `docs/DREAM_FEWSHOT_REAL_STRATEGY.md` 참고.

---

## 4. 레퍼런스 반영 상태

- **PatchCore**  
  - `patchcore.py` 상단에 Roth et al., CVPR 2022, arXiv:2106.08265 인용 있음.  
  - **권장**: 위와 같은 “구현 차이(테이블, 랜덤 coreset)” 요약을 docstring 또는 본 리뷰 문서에 유지.

- **DREAM**  
  - “Few-shot anomaly detection literature (to be cited)” 등 모호한 표현 있음.  
  - **권장**: “Reconstruction-based anomaly detection (e.g. autoencoder on normal data); for related discriminative approach see DRAEM (Zavrtanik et al., ICCV 2021)” 수준으로 구체화.

---

## 5. 합성 데이터 벤치마크 및 문헌과의 비교

### 5.1 비교 시 유의사항

- **데이터**: 문헌은 주로 **MVTec AD**(이미지, 산업 결함). 우리는 **FPCB 합성 시계열/테이블**.
- **도메인·특징이 다르므로 수치를 직접 비교하는 것은 부적절**하고, “같은 데이터셋/지표가 아님”을 전제로 **트렌드·상대 성능**만 참고.

### 5.2 기대 지표 (문헌)

- **PatchCore (MVTec AD)**: 이미지 AUROC 최대 **99.6%**.
- **AE 기반 (MVTec AD 등)**: 이미지 AUROC **~99%** 대 보고 사례 다수.

### 5.3 합성 데이터 벤치마크

- **스크립트**: `scripts/benchmark_phase2_ml.py`  
  - 합성 normal/crack 데이터 생성 → 특징 추출·정규화 → DREAM/PatchCore 학습 → AUC-ROC 계산.
- **실행** (ML 의존성 설치 후):
  ```bash
  pip install -e ".[ml]"
  python scripts/benchmark_phase2_ml.py
  ```
- **해석**:
  - 합성 데이터에서 **AUC-ROC가 0.7~0.95+** 구간이면, 정상/크랙이 구분 가능한 수준으로 동작하는 것으로 해석.
  - 문헌 수치(99% 등)와 **직접 비교하지 말고**, “우리 도메인·합성 설정에서의 상대적 성능 및 재현성” 확인용으로 사용.

- **예시 결과** (프레임별 합성 normal/crack 각 1세트, 동일 데이터 train=test):
  - PatchCore: AUC-ROC **1.00** (합성 normal/crack이 특징 공간에서 명확히 구분됨).
  - DREAM: PyTorch 설치 시 동일 설정에서 유사 구간 기대; 미설치 시 스크립트가 스킵.

---

## 6. 요약 및 권장 사항

| 항목 | 상태 | 권장 |
|------|------|------|
| PatchCore 논문 인용 | ✅ Roth et al. 명시됨 | 구현 차이(테이블, 랜덤 coreset) docstring/문서 유지 |
| DREAM 레퍼런스 | ⚠️ 모호 | AE 기반 문헌 + (선택) DRAEM 관련 문헌 구체화 |
| 합성 벤치마크 | 스크립트 제공 | 주기적 실행으로 회귀·성능 트렌드 확인 |
| Coreset | 랜덤 | (선택) Greedy coreset 도입 시 논문과의 정합성·성능 개선 여지 |

이 문서는 Phase 2 전체 리뷰의 일부로, 레퍼런스 검증과 합성 데이터 벤치마크 비교의 기준으로 사용할 수 있다.
