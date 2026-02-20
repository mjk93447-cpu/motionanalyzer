# FPCB 크랙 탐지 시스템 중장기 개발 계획 (합성 데이터 우선 전략)

**최종 업데이트**: 2026년 2월 19일  
**현재 진행률**: 약 85% (Phase 1, 2 완료, Phase 3.1 모듈 완료, 합성 데이터 고도화 완료, Phase A.1, A.2 완료, Phase B.1-B.5 완료, 배포 준비 완료)  
**핵심 전략**: 실제 크랙 데이터 확보 전까지 합성 데이터로 구현/검증 가능한 작업 우선 진행

### 🎯 프로젝트 최종 목표 (우선순위) — [docs/PROJECT_GOALS.md](PROJECT_GOALS.md)

| 우선순위 | 목표 | 특성 | 핵심 감지 대상 |
|----------|------|------|----------------|
| **1 (최우선)** | **벤딩 중 크랙 감지** | 시계열·국소적 | 크랙 직전/직후 속도 변화, 충격파, 진동, 미세 길이 변화(벌어짐) |
| **2** | **이미 크랙된 패널 감지** | 전체적 패턴 | 손상으로 인한 미묘한 물성·구조 차이, 다른 벤딩 궤적 |

**목표 1 원인**: 과경화, 과도한 벤딩 궤적, 너무 빠른 벤딩 속도  
**목표 2**: 손상 부위가 매우 작아 차이가 미세 → 다양한 AI 모델로 정교하게 구분

### 최근 완료 사항 (2026년 2월 17일)
- ✅ **합성 데이터 고도화**: 충격파(shockwave) 및 미세 진동(micro-vibration) 패턴 추가
  - 크랙 발생 시 가속도 스파이크: 3.24x 증가 확인
  - 진동 패턴: 표준편차 1.40x 증가 확인
  - 검증 스크립트: `scripts/validate_enhanced_synthetic.py`

## 📊 핵심 성과 (Phase 2 완료)

### 성능 개선
- **Recall**: 0.125 → 0.8125 (**+550% 향상**)
- **F1 Score**: 0.21 → 0.81 (**+286% 향상**)
- **ROC AUC**: 0.71 → 0.79 (+11%)
- **Accuracy**: 0.53 → 0.81 (+53%)

### 주요 기술적 성과
1. **크랙 유사 합성 이상 생성**: 물리 기반 패턴 (가속도 스파이크, 충격파, 궤적 편차) 구현
2. **Threshold 최적화**: Precision-Recall 곡선 기반 최적 threshold 탐색
3. **DRAEM 전략 구현**: 정상 데이터만으로 학습하는 이상 감지 모델 완성
4. **Feature Name 매칭 개선**: 유연한 특징 매칭으로 물리 기반 변형 적용

### 신규 핵심 인사이트 (개발 중 확인)
- **라벨 누설(Leakage) 위험**: `auto_optimize.extract_features()`는 기본적으로 `crack_risk`/`crack_risk_*`를 특징에 포함한다.
  - `crack_risk_*`는 Physics 모델의 산출물이므로, DREAM/PatchCore 같은 ML 입력에 포함하면 성능이 “비정상적으로” 좋아질 수 있다.
  - 따라서 **ML(Anomaly Detection) 검증은 `crack_risk_*`를 제외한 kinematic/geometry 특징으로 수행**해야 한다.
- **평가 메트릭 정책**: ROC AUC만으로는 운영 목적(precision-recall 균형)을 충분히 반영하기 어렵다.
  - 문헌(ADBench, time-series metric survey 등) 기준으로 **ROC AUC + PR AUC(AUCPR) 병행**을 기본 정책으로 한다.
- **Change Point Detection 파라미터 튜닝**:
  - CUSUM threshold는 false alarm rate와 detection delay의 균형이 중요 (문헌: 최대 1.5σ shift 감지에 효과적)
  - Window-based detector는 통계적 변화 감지에 효과적 (평균/분산 비율 기반)
  - PELT는 최적해 보장하지만 `ruptures` 라이브러리 필요
  - 시계열 특징 선택: `acceleration_max`가 크랙 발생 시점 감지에 가장 효과적 (합성 데이터 검증 결과)
- **정규화 누설 방지**:
  - 정규화 통계(mean/std)를 normal-only 데이터에서 계산하고 전체 데이터에 적용하여 crack 데이터가 통계에 영향을 주지 않도록 함
  - `normalize_features(..., fit_df=normal_df)` 패턴으로 명시적 fit/transform 분리
- **Temporal 모델 설계**:
  - 시퀀스 길이: 문헌 기준 최적값은 10 프레임 (autocorrelation 기반, 20+에서는 성능 저하)
  - 윈도우 구성: `dataset_path` + `frame` 기준 정렬로 시계열 순서 보장 필수
  - 점수 환산: 슬라이딩 윈도우에서 한 프레임이 여러 시퀀스에 포함되므로 max/mean aggregation 필요
- **고급 특징 엔지니어링**:
  - 고차 통계: 왜도(skewness)는 분포 비대칭성, 첨도(kurtosis)는 꼬리 무게 측정 (문헌: 분포 분석 기반 전처리 결정에 유용)
  - 자기상관: lag-1, lag-2 autocorrelation으로 시계열 패턴 의존성 측정
  - Temporal Features: 프레임 간 변화율(change_rate)과 변화 가속도(change_accel)로 급격한 변화 감지
  - 주파수 도메인: FFT 기반 dominant frequency와 spectral entropy로 주기적 패턴 및 진동 특성 추출
  - 특징 선택: 모든 고급 특징을 항상 포함하지 말고, ML 모델 성능에 따라 선택적으로 사용 (과적합 방지)

### 핵심 인사이트
- **측면 벤딩 카메라에서 직접 크랙 관찰 불가** → 시계열 패턴 분석 필수
- 크랙 발생 시점의 **미세한 궤적, 속도, 가속도 변화 및 미세진동(충격파) 패턴** 감지
- 물리 기반 합성 이상 생성이 단순 노이즈보다 훨씬 효과적
- **GUI 통합 전략**: 각 기능을 독립적인 탭으로 분리하여 사용자 경험 향상 (Analyze, Compare, Crack Model Tuning, ML & Optimization, Time Series Analysis)

---

## 중장기 계획 복습·정리 (요약) - 2026년 2월 업데이트

| Phase | 범위 | 상태 | 주요 산출물 | 성능 지표 |
|-------|------|------|-------------|-----------|
| **1.1** | Crack Model Tuning 탭 | 완료 | 파라미터 편집, 저장/로드 | - |
| **1.2** | 기본 설정·EXE 설정 포함 | 완료 | configs, 사용자 파라미터 자동 로드 | - |
| **2.1** | 데이터 준비 파이프라인 | 완료 | auto_optimize, ML & Optimization 탭 | - |
| **2.2** | DREAM 통합 | 완료 | ml_models.dream, 크랙 유사 합성 이상 생성 | Recall: 0.125→0.8125 (+550%) |
| **2.3** | PatchCore 통합 | 완료 | ml_models.patchcore, runners._run_patchcore | - |
| **2.4** | 파라미터 자동 최적화 | 완료 | optimizers (Grid Search, Bayesian), threshold 최적화 | F1: 0.21→0.81 (+286%) |
| **2.5** | 크랙 유사 합성 이상 생성 | 완료 | 물리 기반 패턴 (가속도 스파이크, 궤적 편차 등) | ROC AUC: 0.71→0.79 |
| **3.1** | Change Point Detection 모듈 | 완료 | changepoint.py (CUSUM, Window-based, PELT) | 합성 데이터 검증 완료 |
| **3.0** | 합성 데이터 고도화 | 완료 | 충격파/진동 패턴 추가 | 스파이크 3.24x, 진동 1.40x 확인 |
| **3** | 시계열 이상 감지 고도화 | 진행 중 | GUI 통합, temporal modeling, 앙상블 | - |
| **4** | EXE 통합·배포 | 일부 | build_exe, 모델 경로 통일 | - |
| **5** | 검증·문서화 | 지속 | 가이드, 검증 시나리오 | - |

**핵심 인사이트 (Phase 2 개발 중 획득):**
- **측면 벤딩 카메라에서 직접 크랙 관찰 불가** → 시계열 패턴 분석 필수
- **크랙 유사 합성 이상 생성**이 단순 노이즈보다 훨씬 효과적 (Recall 550% 향상)
- **Threshold 최적화**가 Precision-Recall 균형에 중요
- **물리 기반 패턴** (가속도 스파이크, 충격파, 궤적 편차)이 크랙 감지에 핵심

**진행 원칙:** 중요한 단계마다 모듈 테스트로 완성도 확인. 모델/최적화 코드는 `gui.runners`를 통해서만 GUI와 연결.

---

## 목표

Windows GUI 기반 EXE로 사용자가 실제 데이터에 맞게 튜닝 가능한 FPCB 구리 배선 크랙 예측 시스템 구축. 비정상(크랙) 데이터가 매우 적은 상황에서도 크랙 발생 여부를 예측하고, 벤딩 중 특정 시점의 미세한 패턴 차이를 감지.

---

## Phase 1: 파라미터 튜닝 GUI (단기, 2-3주)

### 1.1 GUI 확장: Crack Model 파라미터 편집 UI

**현재 상태:**
- `desktop_gui.py`: 기본 분석/비교 탭만 존재
- `crack_model.py`: `CrackModelParams`로 파라미터화 가능하나 GUI에서 조정 불가

**구현 내용:**
- **새 탭 "Crack Model Tuning"** 추가
  - `CrackModelParams` 필드별 슬라이더/입력 필드:
    - Caps: `strain_cap`, `curvature_concentration_cap`, `bend_angle_cap_deg`, `impact_cap_px_s2`
    - Weights: `w_strain`, `w_stress`, `w_curvature_concentration`, `w_bend_angle`, `w_impact`
    - Sigmoid: `sigmoid_steepness`, `sigmoid_center`
  - 실시간 미리보기: 파라미터 변경 시 선택된 데이터셋에 대한 `crack_risk` 재계산 및 히스토그램/시계열 플롯 업데이트
  - 저장/로드: JSON으로 파라미터 세트 저장/불러오기

**파일:**
- `src/motionanalyzer/desktop_gui.py`: 탭 추가, 파라미터 편집 위젯
- `src/motionanalyzer/crack_model.py`: 파라미터 검증/로드/저장 유틸리티 추가

**검증:**
- 정상/크랙 시나리오 합성 데이터로 파라미터 조정 → `max_crack_risk` 차이 확인

---

### 1.2 EXE 빌드에 파라미터 설정 포함

**구현 내용:**
- `CrackModelParams` 기본값을 `configs/crack_model_default.json`으로 분리
- EXE 실행 시 `%APPDATA%/motionanalyzer/crack_model_params.json`에서 사용자 설정 로드 (없으면 기본값)
- GUI에서 저장한 설정이 EXE 재시작 후에도 유지

**파일:**
- `configs/crack_model_default.json`: 기본 파라미터
- `src/motionanalyzer/config.py`: 설정 로드/저장 로직
- `scripts/build_exe.ps1`: configs 디렉토리 포함 확인

---

## Phase 2: 딥러닝 자동 최적화 (중기, 4-6주)

### 2.1 데이터 준비 및 전처리 파이프라인

**요구사항:**
- 사용자가 정상 데이터셋(폴더)과 크랙 데이터셋(폴더)을 GUI에서 선택
- 각 데이터셋은 `frame_*.txt` 묶음 + `frame_metrics.csv` (선택)
- 라벨: 정상=0, 크랙=1 (전역 또는 프레임별)

**구현 내용:**
- **새 탭 "Auto Optimization"**
  - 입력:
    - 정상 데이터셋 경로 (여러 세션 가능)
    - 크랙 데이터셋 경로 (여러 세션 가능)
    - 라벨링 방식: 전역(세션 전체) 또는 프레임별(시계열)
  - 전처리:
    - 각 세션에 대해 `run_analysis` 실행 → `vectors.csv` 생성
    - 특징 추출: `strain_surrogate`, `stress_surrogate`, `impact_surrogate`, `curvature_like`, `acceleration`, `bend_angle_deg`, `curvature_concentration` (프레임별 또는 전역 통계)
    - 정규화 및 시계열 윈도우 생성 (옵션)

**파일:**
- `src/motionanalyzer/auto_optimize.py`: 데이터 로더, 특징 추출기
- `src/motionanalyzer/desktop_gui.py`: Auto Optimization 탭

---

### 2.2 Few-Shot Anomaly Detection 모델 통합 ✅ 완료

**문헌 기반 접근:**

#### 2.2.1 DREAM (DRAEM 전략) ✅ 완료
- **핵심 아이디어:** 정상 데이터만으로 오토인코더 학습 → 재구성 오차가 높은 구간 = 비정상
- **적용:** FPCB 벤딩 시계열에서 정상 패턴 학습, 크랙 발생 시 재구성 오차 급증 감지
- **장점:** 크랙 샘플이 매우 적어도 정상 데이터만으로 학습 가능
- **구현 완료:**
  - 입력: 시계열 특징 벡터 (프레임별 또는 윈도우)
  - 아키텍처: MLP 기반 인코더-디코더 (DRAEM 전략)
  - 손실: MSE 재구성 오차 + 판별기 BCE 손실 (discriminative head)
  - **크랙 유사 합성 이상 생성**: 물리 기반 패턴 (가속도 스파이크, 충격파, 궤적 편차, 변형 집중, 미세 진동)
  - **Threshold 최적화**: Precision-Recall 곡선에서 최적점 탐색 (balanced, f1, precision, recall 옵션)
- **성능 (합성 데이터 검증)**:
  - Accuracy: 0.8125, Precision: 0.8125, Recall: 0.8125, F1: 0.8125, ROC AUC: 0.7891
  - 이전 대비 Recall 550% 향상, F1 286% 향상
- **참고 문헌**: Zavrtanik et al., "DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection", ICCV 2021

#### 2.2.2 PatchCore (Memory Bank 기반) ✅ 완료
- **핵심 아이디어:** 정상 특징의 메모리 뱅크 구축 → 테스트 시 가장 가까운 정상 패치와의 거리로 이상 점수 계산
- **적용:** FPCB 벤딩의 공간-시간 패치(예: 곡률 분포, 가속도 패턴)를 메모리 뱅크에 저장
- **장점:** 미세한 패턴 차이도 거리 기반으로 감지 가능
- **구현 완료:**
  - 특징 추출: 프레임별 또는 시공간 패치별 특징 벡터
  - 메모리 뱅크: 정상 데이터의 대표 특징 벡터들 (Random Coreset 선택)
  - 이상 점수: 테스트 특징과 메모리 뱅크 k-NN 평균 거리
  - Scikit-learn 기반 구현 (NearestNeighbors)
- **참고 문헌**: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022

#### 2.2.3 하이브리드 접근 (Phase 3에서 구현 예정)
- **1단계:** DREAM으로 전역 이상 감지 (시계열 전체)
- **2단계:** PatchCore로 국소 패턴 차이 감지 (특정 프레임/인덱스)
- **결합:** 두 점수의 가중 평균 또는 최대값
- **예상 효과**: 단일 모델 대비 성능 향상 (앙상블 효과)

**파일:**
- `src/motionanalyzer/ml_models/dream.py`: DREAM 모델 구현
- `src/motionanalyzer/ml_models/patchcore.py`: PatchCore 구현
- `src/motionanalyzer/ml_models/hybrid.py`: 하이브리드 통합
- `requirements-ml.txt`: PyTorch, scikit-learn 등 ML 의존성

**의존성:**
```txt
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

---

### 2.3 파라미터 자동 최적화

**목표:** 정상/크랙 데이터로 `CrackModelParams` 자동 튜닝

**방법 1: 그리드 서치 + 검증 지표**
- `CrackModelParams` 공간에서 그리드 서치
- 각 파라미터 조합에 대해 정상/크랙 데이터의 `crack_risk` 분포 계산
- 지표: AUC-ROC, F1-score (크랙=1, 정상=0)
- 최적 파라미터: 검증 지표 최대화

**방법 2: 베이지안 최적화**
- `optuna` 또는 `scikit-optimize` 사용
- 목적 함수: 검증 지표 (AUC-ROC)
- 제약: 파라미터 범위 (예: weights 합=1, caps > 0)

**방법 3: 딥러닝 메타 학습**
- 작은 신경망이 `CrackModelParams`를 출력하도록 학습
- 입력: 정상/크랙 데이터의 통계 특징
- 출력: 최적 파라미터
- 손실: 검증 지표 (AUC-ROC)의 음수

**구현:**
- `src/motionanalyzer/auto_optimize.py`: 최적화 엔진
- GUI에서 "Start Optimization" 버튼 → 백그라운드 실행 → 진행률 표시 → 결과 표시

**파일:**
- `src/motionanalyzer/optimizers/grid_search.py`
- `src/motionanalyzer/optimizers/bayesian.py`
- `src/motionanalyzer/optimizers/meta_learning.py`

---

## Phase 3: 시계열 이상 감지 고도화 (합성 데이터 우선 전략)

### 3.0 핵심 인사이트 기반 전략 수정

**Phase 2 개발 중 획득한 핵심 인사이트:**
- 측면 벤딩 카메라에서 직접 크랙 관찰 불가 → **시계열 패턴 분석 필수**
- 크랙 발생 시점의 **미세한 궤적, 속도, 가속도 변화 및 미세진동(충격파) 패턴** 감지
- 물리 기반 합성 이상 생성이 효과적 → 실제 크랙 데이터로 fine-tuning 필요

**합성 데이터 우선 전략:**
- **실제 크랙 데이터 확보 전까지 합성 데이터로 구현/검증 가능한 작업 우선**
- 정교한 물리 기반 합성 데이터 생성으로 현실성 향상
- 단계별 개발: 합성 데이터 생성 개선 → 테스트 → 기능 개발 → 검증

**전략 조정:**
- **Phase A**: 합성 데이터로 검증 가능한 기능 완성 (EXE, GUI 통합, 앙상블, Temporal Modeling)
- **Phase B**: 합성 데이터 고도화 (충격파, 진동 패턴 추가)
- **Phase C**: 실제 데이터 확보 후 Few-shot Fine-tuning 진행

### 3.1 벤딩 중 특정 시점 크랙 감지 (우선순위: 높음)

**요구사항:** 벤딩 진행 중 어느 프레임에서 크랙이 발생했는지 정확히 감지

**접근:**

#### 3.1.1 Change Point Detection (우선 구현)
- **CUSUM (Cumulative Sum)**: 누적 편차가 임계값 초과 시 변화점 감지
  - 적용: `acceleration`, `curvature_concentration`, `strain_surrogate` 시계열
  - 크랙 발생 시 급격한 변화 감지 (충격파 패턴)
- **PELT (Pruned Exact Linear Time)**: 동적 프로그래밍으로 최적 분할점 찾기
  - `ruptures` 라이브러리 활용
  - 비용 함수: L2 norm 또는 custom cost (크랙 시그니처 기반)
- **Window-based Detection**: 슬라이딩 윈도우로 통계적 변화 감지
  - 평균/분산 변화, 이상치 비율 급증

**구현 파일:**
- `src/motionanalyzer/time_series/changepoint.py`: CUSUM, PELT, Window-based
- GUI 통합: "Time Series Analysis" 탭에 Change Point Detection 모드 추가

#### 3.1.2 Temporal Dependency 모델링 (DREAM 확장)
- **LSTM/GRU 기반 시계열 오토인코더**: 프레임 간 연속성 모델링
  - 현재 MLP 기반 DREAM을 시계열 인코더로 확장
  - 입력: 시퀀스 (T frames) → 출력: 재구성 시퀀스
- **Transformer 기반 시계열 인코더**: Attention으로 장거리 의존성 포착
  - Self-attention으로 크랙 발생 전후 패턴 변화 감지
- **Temporal Contrastive Learning**: 정상 시퀀스는 유사하게, 이상 시퀀스는 다르게 학습

**구현 파일:**
- `src/motionanalyzer/ml_models/dream_temporal.py`: 시계열 확장 DREAM
- `src/motionanalyzer/ml_models/transformer_anomaly.py`: Transformer 기반 이상 감지

#### 3.1.3 Few-shot Fine-tuning with Real Crack Data (Phase C - 실제 데이터 확보 후)
- **소수 실제 크랙 데이터 활용 전략** (참고: `docs/DREAM_FEWSHOT_REAL_STRATEGY.md`)
- **옵션 A - 판별기만 Fine-tuning**:
  - 재구성 서브넷 고정, 판별기만 실제 크랙 데이터로 추가 학습
  - 작은 learning rate (1e-4), 적은 epoch (10-20), 조기 종료
- **옵션 B - 재구성 + 판별 동시 Fine-tuning**:
  - 실제 크랙 입력에 대해 정상으로 복원하도록 재구성기도 학습
  - 과적합 주의 (L2 정규화, Dropout 유지)
- **옵션 C - Prototype-based Calibration**:
  - 실제 크랙 임베딩과의 유사도로 이상 점수 보정

**구현 파일:**
- `src/motionanalyzer/ml_models/dream.py`: `fit_fewshot_anomaly()` 메서드 추가
- 검증: 소수 실제 크랙 도입 전·후 메트릭 비교 (Accuracy, Precision, Recall, F1, ROC AUC)
- **상태**: 실제 크랙 데이터 확보 전까지 대기 ⚠️

**파일:**
- `src/motionanalyzer/time_series/changepoint.py`: CUSUM, PELT
- `src/motionanalyzer/time_series/classifier.py`: 프레임별 분류기
- `src/motionanalyzer/time_series/attention.py`: Attention 기반 감지

---

### 3.2 앙상블 및 미세 패턴 차이 감지

**요구사항:** DREAM과 PatchCore의 강점을 결합하여 성능 향상

**접근:**

#### 3.2.1 DREAM + PatchCore 앙상블 (우선 구현)
- **전략 1 - 가중 평균**: `score_ensemble = α * score_dream + (1-α) * score_patchcore`
  - α는 검증 세트에서 최적화 (예: 0.6-0.7)
- **전략 2 - 최대값**: `score_ensemble = max(score_dream, score_patchcore)`
  - 둘 중 하나라도 이상 감지 시 이상으로 판단 (Recall 향상)
- **전략 3 - 스태킹**: DREAM과 PatchCore 점수를 입력으로 하는 메타 분류기
  - 작은 MLP로 두 점수를 결합하여 최종 판단

**구현 파일:**
- `src/motionanalyzer/ml_models/hybrid.py`: 앙상블 전략 구현
- GUI 통합: "ML & Optimization" 탭에 "Ensemble (DREAM + PatchCore)" 모드 추가

#### 3.2.2 Contrastive Learning (선택)
- 정상-크랙 쌍을 멀리, 정상-정상 쌍을 가깝게 학습
- 특징 공간에서 크랙이 정상과 구분되도록 학습
- Few-shot: 크랙 샘플이 적어도 contrastive loss로 효과적 학습 가능

#### 3.2.3 Advanced Feature Engineering
- **고차 통계**: 왜도, 첨도, 자기상관
- **주파수 도메인**: FFT, 웨이블릿 변환 (충격파 주파수 분석)
- **공간 패턴**: 곡률 분포의 히스토그램, 가속도 벡터장의 발산/회전
- **Temporal Features**: 프레임 간 변화율, 가속도 변화율, 궤적 곡률 변화율

**구현 파일:**
- `src/motionanalyzer/feature_engineering/advanced_features.py`: 고급 특징 추출

**파일:**
- `src/motionanalyzer/feature_engineering/contrastive.py`
- `src/motionanalyzer/feature_engineering/metric_learning.py`
- `src/motionanalyzer/feature_engineering/advanced_features.py`

---

## Phase 4: EXE 통합 및 배포 (단기, 1-2주)

### 4.1 EXE 빌드 스크립트 업데이트

**요구사항:**
- ML 모델 가중치 포함
- 사용자 설정 디렉토리 (`%APPDATA%/motionanalyzer/`) 지원
- 모델 파일 크기 최적화 (양자화, ONNX 변환 고려)

**구현:**
- `scripts/build_exe.ps1`: ML 의존성 포함, 모델 가중치 번들링
- `src/motionanalyzer/ml_models/__init__.py`: 모델 로더 (ONNX 또는 PyTorch)
- 사용자 설정: GUI에서 저장한 파라미터가 EXE 재시작 후에도 유지

---

### 4.2 GUI 통합

**최종 GUI 구조:**
1. **Analyze 탭**: 기존 분석 기능
2. **Compare 탭**: 기존 비교 기능
3. **Crack Model Tuning 탭**: 파라미터 편집 (Phase 1)
4. **Auto Optimization 탭**: 딥러닝 자동 최적화 (Phase 2)
5. **Time Series Analysis 탭**: 시계열 이상 감지 (Phase 3)

**파일:**
- `src/motionanalyzer/desktop_gui.py`: 모든 탭 통합
- `src/motionanalyzer/gui_components/`: 재사용 가능한 위젯 모듈화

---

## Phase 5: 검증 및 문서화 (지속)

### 5.1 검증 시나리오

1. **합성 데이터 검증:**
   - 정상/크랙 시나리오로 파라미터 튜닝 → `max_crack_risk` 차이 확인
   - DREAM/PatchCore가 크랙 시나리오를 이상으로 감지하는지 확인

2. **실제 데이터 검증 (사내망):**
   - 알려진 크랙 케이스로 모델 검증
   - False Positive/Negative 비율 측정
   - 사용자 피드백 반영

### 5.2 문서화

- `docs/CRACK_MODEL_TUNING_GUIDE.md`: 파라미터 튜닝 가이드
- `docs/AUTO_OPTIMIZATION_GUIDE.md`: 자동 최적화 사용법
- `docs/ML_MODELS.md`: DREAM, PatchCore 모델 설명
- `docs/TIMESERIES_DETECTION.md`: 시계열 이상 감지 방법론

---

## 기술 스택 요약

### 현재
- Python 3.12+
- tkinter (GUI)
- PyInstaller (EXE 빌드)
- NumPy, Pandas, Matplotlib

### 추가 필요
- **ML 프레임워크:** PyTorch 2.0+
- **최적화:** Optuna 또는 scikit-optimize
- **시계열:** tslearn, ruptures (change point)
- **특징 추출:** scikit-learn, scipy

### EXE 크기 고려
- PyTorch 포함 시 EXE 크기 증가 → CPU-only 빌드 또는 ONNX Runtime 사용 고려
- 모델 가중치: 양자화 또는 압축

---

## 일정 요약 (2026년 2월 업데이트)

| Phase | 기간 | 주요 산출물 | 상태 |
|-------|------|------------|------|
| Phase 1 | 2-3주 | 파라미터 튜닝 GUI | ✅ 완료 |
| Phase 2 | 4-6주 | 딥러닝 자동 최적화 (DREAM, PatchCore, 크랙 유사 합성 이상) | ✅ 완료 |
| Phase 3 | 6-8주 | 시계열 이상 감지 고도화 (Change Point, Few-shot Fine-tuning, 앙상블) | 🔄 진행 중 (3.1 완료) |
| Phase 4 | 1-2주 | EXE 통합 및 배포 | ✅ 부분 완료 (A.1 완료) |
| Phase 5 | 지속 | 검증 및 문서화 | 🔄 진행 중 |

**총 예상 기간:** 13-19주 (약 3-5개월)  
**현재 진행률:** 약 52% (Phase 1, 2 완료, Phase 3.1 모듈 완료, Phase A.1 완료)

**Phase 3 세부 일정:**
- 3.1 Change Point Detection: 2주
- 3.2 Few-shot Fine-tuning: 2주
- 3.3 앙상블: 1주
- 3.4 Temporal Modeling: 2-3주

---

## 참고 문헌

### Phase 2 구현 완료 (참고 문서 포함)

1. **DRAEM (DREAM 구현 기반)**
   - Zavrtanik, V., Kristan, M., & Skočaj, D. (2021). "DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection". ICCV 2021, pp. 8330-8339.
   - arXiv: [2108.07610](https://arxiv.org/abs/2108.07610)
   - Code: [VitjanZ/DRAEM](https://github.com/VitjanZ/DRAEM)
   - **참고 문서**: `docs/DREAM_DRAEM_REFERENCE.md`, `docs/DREAM_CRACK_LIKE_ANOMALY.md`

2. **PatchCore**
   - Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection". CVPR 2022.
   - Memory bank 기반 few-shot anomaly detection
   - Coreset selection for efficient memory usage

3. **Few-shot Learning for Anomaly Detection**
   - **참고 문서**: `docs/DREAM_FEWSHOT_REAL_STRATEGY.md`
   - 소수 실제 크랙 데이터 활용 전략

### Phase 3 구현 예정 (추가 조사 필요)

4. **Change Point Detection**
   - **CUSUM**: Page, E. S. (1954). "Continuous Inspection Schemes". Biometrika, 41(1/2), 100-115.
   - **PELT**: Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of changepoints with a linear computational cost". Journal of the American Statistical Association, 107(500), 1590-1598.
   - **라이브러리**: `ruptures` (Python) - PELT 구현 포함
   - 시계열 분할 및 변화점 감지

5. **Temporal Modeling for Anomaly Detection**
   - LSTM/GRU 기반 시계열 오토인코더
   - Transformer 기반 시계열 이상 감지
   - Temporal Contrastive Learning

6. **Ensemble Methods**
   - 앙상블 전략 (가중 평균, 스태킹)
   - 모델 다양성과 성능 향상

7. **Contrastive Learning for Anomaly Detection** (선택)
   - SimCLR, MoCo 등 self-supervised learning
   - Few-shot learning과의 결합

---

## 현재 구현 상태 및 구체적 다음 단계

### 완료된 항목
- [x] Phase 1.1: Crack Model Tuning 탭, 파라미터 저장/로드
- [x] Phase 1.2: configs/crack_model_default.json, run_analysis 사용자 파라미터 자동 로드
- [x] Phase 2.1: auto_optimize.py 데이터 준비, GUI "ML & Optimization" 탭
- [x] Phase 2.2: DREAM PyTorch 구현 (DRAEM 전략), 크랙 유사 합성 이상 생성, Threshold 최적화
- [x] Phase 2.3: PatchCore Scikit-learn 구현 (fit/predict/save/load), `_run_patchcore` 연동
- [x] Phase 2.4: Grid Search / Bayesian (optimizers), runners·GUI 연동, 최적 파라미터 사용자 설정 저장
- [x] Phase 2.5: 크랙 유사 합성 이상 생성 (물리 기반 패턴), Feature name 매칭 개선
- [x] GUI 아키텍처: `docs/GUI_ARCHITECTURE.md`, 모드별 러너 분리 (`gui.runners`)
- [x] 문서화: `docs/DREAM_DRAEM_REFERENCE.md`, `docs/DREAM_FEWSHOT_REAL_STRATEGY.md`, `docs/DREAM_CRACK_LIKE_ANOMALY.md`
- [x] Phase 3.1: Change Point Detection 모듈 구현 (`time_series/changepoint.py`), 단위 테스트 및 합성 데이터 검증 완료
- [x] Phase A.1.1: EXE 빌드 스크립트 ML 포함/미포함 옵션화 (`build_exe.ps1 -IncludeML`), 모델 저장 경로 통일 (`paths.py`)
- [x] Phase A.1.2: Analyze 탭 모드 확장 (Physics/DREAM/PatchCore), 모델 로드/추론, 이상 점수 시각화
- [x] Phase A.2: Change Point Detection GUI 통합 (Time Series Analysis 탭, CUSUM/Window-based/PELT, 시계열 특징 선택, 파라미터 조정, 변화점 시각화)
- [x] Phase B.2: 앙상블 구현 (`ml_models/hybrid.py`, 가중 평균/최대값/스태킹 전략, GUI 통합, 가중치 최적화)
- [x] Phase B.3: Temporal Modeling 구현 (`ml_models/dream_temporal.py`, LSTM/GRU autoencoder, 시퀀스 기반 재구성 오차, GUI 통합)
- [x] Phase B.4: 고급 특징 엔지니어링 구현 (고차 통계, Temporal Features, 주파수 도메인 FFT, GUI 옵션 추가)
- [x] 정규화 누설 방지 개선: `normalize_features`에 `fit_df` 파라미터 추가, normal-only fit으로 통계 계산
- [x] 합성 데이터 QA 게이트: `evaluate_synthetic_dataset_quality.py`, `validate_enhanced_dream.py` (라벨 누설 방지, PR AUC 포함)

### 모드별 코드 분리 (완료)
- **Physics**: Crack Model Tuning 탭에서 파라미터 편집; Analyze 탭에서 run_analysis(crack_params 자동 로드)
- **DREAM**: `gui.runners._run_dream` → `ml_models.dream.DREAMAnomalyDetector` (학습/저장/평가)
- **PatchCore**: `gui.runners._run_patchcore` → `ml_models.patchcore.PatchCoreScikitLearn` (학습/저장/평가 구현 완료)
- **Grid Search / Bayesian**: `gui.runners._run_grid_search`, `_run_bayesian` → `optimizers.grid_search`, `optimizers.bayesian` (경로 전달 시 학습·저장)

### 구체적 다음 단계 (우선순위 순)

#### Phase 3.1: Change Point Detection ✅ 완료
1. **Change Point Detection 모듈 구현** ✅ 완료
   - [x] `src/motionanalyzer/time_series/changepoint.py`: CUSUM, Window-based, PELT 구현
   - [x] 단위 테스트 작성 및 통과 (11 passed, 1 skipped)
   - [x] 합성 데이터 검증 스크립트 (`scripts/validate_changepoint_synthetic.py`)
   - [x] 문서화 (`docs/CHANGEPOINT_DETECTION.md`)
   - [x] GUI "Time Series Analysis" 탭에 Change Point Detection 모드 추가 (`desktop_gui._build_timeseries_tab`)

#### Phase 3.2: Few-shot Fine-tuning (우선순위: 높음)
2. **소수 실제 크랙 데이터 활용**
   - [ ] `ml_models/dream.py`: `fit_fewshot_anomaly()` 메서드 추가
   - [ ] 옵션 A (판별기만 fine-tuning) 구현
   - [ ] 옵션 B (재구성 + 판별 fine-tuning) 구현
   - [ ] 검증 스크립트: 소수 실제 크랙 도입 전·후 메트릭 비교
   - [ ] 문서화: 실제 크랙 데이터 수집·라벨링 가이드

#### Phase 3.3: 앙상블 ✅ (우선순위: 중간) - 완료
3. **DREAM + PatchCore 앙상블** ✅
   - [x] `ml_models/hybrid.py`: 앙상블 전략 구현 (가중 평균, 최대값, 스태킹)
   - [x] GUI "ML & Optimization" 탭에 "Ensemble" 모드 추가
   - [x] Analyze 탭에 "ensemble" 모드 추가
   - [x] 앙상블 가중치 최적화 (검증 세트 기준, `optimize_weights()`)
   - [ ] 성능 비교: 단일 모델 vs 앙상블 (검증 단계)

#### Phase 3.4: Temporal Modeling ✅ (우선순위: 중간) - 완료
4. **시계열 의존성 모델링** ✅
   - [x] `ml_models/dream_temporal.py`: LSTM/GRU 기반 시계열 오토인코더
   - [x] 슬라이딩 윈도우 시퀀스 구성 및 재구성 오차 기반 점수 계산
   - [x] GUI "ML & Optimization" 탭에 "Temporal (LSTM/GRU)" 모드 추가
   - [x] 검증 스크립트 (`scripts/validate_temporal_synthetic.py`) - ROC AUC + PR AUC
   - [ ] `ml_models/transformer_anomaly.py`: Transformer 기반 이상 감지 (선택적 향후 작업)
   - [ ] Temporal Contrastive Learning 구현 (선택적 향후 작업)
   - [ ] 성능 비교: MLP DREAM vs Temporal DREAM (검증 단계)

#### Phase 4: EXE 및 GUI 통합 ✅ (부분 완료)
5. **Analyze 탭 분석 모드 확장** ✅
   - [x] 콤보 "Analysis mode: Physics | DREAM | PatchCore"
   - [x] DREAM/PatchCore 선택 시 저장된 모델 로드 후 predict
   - [x] 이상 점수 시각화 (히스토그램, 시계열 플롯)
   - [x] 모델 없을 경우 안내 메시지 (graceful error handling)
6. **EXE 완성도** ✅
   - [x] `scripts/build_exe.ps1`에 ML 포함/미포함 옵션 추가 (`-IncludeML`)
   - [x] 모델 저장 경로 `%APPDATA%/motionanalyzer/models/` 통일 (`paths.py`)
   - [x] 기본 빌드: 경량 EXE (torch 제외), ML 포함 빌드: `-IncludeML` 옵션  

---

## 다음 단계 (합성 데이터 우선 전략)

**상세 우선순위 리스트**: 
- `docs/DEVELOPMENT_PRIORITIES.md`: 전체 우선순위 (이전 버전)
- **`docs/DEVELOPMENT_PRIORITIES_SYNTHETIC_FIRST.md`**: 합성 데이터 우선 전략 (현재 권장) ⭐
- **`docs/SYNTHETIC_DATA_ENHANCEMENT_PLAN.md`**: 합성 데이터 고도화 계획 ⭐

**핵심 전략**: 실제 크랙 데이터 확보 전까지 합성 데이터로 구현/검증 가능한 작업 우선 진행

### 개발 순서 (합성 데이터 우선)

#### Phase A: 즉시 사용 가능한 기능 (2-3주)
1. **EXE 빌드 완성** (3-5일) - 합성 데이터로 검증 ✅
2. **Analyze 탭 확장** (5-7일) - 합성 데이터로 검증 ✅
3. **Change Point Detection GUI 통합** (5-7일) - 합성 데이터로 검증 ✅

#### Phase B: 합성 데이터 고도화 및 기능 강화 (4-5주)
4. **합성 데이터 물리 현상 추가** (1주)
   - 충격파 패턴, 미세 진동 패턴 추가
   - 시계열 현실성 향상
   - 크랙 패턴 다양화
5. **앙상블 구현** (5-7일) - 개선된 합성 데이터로 검증 ✅
6. **Temporal Modeling** (10-14일) - 개선된 합성 데이터로 검증 ✅
7. **고급 특징 엔지니어링** (7-10일) - 개선된 합성 데이터로 검증 ✅

#### Phase C: 실제 데이터 확보 후 (대기)
8. **Few-shot Fine-tuning** - 실제 크랙 데이터 필요 ⚠️
9. **Contrastive Learning** - 실제 크랙 데이터 필요 ⚠️

### 즉시 시작 가능한 작업 (우선순위 1)

1. **EXE 빌드 완성** (우선순위 1.1)
   - `scripts/build_exe.ps1`에 ML 모델 의존성 포함 확인
   - 모델 저장 경로 통일 (`%APPDATA%/motionanalyzer/models/`)
   - PyTorch CPU-only 빌드 또는 ONNX Runtime 사용 고려
   - **예상 시간**: 3-5일

2. **Analyze 탭 분석 모드 확장** (우선순위 1.1)
   - 콤보 "Analysis mode: Physics | DREAM | PatchCore"
   - 저장된 모델 로드 후 predict 및 시각화
   - **예상 시간**: 5-7일

3. **Change Point Detection GUI 통합** (우선순위 1.2)
   - Time Series Analysis 탭 추가
   - CUSUM, Window-based, PELT 모드 선택
   - 변화점 시각화 및 결과 저장
   - **예상 시간**: 5-7일

### 다음 단계 (우선순위 2)

4. **Few-shot Fine-tuning 구현** (우선순위 2.1)
   - `fit_fewshot_anomaly()` 메서드 추가
   - 실제 크랙 데이터로 성능 향상 검증
   - **예상 시간**: 7-10일

5. **앙상블 구현** (우선순위 2.2)
   - DREAM + PatchCore 결합
   - GUI에 Ensemble 모드 추가
   - **예상 시간**: 5-7일

### 문서화 (지속)

- `docs/ML_MODELS.md`: DREAM, PatchCore 상세 설명
- `docs/TIMESERIES_DETECTION.md`: Change Point Detection 방법론
- `docs/ENSEMBLE_STRATEGY.md`: 앙상블 전략 및 성능 비교
- `docs/DEVELOPMENT_PRIORITIES.md`: 개발 우선순위 상세 리스트
