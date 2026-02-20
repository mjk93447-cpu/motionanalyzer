# 개발 우선순위 리스트 (합성 데이터 우선 전략)

**최종 업데이트**: 2026년 2월 18일  
**현재 진행률**: 약 45% (Phase 1, 2 완료, Phase 3.1 모듈 완료)  
**전략**: 실제 크랙 데이터 확보 전까지 합성 데이터로 구현/검증 가능한 작업 우선

## 합성 데이터 현황

현재 시스템은 다음 합성 데이터 생성 기능을 보유:
- **시나리오**: `normal`, `crack`, `pre_damage`, `thick_panel`, `uv_overcured`
- **물리 기반 모델**: 현실적인 벤딩 패턴 생성
- **검증 스크립트**: `scripts/validate_dream_synthetic.py`, `scripts/validate_changepoint_synthetic.py`
- **특징 추출**: `auto_optimize.py`로 프레임별/전역 특징 추출 가능

### 합성 데이터 “사전 품질 평가(QA 게이트)” 정책 (중요 단계 시작 전 필수)

- **목표**: 각 개발 단계에서 사용하는 합성 데이터가 precision-recall 검증에 적합한지 사전에 평가
- **실행 스크립트**:
  - `scripts/evaluate_synthetic_dataset_quality.py`: 밸런스/분리도/시계열 시그니처(충격파/진동) 체크
  - `scripts/validate_enhanced_dream.py`: ROC AUC + PR AUC(AUCPR) 포함 DREAM 정밀 검증
- **라벨 누설 방지(필수)**:
  - DREAM/PatchCore/Ensemble 등 ML 검증에서는 **Physics 산출물인 `crack_risk`, `crack_risk_*`를 특징에서 제외**한다.
  - `crack_risk_*` 포함 시 모델 성능이 과대평가될 수 있으므로(순환/누설) “통과”로 인정하지 않는다.

**합성 데이터로 검증 가능한 작업**: ✅  
**실제 크랙 데이터 필요 작업**: ⚠️

---

## 🔴 우선순위 1: 즉시 사용 가능한 기능 완성 (2-3주)

### 1.1 EXE 배포 준비 완료
**목표**: 사용자가 실제 데이터로 테스트할 수 있도록 EXE 완성  
**검증 방법**: 합성 데이터로 기본 기능 동작 확인 ✅

- [ ] **EXE 빌드 스크립트 완성**
  - [ ] `scripts/build_exe.ps1`에 ML 모델 의존성 포함 확인
  - [ ] 모델 저장 경로 `%APPDATA%/motionanalyzer/models/` 통일
  - [ ] PyTorch CPU-only 빌드 또는 ONNX Runtime 사용 고려 (EXE 크기 최적화)
  - [ ] 사용자 설정 디렉토리 (`%APPDATA%/motionanalyzer/`) 지원 확인
  - **예상 시간**: 3-5일
  - **의존성**: 없음
  - **검증**: EXE 빌드 성공, 합성 데이터로 기본 기능 동작 확인

- [ ] **Analyze 탭 분석 모드 확장**
  - [ ] 콤보 "Analysis mode: Physics | DREAM | PatchCore"
  - [ ] DREAM/PatchCore 선택 시 저장된 모델 로드 후 predict
  - [ ] 이상 점수 시각화 (히스토그램, 시계열 플롯)
  - [ ] 모델이 없을 경우 안내 메시지 및 학습 유도
  - **예상 시간**: 5-7일
  - **의존성**: Phase 2 완료
  - **검증**: 합성 데이터로 각 모드 분석 실행 및 결과 확인

### 1.2 Change Point Detection GUI 통합
**목표**: 크랙 발생 시점 감지 기능을 GUI에서 사용 가능하게  
**검증 방법**: 합성 데이터로 크랙 발생 시점 정확도 확인 ✅

- [ ] **Time Series Analysis 탭 추가**
  - [ ] Change Point Detection 모드 선택 (CUSUM, Window-based, PELT)
  - [ ] 시계열 특징 선택 (acceleration_max, curvature_concentration 등)
  - [ ] 파라미터 조정 UI (threshold, window_size 등)
  - [ ] 변화점 시각화 (시계열 플롯에 변화점 표시)
  - [ ] 결과 저장 및 리포트 생성
  - **예상 시간**: 5-7일
  - **의존성**: Phase 3.1 Change Point Detection 모듈 완료 ✅
  - **검증**: 합성 crack 시나리오로 크랙 발생 시점 정확도 확인

---

## 🟠 우선순위 2: 합성 데이터로 검증 가능한 기능 강화 (3-4주)

### 2.1 앙상블 구현
**목표**: DREAM과 PatchCore 결합으로 성능 향상  
**검증 방법**: 합성 데이터로 단일 모델 대비 성능 향상 확인 ✅

- [ ] **앙상블 모듈 구현**
  - [ ] `ml_models/hybrid.py`: 앙상블 전략 구현
  - [ ] 가중 평균 전략 (α 최적화)
  - [ ] 최대값 전략 (Recall 향상)
  - [ ] 스태킹 전략 (메타 분류기)
  - [ ] GUI "ML & Optimization" 탭에 "Ensemble" 모드 추가
  - [ ] 합성 데이터로 앙상블 가중치 최적화
  - **예상 시간**: 5-7일
  - **의존성**: Phase 2.2, 2.3 완료 ✅
  - **검증**: 합성 normal/crack 데이터로 단일 모델 대비 성능 향상 확인

### 2.2 Temporal Modeling
**목표**: 프레임 간 의존성 모델링으로 시계열 패턴 강화  
**검증 방법**: 합성 데이터로 MLP DREAM 대비 성능 향상 확인 ✅

- [ ] **시계열 DREAM 확장**
  - [ ] `ml_models/dream_temporal.py`: LSTM/GRU 기반 시계열 오토인코더
  - [ ] 입력: 시퀀스 (T frames) → 출력: 재구성 시퀀스
  - [ ] Temporal Contrastive Learning 구현 (합성 데이터로 학습)
  - [ ] 성능 비교: MLP DREAM vs Temporal DREAM
  - **예상 시간**: 10-14일
  - **의존성**: Phase 2.2 완료 ✅
  - **검증**: 합성 시계열 데이터에서 성능 향상 확인

- [ ] **Transformer 기반 이상 감지** (선택)
  - [ ] `ml_models/transformer_anomaly.py`: Transformer 기반 시계열 인코더
  - [ ] Self-attention으로 장거리 의존성 포착
  - [ ] 크랙 발생 전후 패턴 변화 감지
  - **예상 시간**: 10-14일
  - **의존성**: Phase 2.2 완료 ✅
  - **검증**: Temporal DREAM 대비 성능 비교 (합성 데이터)

### 2.3 고급 특징 엔지니어링
**목표**: 미세 패턴 차이 감지를 위한 특징 강화  
**검증 방법**: 합성 데이터로 특징 중요도 분석 및 성능 향상 확인 ✅

- [ ] **고급 특징 추출**
  - [ ] 고차 통계 (왜도, 첨도, 자기상관)
  - [ ] 주파수 도메인 (FFT, 웨이블릿 변환)
  - [ ] 공간 패턴 (곡률 분포 히스토그램, 가속도 벡터장 발산/회전)
  - [ ] Temporal Features (프레임 간 변화율, 가속도 변화율)
  - [ ] `auto_optimize.py`에 고급 특징 추출 옵션 추가
  - **예상 시간**: 7-10일
  - **의존성**: Phase 2.1 완료 ✅
  - **검증**: 합성 데이터로 특징 중요도 분석, DREAM/PatchCore 성능 향상 확인

### 2.4 Change Point Detection 고도화
**목표**: Change Point Detection 정확도 향상  
**검증 방법**: 합성 데이터로 정확도 향상 확인 ✅

- [ ] **파라미터 자동 튜닝**
  - [ ] 검증 세트 기반 최적 파라미터 탐색 (Grid Search/Bayesian)
  - [ ] GUI에 파라미터 자동 튜닝 옵션 추가
  - **예상 시간**: 5-7일
  - **의존성**: Phase 3.1 완료 ✅
  - **검증**: 합성 데이터로 최적 파라미터로 정확도 향상 확인

- [ ] **다중 특징 Change Point Detection**
  - [ ] acceleration, curvature, strain 등 여러 특징 동시 분석
  - [ ] 특징 간 상관관계 고려
  - [ ] 앙상블 Change Point Detection
  - **예상 시간**: 7-10일
  - **의존성**: Phase 3.1 완료 ✅
  - **검증**: 합성 데이터로 단일 특징 대비 정확도 향상 확인

---

## 🟡 우선순위 3: 실제 데이터 확보 후 진행 (대기)

### 3.1 Few-shot Fine-tuning ⚠️
**목표**: 소수 실제 크랙 데이터로 모델 성능 향상  
**검증 방법**: 실제 크랙 데이터로 Recall/Precision 향상 확인 ⚠️

- [ ] **DREAM Few-shot Fine-tuning API**
  - [ ] `ml_models/dream.py`: `fit_fewshot_anomaly()` 메서드 추가
  - [ ] 옵션 A: 판별기만 fine-tuning 구현
  - [ ] 옵션 B: 재구성 + 판별 fine-tuning 구현
  - [ ] 과적합 방지 (작은 learning rate, 조기 종료, L2 정규화)
  - [ ] 검증 스크립트: 소수 실제 크랙 도입 전·후 메트릭 비교
  - **예상 시간**: 7-10일
  - **의존성**: Phase 2.2 DREAM 완료 ✅, **실제 크랙 데이터 필요** ⚠️
  - **검증**: 실제 크랙 데이터로 Recall/Precision 향상 확인
  - **참고 문서**: `docs/DREAM_FEWSHOT_REAL_STRATEGY.md`

- [ ] **실제 크랙 데이터 파이프라인**
  - [ ] 현미경 검사로 확인된 크랙 패널 데이터 수집 가이드
  - [ ] 데이터 라벨링 및 전처리 스크립트
  - [ ] 특징 추출 및 DREAM 입력 포맷 변환
  - **예상 시간**: 3-5일 (데이터 수집 제외)
  - **의존성**: **실제 크랙 데이터 필요** ⚠️
  - **검증**: 데이터 품질 확인, 특징 추출 정확도

### 3.2 Contrastive Learning ⚠️
**목표**: 정상-크랙 특징 공간 분리  
**검증 방법**: 실제 크랙 데이터로 특징 공간 시각화 및 성능 향상 확인 ⚠️

- [ ] **Contrastive Learning 구현**
  - [ ] 정상-크랙 쌍을 멀리, 정상-정상 쌍을 가깝게 학습
  - [ ] Few-shot: 크랙 샘플이 적어도 효과적 학습 가능
  - [ ] Metric Learning (Triplet loss, N-pair loss)
  - **예상 시간**: 10-14일
  - **의존성**: **실제 크랙 데이터 필요** ⚠️
  - **검증**: 특징 공간 시각화, 성능 향상 확인

---

## 📋 우선순위별 일정 요약 (합성 데이터 우선)

| 우선순위 | 범위 | 예상 시간 | 주요 산출물 | 검증 방법 | 상태 |
|---------|------|----------|------------|----------|------|
| **1.1** | EXE 배포 준비 | 3-5일 | 완성된 EXE, Analyze 탭 확장 | 합성 데이터 ✅ | 🔄 진행 예정 |
| **1.2** | Change Point GUI 통합 | 5-7일 | Time Series Analysis 탭 | 합성 데이터 ✅ | 🔄 진행 예정 |
| **2.1** | 앙상블 구현 | 5-7일 | Hybrid 모델, GUI 통합 | 합성 데이터 ✅ | 📋 대기 |
| **2.2** | Temporal Modeling | 10-14일 | Temporal DREAM, Transformer | 합성 데이터 ✅ | 📋 대기 |
| **2.3** | 고급 특징 엔지니어링 | 7-10일 | 고급 특징 추출기 | 합성 데이터 ✅ | 📋 대기 |
| **2.4** | Change Point 고도화 | 12-17일 | 파라미터 튜닝, 다중 특징 | 합성 데이터 ✅ | 📋 대기 |
| **3.1** | Few-shot Fine-tuning | 7-10일 | Fine-tuning API | 실제 데이터 ⚠️ | ⏸️ 대기 |
| **3.2** | Contrastive Learning | 10-14일 | Contrastive 모델 | 실제 데이터 ⚠️ | ⏸️ 대기 |

**합성 데이터로 진행 가능한 작업 총 예상 시간**: 약 6-9주 (우선순위 1-2)  
**실제 데이터 필요 작업**: 우선순위 3 (데이터 확보 후 진행)

---

## 최적 개발 순서 (합성 데이터 우선)

### Phase A: 즉시 사용 가능한 기능 (2-3주)

1. **EXE 빌드 완성** (우선순위 1.1)
   - 사용자가 실제 데이터로 테스트 가능
   - 합성 데이터로 기본 기능 검증 ✅

2. **Analyze 탭 확장** (우선순위 1.1)
   - DREAM/PatchCore 모델을 GUI에서 직접 사용
   - 합성 데이터로 각 모드 검증 ✅

3. **Change Point Detection GUI 통합** (우선순위 1.2)
   - 크랙 발생 시점 감지 기능 완성
   - 합성 crack 시나리오로 검증 ✅

### Phase B: 합성 데이터로 검증 가능한 기능 강화 (3-4주)

4. **앙상블 구현** (우선순위 2.1)
   - DREAM + PatchCore 결합
   - 합성 데이터로 성능 향상 검증 ✅

5. **Temporal Modeling** (우선순위 2.2)
   - 시계열 의존성 모델링
   - 합성 데이터로 학습 및 검증 ✅

6. **고급 특징 엔지니어링** (우선순위 2.3)
   - 미세 패턴 차이 감지 강화
   - 합성 데이터로 특징 중요도 분석 ✅

7. **Change Point Detection 고도화** (우선순위 2.4)
   - 파라미터 자동 튜닝
   - 다중 특징 분석
   - 합성 데이터로 정확도 향상 검증 ✅

### Phase C: 실제 데이터 확보 후 진행 (대기)

8. **Few-shot Fine-tuning** (우선순위 3.1) ⚠️
   - 실제 크랙 데이터 필요
   - 합성 데이터로 기본 API는 구현 가능하나 검증은 실제 데이터 필요

9. **Contrastive Learning** (우선순위 3.2) ⚠️
   - 실제 크랙 데이터 필요

---

## 의존성 그래프 (합성 데이터 우선)

```
Phase A (즉시 사용)
├── EXE 빌드 완성
│   └── Analyze 탭 확장
└── Change Point GUI 통합
    └── Phase 3.1 모듈 (완료 ✅)

Phase B (합성 데이터 검증)
├── 앙상블 구현
│   ├── Phase 2.2 DREAM (완료 ✅)
│   └── Phase 2.3 PatchCore (완료 ✅)
├── Temporal Modeling
│   └── Phase 2.2 DREAM (완료 ✅)
├── 고급 특징 엔지니어링
│   └── Phase 2.1 완료 ✅
└── Change Point 고도화
    └── Phase 3.1 완료 ✅

Phase C (실제 데이터 필요) ⚠️
├── Few-shot Fine-tuning
│   ├── Phase 2.2 DREAM (완료 ✅)
│   └── 실제 크랙 데이터 필요 ⚠️
└── Contrastive Learning
    └── 실제 크랙 데이터 필요 ⚠️
```

---

## 진행 원칙 (합성 데이터 우선)

1. **합성 데이터로 검증 가능한 작업 우선**: 실제 데이터 확보 전까지 진행
2. **모듈별 완성도 유지**: 각 개발 단위마다 테스트와 검증 필수
3. **문헌 기반 개발**: 모호한 부분은 논문 조사 후 구현
4. **점진적 개선**: 작은 단위로 나누어 천천히 진행
5. **검증 중심**: 합성 데이터로 완전히 검증 가능한 작업부터 완성
6. **실제 데이터 준비**: 실제 데이터 확보 시 즉시 적용 가능하도록 API 설계

---

## 다음 단계

**즉시 시작**: 우선순위 1.1 - EXE 빌드 완성 및 Analyze 탭 확장

이 작업을 완료하면 사용자가 합성 데이터로 DREAM/PatchCore 모델을 테스트할 수 있으며, 실제 데이터 확보 시 바로 적용 가능합니다.

**실제 크랙 데이터 확보 시**: 우선순위 3 작업 시작 (Few-shot Fine-tuning, Contrastive Learning)
