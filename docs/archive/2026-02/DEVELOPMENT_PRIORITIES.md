# 개발 우선순위 리스트 (2026년 2월 업데이트)

**최종 업데이트**: 2026년 2월 17일  
**현재 진행률**: 약 45% (Phase 1, 2 완료, Phase 3.1 부분 완료)

## 우선순위 기준

1. **즉시 사용 가능성**: 사용자가 바로 활용할 수 있는 기능
2. **실제 데이터 검증 가능성**: 실제 크랙 데이터로 검증 가능한 기능
3. **완성도 향상**: 기존 기능의 완성도 및 안정성 향상
4. **사용자 경험**: GUI 통합 및 사용 편의성
5. **장기적 확장성**: 향후 확장을 위한 기반 구축

---

## 🔴 우선순위 1: 즉시 사용 가능한 기능 완성 (2-3주)

### 1.1 EXE 배포 준비 완료
**목표**: 사용자가 실제 데이터로 테스트할 수 있도록 EXE 완성

- [ ] **EXE 빌드 스크립트 완성**
  - [ ] `scripts/build_exe.ps1`에 ML 모델 의존성 포함 확인
  - [ ] 모델 저장 경로 `%APPDATA%/motionanalyzer/models/` 통일
  - [ ] PyTorch CPU-only 빌드 또는 ONNX Runtime 사용 고려 (EXE 크기 최적화)
  - [ ] 사용자 설정 디렉토리 (`%APPDATA%/motionanalyzer/`) 지원 확인
  - **예상 시간**: 3-5일
  - **의존성**: 없음
  - **검증**: EXE 빌드 성공, 기본 기능 동작 확인

- [ ] **Analyze 탭 분석 모드 확장**
  - [ ] 콤보 "Analysis mode: Physics | DREAM | PatchCore"
  - [ ] DREAM/PatchCore 선택 시 저장된 모델 로드 후 predict
  - [ ] 이상 점수 시각화 (히스토그램, 시계열 플롯)
  - [ ] 모델이 없을 경우 안내 메시지 및 학습 유도
  - **예상 시간**: 5-7일
  - **의존성**: Phase 2 완료
  - **검증**: GUI에서 각 모드로 분석 실행 및 결과 확인

### 1.2 Change Point Detection GUI 통합
**목표**: 크랙 발생 시점 감지 기능을 GUI에서 사용 가능하게

- [ ] **Time Series Analysis 탭 추가**
  - [ ] Change Point Detection 모드 선택 (CUSUM, Window-based, PELT)
  - [ ] 시계열 특징 선택 (acceleration_max, curvature_concentration 등)
  - [ ] 파라미터 조정 UI (threshold, window_size 등)
  - [ ] 변화점 시각화 (시계열 플롯에 변화점 표시)
  - [ ] 결과 저장 및 리포트 생성
  - **예상 시간**: 5-7일
  - **의존성**: Phase 3.1 Change Point Detection 모듈 완료
  - **검증**: 합성 데이터로 크랙 발생 시점 정확도 확인

---

## 🟠 우선순위 2: 실제 데이터 검증 준비 (2-3주)

### 2.1 Few-shot Fine-tuning 구현
**목표**: 소수 실제 크랙 데이터로 모델 성능 향상

- [ ] **DREAM Few-shot Fine-tuning API**
  - [ ] `ml_models/dream.py`: `fit_fewshot_anomaly()` 메서드 추가
  - [ ] 옵션 A: 판별기만 fine-tuning 구현
  - [ ] 옵션 B: 재구성 + 판별 fine-tuning 구현
  - [ ] 과적합 방지 (작은 learning rate, 조기 종료, L2 정규화)
  - [ ] 검증 스크립트: 소수 실제 크랙 도입 전·후 메트릭 비교
  - **예상 시간**: 7-10일
  - **의존성**: Phase 2.2 DREAM 완료, 실제 크랙 데이터 필요
  - **검증**: 실제 크랙 데이터로 Recall/Precision 향상 확인
  - **참고 문서**: `docs/DREAM_FEWSHOT_REAL_STRATEGY.md`

- [ ] **실제 크랙 데이터 파이프라인**
  - [ ] 현미경 검사로 확인된 크랙 패널 데이터 수집 가이드
  - [ ] 데이터 라벨링 및 전처리 스크립트
  - [ ] 특징 추출 및 DREAM 입력 포맷 변환
  - **예상 시간**: 3-5일 (데이터 수집 제외)
  - **의존성**: 실제 크랙 데이터 필요
  - **검증**: 데이터 품질 확인, 특징 추출 정확도

### 2.2 앙상블 구현
**목표**: DREAM과 PatchCore 결합으로 성능 향상

- [ ] **앙상블 모듈 구현**
  - [ ] `ml_models/hybrid.py`: 앙상블 전략 구현
  - [ ] 가중 평균 전략 (α 최적화)
  - [ ] 최대값 전략 (Recall 향상)
  - [ ] 스태킹 전략 (메타 분류기)
  - [ ] GUI "ML & Optimization" 탭에 "Ensemble" 모드 추가
  - **예상 시간**: 5-7일
  - **의존성**: Phase 2.2, 2.3 완료
  - **검증**: 단일 모델 대비 성능 향상 확인

---

## 🟡 우선순위 3: 완성도 향상 (3-4주)

### 3.1 Temporal Modeling
**목표**: 프레임 간 의존성 모델링으로 시계열 패턴 강화

- [ ] **시계열 DREAM 확장**
  - [ ] `ml_models/dream_temporal.py`: LSTM/GRU 기반 시계열 오토인코더
  - [ ] 입력: 시퀀스 (T frames) → 출력: 재구성 시퀀스
  - [ ] Temporal Contrastive Learning 구현
  - [ ] 성능 비교: MLP DREAM vs Temporal DREAM
  - **예상 시간**: 10-14일
  - **의존성**: Phase 2.2 완료
  - **검증**: 시계열 데이터에서 성능 향상 확인

- [ ] **Transformer 기반 이상 감지** (선택)
  - [ ] `ml_models/transformer_anomaly.py`: Transformer 기반 시계열 인코더
  - [ ] Self-attention으로 장거리 의존성 포착
  - [ ] 크랙 발생 전후 패턴 변화 감지
  - **예상 시간**: 10-14일
  - **의존성**: Phase 2.2 완료
  - **검증**: Temporal DREAM 대비 성능 비교

### 3.2 고급 특징 엔지니어링
**목표**: 미세 패턴 차이 감지를 위한 특징 강화

- [ ] **고급 특징 추출**
  - [ ] 고차 통계 (왜도, 첨도, 자기상관)
  - [ ] 주파수 도메인 (FFT, 웨이블릿 변환)
  - [ ] 공간 패턴 (곡률 분포 히스토그램, 가속도 벡터장 발산/회전)
  - [ ] Temporal Features (프레임 간 변화율, 가속도 변화율)
  - **예상 시간**: 7-10일
  - **의존성**: Phase 2.1 완료
  - **검증**: 특징 중요도 분석, 성능 향상 확인

### 3.3 파라미터 자동 튜닝 개선
**목표**: Change Point Detection 파라미터 자동 최적화

- [ ] **Change Point Detection 파라미터 최적화**
  - [ ] 검증 세트 기반 최적 파라미터 탐색
  - [ ] Grid Search 또는 Bayesian Optimization 적용
  - [ ] GUI에 파라미터 자동 튜닝 옵션 추가
  - **예상 시간**: 5-7일
  - **의존성**: Phase 3.1 완료
  - **검증**: 최적 파라미터로 정확도 향상 확인

---

## 🟢 우선순위 4: 장기적 확장성 (4-6주)

### 4.1 Contrastive Learning (선택)
**목표**: 정상-크랙 특징 공간 분리

- [ ] **Contrastive Learning 구현**
  - [ ] 정상-크랙 쌍을 멀리, 정상-정상 쌍을 가깝게 학습
  - [ ] Few-shot: 크랙 샘플이 적어도 효과적 학습 가능
  - [ ] Metric Learning (Triplet loss, N-pair loss)
  - **예상 시간**: 10-14일
  - **의존성**: 실제 크랙 데이터 필요
  - **검증**: 특징 공간 시각화, 성능 향상 확인

### 4.2 다중 특징 Change Point Detection
**목표**: 여러 특징 동시 분석으로 정확도 향상

- [ ] **다중 특징 결합**
  - [ ] acceleration, curvature, strain 등 여러 특징 동시 분석
  - [ ] 특징 간 상관관계 고려
  - [ ] 앙상블 Change Point Detection
  - **예상 시간**: 7-10일
  - **의존성**: Phase 3.1 완료
  - **검증**: 단일 특징 대비 정확도 향상 확인

---

## 📋 우선순위별 일정 요약

| 우선순위 | 범위 | 예상 시간 | 주요 산출물 | 상태 |
|---------|------|----------|------------|------|
| **1.1** | EXE 배포 준비 | 3-5일 | 완성된 EXE, Analyze 탭 확장 | 🔄 진행 예정 |
| **1.2** | Change Point GUI 통합 | 5-7일 | Time Series Analysis 탭 | 🔄 진행 예정 |
| **2.1** | Few-shot Fine-tuning | 7-10일 | Fine-tuning API, 검증 스크립트 | 📋 대기 |
| **2.2** | 앙상블 구현 | 5-7일 | Hybrid 모델, GUI 통합 | 📋 대기 |
| **3.1** | Temporal Modeling | 10-14일 | Temporal DREAM, Transformer | 📋 대기 |
| **3.2** | 고급 특징 엔지니어링 | 7-10일 | 고급 특징 추출기 | 📋 대기 |
| **3.3** | 파라미터 자동 튜닝 | 5-7일 | 최적화 엔진 | 📋 대기 |
| **4.1** | Contrastive Learning | 10-14일 | Contrastive 모델 | 📋 대기 |
| **4.2** | 다중 특징 CPD | 7-10일 | 다중 특징 분석 | 📋 대기 |

**총 예상 시간**: 약 8-12주 (우선순위 1-3 기준)

---

## 즉시 시작 가능한 작업 (우선순위 1)

### 다음 개발 단위 (순서대로)

1. **EXE 빌드 완성** (우선순위 1.1)
   - 사용자가 실제 데이터로 테스트 가능
   - 즉시 가치 제공

2. **Analyze 탭 확장** (우선순위 1.1)
   - DREAM/PatchCore 모델을 GUI에서 직접 사용
   - 사용자 경험 향상

3. **Change Point Detection GUI 통합** (우선순위 1.2)
   - 크랙 발생 시점 감지 기능 완성
   - Phase 3.1 모듈 활용

---

## 의존성 그래프

```
우선순위 1 (즉시 사용)
├── EXE 빌드 완성
│   └── Analyze 탭 확장
└── Change Point GUI 통합
    └── Phase 3.1 모듈 (완료)

우선순위 2 (실제 데이터 검증)
├── Few-shot Fine-tuning
│   ├── Phase 2.2 DREAM (완료)
│   └── 실제 크랙 데이터 필요
└── 앙상블 구현
    ├── Phase 2.2 DREAM (완료)
    └── Phase 2.3 PatchCore (완료)

우선순위 3 (완성도 향상)
├── Temporal Modeling
│   └── Phase 2.2 DREAM (완료)
├── 고급 특징 엔지니어링
│   └── Phase 2.1 완료
└── 파라미터 자동 튜닝
    └── Phase 3.1 완료

우선순위 4 (장기적 확장)
├── Contrastive Learning
│   └── 실제 크랙 데이터 필요
└── 다중 특징 CPD
    └── Phase 3.1 완료
```

---

## 진행 원칙

1. **모듈별 완성도 유지**: 각 개발 단위마다 테스트와 검증 필수
2. **문헌 기반 개발**: 모호한 부분은 논문 조사 후 구현
3. **점진적 개선**: 작은 단위로 나누어 천천히 진행
4. **사용자 가치 우선**: 즉시 사용 가능한 기능부터 완성
5. **검증 중심**: 합성 데이터 → 실제 데이터 순서로 검증

---

## 다음 단계

**즉시 시작**: 우선순위 1.1 - EXE 빌드 완성 및 Analyze 탭 확장

이 작업을 완료하면 사용자가 실제 데이터로 DREAM/PatchCore 모델을 테스트할 수 있습니다.
