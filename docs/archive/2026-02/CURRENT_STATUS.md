# 현재 개발 상태 (2026-02-17)

## ✅ 완료된 작업

### Phase 1: 파라미터 튜닝 GUI
- ✅ Crack Model Tuning 탭 (모든 파라미터 슬라이더)
- ✅ 파라미터 저장/로드 (JSON)
- ✅ 사용자 설정 자동 적용 (`run_analysis`)

### Phase 1.2: EXE 통합
- ✅ 기본 설정 파일 (`configs/crack_model_default.json`)
- ✅ 사용자 파라미터 자동 로드

### Phase 2.1: 데이터 준비
- ✅ `auto_optimize.py`: 데이터셋 로더 및 특징 추출
- ✅ GUI "Auto Optimization" 탭
- ✅ 데이터 준비 기능

### Phase 2.2: DREAM 모델 구현
- ✅ PyTorch 기반 오토인코더 구현
- ✅ MLP 아키텍처 (Encoder-Decoder)
- ✅ 정상 데이터 학습
- ✅ 재구성 오차 기반 이상 점수
- ✅ GUI 통합 (학습 및 평가)

---

## 🔄 진행 중 / 다음 작업

### Phase 2.3: PatchCore 구현
- [ ] scikit-learn 기반 메모리 뱅크 구축
- [ ] Coreset 선택 알고리즘
- [ ] 거리 기반 이상 점수 계산
- [ ] GUI 통합

### Phase 2.4: 파라미터 최적화
- [ ] Grid Search 구현
- [ ] Bayesian Optimization (Optuna)
- [ ] 검증 지표 계산 (AUC-ROC, F1-score)
- [ ] 최적 파라미터 저장 및 적용

### Phase 3: 시계열 이상 감지
- [ ] Change Point Detection (CUSUM, PELT)
- [ ] 프레임별 분류기
- [ ] Attention 기반 감지

---

## 📦 의존성

### 필수
- numpy, pandas, matplotlib, scipy (기본 분석)

### 선택적 (ML 기능)
```bash
pip install -e ".[ml]"  # torch, scikit-learn
```

---

## 🚀 사용 방법

### 1. 파라미터 튜닝
```
python -m motionanalyzer.desktop_gui
→ "Crack Model Tuning" 탭
→ 슬라이더 조정 → Preview → Save to User Config
```

### 2. DREAM 모델 학습
```
→ "Auto Optimization" 탭
→ 정상/크랙 데이터셋 추가
→ "Prepare Data"
→ Method: "DREAM Model"
→ "Start Optimization"
```

### 3. 분석 (자동 파라미터 적용)
```
→ "Analyze" 탭
→ Run Analysis
→ 사용자 파라미터 자동 적용
```

---

## 📝 주요 파일

```
src/motionanalyzer/
├── crack_model.py          ✅ 파라미터 저장/로드
├── desktop_gui.py          ✅ 4개 탭 (Tuning, Auto Opt)
├── analysis.py             ✅ 사용자 파라미터 자동 로드
├── auto_optimize.py        ✅ 데이터 준비 파이프라인
└── ml_models/
    ├── dream.py            ✅ PyTorch 구현 완료
    └── patchcore.py        ⚠️ 스켈레톤 (구현 대기)
```

---

## ⚠️ 알려진 제한사항

1. **PyTorch 의존성**: DREAM 모델 사용 시 `pip install torch` 필요
2. **EXE 크기**: PyTorch 포함 시 EXE 크기 증가 (CPU-only 빌드 고려 필요)
3. **PatchCore**: 아직 구현되지 않음 (Phase 2.3 예정)
4. **파라미터 최적화**: Grid Search/Bayesian 아직 구현되지 않음 (Phase 2.4 예정)

---

## 🎯 다음 우선순위

1. **PatchCore 구현** (scikit-learn 기반)
2. **파라미터 최적화** (Grid Search / Optuna)
3. **시계열 이상 감지** (Change Point Detection)
