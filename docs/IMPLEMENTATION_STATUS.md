# FPCB 크랙 탐지 시스템 구현 현황

## 완료된 작업 (Phase 1.1)

### 1. 파라미터 저장/로드 기능 (`crack_model.py`)
- ✅ `save_params(params, path)`: CrackModelParams를 JSON으로 저장
- ✅ `load_params(path)`: JSON에서 CrackModelParams 로드
- ✅ `get_user_params_path()`: Windows `%APPDATA%/motionanalyzer/` 경로 지원

### 2. GUI 확장 (`desktop_gui.py`)
- ✅ **"Crack Model Tuning" 탭** 추가
- ✅ 모든 `CrackModelParams` 필드를 슬라이더로 편집 가능:
  - Caps: `strain_cap`, `curvature_concentration_cap`, `bend_angle_cap_deg`, `impact_cap_px_s2`
  - Weights: `w_strain`, `w_stress`, `w_curvature_concentration`, `w_bend_angle`, `w_impact`
  - Sigmoid: `sigmoid_steepness`, `sigmoid_center`
- ✅ 파라미터 파일 저장/로드 (JSON)
- ✅ 사용자 설정 디렉토리에 자동 저장
- ✅ 실시간 미리보기: 선택된 데이터셋에 대해 파라미터 적용 후 `crack_risk` 통계 표시

### 3. 딥러닝 모델 기본 구조
- ✅ `ml_models/dream.py`: DREAM (Deep Reconstruction Error-based Anomaly Model) 스켈레톤
- ✅ `ml_models/patchcore.py`: PatchCore (Memory Bank 기반) 스켈레톤
- ✅ 추상 클래스 인터페이스 정의 (fit, predict, predict_binary, save, load)
- ⚠️ 실제 PyTorch/scikit-learn 구현은 Phase 2에서 진행

---

## 완료된 작업 (Phase 1.2)

### 1. 기본 설정 파일
- ✅ `configs/crack_model_default.json`: 기본 파라미터 JSON 파일 생성
- ✅ `run_analysis()`: 사용자 파라미터 자동 로드 (사용자 설정 > 기본값)

### 2. 사용자 파라미터 자동 적용
- ✅ `run_analysis()`가 `%APPDATA%/motionanalyzer/crack_model_params.json`에서 자동 로드
- ✅ GUI에서 저장한 파라미터가 분석 시 자동 적용됨

---

## 완료된 작업 (Phase 2.1)

### 1. 데이터 준비 파이프라인 (`auto_optimize.py`)
- ✅ `load_dataset()`: 정상/크랙 데이터셋 로드 및 벡터 계산
- ✅ `extract_features()`: 특징 추출 (프레임별, 포인트별, 전역 통계)
- ✅ `prepare_training_data()`: 여러 데이터셋 결합 및 라벨링
- ✅ `normalize_features()`: 특징 정규화 (z-score)

### 2. GUI "Auto Optimization" 탭
- ✅ 정상/크랙 데이터셋 선택 (여러 세션 지원)
- ✅ 특징 추출 옵션 (프레임별, 포인트별, 전역 통계)
- ✅ 최적화 방법 선택 (Grid Search, Bayesian, DREAM, PatchCore)
- ✅ 데이터 준비 실행 및 결과 표시
- ⚠️ 최적화 실행은 Phase 2.2-2.4에서 구현 예정

---

## 다음 단계 (Phase 2.2-2.4)

### Phase 2.2: DREAM 구현
- [ ] PyTorch 기반 오토인코더 구현
- [ ] LSTM/Transformer 인코더-디코더 아키텍처
- [ ] 정상 데이터만으로 학습
- [ ] 재구성 오차 기반 이상 점수 계산

### Phase 2.3: PatchCore 구현
- [ ] 특징 추출 (프레임별 또는 패치별)
- [ ] Coreset 선택 알고리즘 (k-means 또는 랜덤 샘플링)

#### 2.2 DREAM 구현
- [ ] PyTorch 기반 오토인코더 구현
- [ ] LSTM/Transformer 인코더-디코더 아키텍처
- [ ] 정상 데이터만으로 학습
- [ ] 재구성 오차 기반 이상 점수 계산

#### 2.3 PatchCore 구현
- [ ] 특징 추출 (프레임별 또는 패치별)
- [ ] Coreset 선택 알고리즘 (k-means 또는 랜덤 샘플링)
- [ ] NearestNeighbors 기반 거리 계산
- [ ] 메모리 뱅크 구축 및 이상 점수 계산

#### 2.4 파라미터 자동 최적화
- [ ] 그리드 서치 또는 베이지안 최적화 (`optuna`)
- [ ] 검증 지표: AUC-ROC, F1-score
- [ ] GUI에서 최적화 실행 및 진행률 표시

---

## 사용 방법 (현재 구현)

### 1. GUI에서 파라미터 튜닝

```powershell
python -m motionanalyzer.desktop_gui
```

1. **"Crack Model Tuning" 탭** 열기
2. 슬라이더로 파라미터 조정
3. "Test dataset"에 데이터셋 경로 입력
4. "Preview" 클릭 → `crack_risk` 통계 확인
5. "Save to User Config" 클릭 → `%APPDATA%/motionanalyzer/crack_model_params.json`에 저장

### 2. 분석 시 사용자 파라미터 사용

현재 `run_analysis`는 기본 `CrackModelParams()`를 사용합니다.  
사용자 파라미터를 적용하려면:

```python
from motionanalyzer.crack_model import get_user_params_path, load_params
from motionanalyzer.analysis import load_bundle, compute_vectors
from motionanalyzer.crack_model import compute_crack_risk, load_frame_metrics

params = load_params(get_user_params_path())
df, fps, mpp = load_bundle(input_dir)
vectors = compute_vectors(df, fps, mpp)
frame_metrics = load_frame_metrics(input_dir / "frame_metrics.csv")
vectors = compute_crack_risk(vectors, frame_metrics, 1/fps, meters_per_pixel=mpp, params=params)
```

`run_analysis()`는 사용자 파라미터를 자동 로드함 (`get_user_params_path()` → `load_params()`).

---

## 파일 구조

```
src/motionanalyzer/
├── crack_model.py          ✅ 파라미터 저장/로드
├── desktop_gui.py          ✅ 4개 탭 (Analyze, Compare, Tuning, ML & Optimization)
├── analysis.py             ✅ 사용자 파라미터 자동 로드
├── auto_optimize.py        ✅ 데이터 준비 파이프라인
├── gui/
│   ├── __init__.py         ✅
│   └── runners.py          ✅ 모드별 러너 (physics/dream/patchcore/grid_search/bayesian)
└── ml_models/
    ├── dream.py            ✅ PyTorch 구현 완료
    └── patchcore.py        ⚠️ 스켈레톤 (Phase 2.3 구현 대기)

docs/
├── CRACK_DETECTION_ROADMAP.md  ✅ 중장기 계획 + 구체적 다음 단계
├── GUI_ARCHITECTURE.md         ✅ GUI 일관성·모델 모드·네이밍 규칙
└── IMPLEMENTATION_STATUS.md    ✅ 이 문서
```

---

## 테스트

```powershell
# GUI 테스트
python -m motionanalyzer.desktop_gui

# 파라미터 저장/로드 테스트
python -c "from motionanalyzer.crack_model import CrackModelParams, save_params, load_params, get_user_params_path; p = CrackModelParams(); save_params(p, get_user_params_path()); print('Saved:', get_user_params_path())"
```

---

## 참고

- 중장기 계획: `docs/CRACK_DETECTION_ROADMAP.md`
- 현재 구현: Phase 1.1 완료, Phase 1.2 및 Phase 2 진행 예정
