# 오프라인 Windows GUI 아키텍처 및 일관성 규칙

## 1. 탭·기능 관계

| 탭 | 역할 | 입력 | 출력 | 사용 모델 |
|----|------|------|------|------------|
| **Analyze** | 단일 번들 분석 | input dir, output dir, FPS | vectors.csv, summary, vector_map | **Physics** (CrackModelParams) |
| **Compare** | 두 분석 결과 비교 | base summary, candidate summary | Delta 수치 | 없음 (요약 비교) |
| **Crack Model Tuning** | Physics 모델 파라미터 편집 | 슬라이더, 테스트 데이터셋 경로 | 사용자 설정 저장, Preview 통계 | **Physics** |
| **ML & Optimization** | 학습/최적화 | 정상·크랙 데이터셋, 모드 선택 | 학습된 모델 또는 최적 파라미터 | **Physics / DREAM / PatchCore / Grid / Bayesian** |

- **분석(추론)** 은 Analyze 탭에서만 수행. 사용 중인 “분석 모드”는 향후 Physics / DREAM / PatchCore 중 선택 가능하도록 확장.
- **학습·최적화** 는 ML & Optimization 탭에서만 수행. 모드별로 코드 경로가 완전히 분리됨.

---

## 2. 모델 모드 정의 및 코드 분리

### 2.1 모드 종류

| 모드 ID | 설명 | 학습 필요 | 추론 시 사용 데이터 | 담당 모듈 |
|---------|------|-----------|--------------------|-----------|
| `physics` | 물리 기반 P(crack) (스트레인·곡률·임팩트) | 아니오 (파라미터만 튜닝) | CrackModelParams + vectors | `crack_model` |
| `dream` | 정상만 학습 오토인코더, 재구성 오차로 이상 점수 | 예 (정상 데이터) | 학습된 DREAM 모델 | `ml_models.dream` |
| `patchcore` | 정상 특징 메모리 뱅크, 거리로 이상 점수 | 예 (정상 데이터) | 학습된 PatchCore 모델 | `ml_models.patchcore` |
| `grid_search` | CrackModelParams 그리드 서치 | 아니오 (정상+크랙으로 검증) | 최적 CrackModelParams | `optimizers.grid_search` |
| `bayesian` | CrackModelParams 베이지안 최적화 | 아니오 (정상+크랙으로 검증) | 최적 CrackModelParams | `optimizers.bayesian` |

- **학습/최적화** 는 `gui.runners`에서 모드별로 한 진입점만 호출 (예: `run_training_or_optimization(mode, ...)`).
- GUI는 모드 선택 후 “실행” 시 해당 러너만 호출하므로, 모델 간 의존성/중복 없이 코드 완전 분리.

### 2.2 Analyze 탭의 “분석 모드” (향후 확장)

- 현재: 항상 **Physics** (run_analysis → crack_model).
- 확장 시: 콤보박스로 [Physics | DREAM | PatchCore] 선택 시,
  - Physics: 기존 `run_analysis` (crack_risk).
  - DREAM/PatchCore: 저장된 모델 로드 후 `model.predict(features)` → 이상 점수를 crack_risk 대신 표시 (동일 UI 재사용).

---

## 3. 네이밍 규칙

### 3.1 GUI 위젯·변수

| 용도 | 규칙 | 예 |
|------|------|-----|
| 탭 프레임 | `_build_<tab>_tab(parent)` | `_build_analyze_tab`, `_build_ml_optimization_tab` |
| 버튼/동작 핸들러 | `_on_<action>` | `_on_run_analysis`, `_on_compare_summaries` |
| 브라우저(파일/폴더 선택) | `_browse_<what>` | `_browse_input_dir`, `_browse_preview_dataset` |
| 문자열/숫자 입력 변수 | `<name>_var` | `input_dir_var`, `fps_var`, `opt_method_var` |
| 텍스트 출력 위젯 | `<context>_text` | `summary_text`, `log_text`, `auto_opt_text` |
| 내부 상태(캔버스 등) | `_<name>` (private) | `_vector_map_canvas`, `_vector_map_fig` |

### 3.2 패킹·레이아웃

- `padx=8, pady=4` 통일.
- LabelFrame 제목: 영문 (코드·번역 일관성).
- 버튼 텍스트: "Run Analysis", "Browse...", "Prepare Data" 등 동사/명령형.

### 3.3 모델·러너

- 모드 상수: 소문자 스네이크 `physics`, `dream`, `patchcore`, `grid_search`, `bayesian`.
- 러너 함수: `run_<mode>_training` 또는 `run_<mode>_optimization` (gui.runners에서만 사용).
- GUI에서는 `_on_start_ml_or_optimization` → `runners.run(mode, ...)` 한 번만 호출.

---

## 4. 코드 패턴

### 4.1 탭 빌드

- 한 탭 = 한 `_build_*_tab` 메서드.
- 탭 내 섹션 = LabelFrame + 자식 Frame으로 그룹화.
- “진행/결과” 텍스트는 탭당 하나의 Text 위젯 (예: `auto_opt_text`).

### 4.2 에러 처리

- try/except 후 사용자 메시지: `messagebox.showerror("Error", ...)`.
- 로그/진행 창: `self.<tab>_text.insert(tk.END, ...)` + `see(tk.END)`.
- traceback은 로그 창에만 출력, 메시지박스에는 요약만.

### 4.3 데이터 흐름

- Analyze: `input_dir_var` → `run_analysis()` → `summary`, `vectors.csv` → Summary 표시, Vector Map 표시.
- Compare: `base_summary_var`, `cand_summary_var` → `compare_summaries()` → `compare_text`.
- Tuning: `param_vars` ↔ `CrackModelParams` ↔ 파일/사용자 설정; Preview 시 선택 데이터셋으로만 `compute_crack_risk` 호출.
- ML & Optimization: 데이터셋 목록 → Prepare Data → `training_features`, `training_labels` → 모드 선택 → `runners.run(mode, ...)` → 결과 텍스트.

---

## 5. 중복 제거

- “데이터 준비” 로직: `auto_optimize.prepare_training_data` + `FeatureExtractionConfig` 한 곳만 사용.
- “Physics 파라미터 로드”: `get_user_params_path()` + `load_params()` 한 곳; Analyze와 Tuning 모두 이 경로 사용.
- 모델 학습/최적화: GUI에 분기 여러 개 두지 말고, `gui.runners.run(mode, features_df, labels, **options)` 형태로 통합.

---

## 6. EXE 완성도를 위한 체크리스트

- [ ] 모든 탭이 동일한 패딩/스타일 사용
- [ ] 모드 선택은 라디오 또는 콤보로 한 곳에서만 (ML & Optimization 탭)
- [ ] Physics / DREAM / PatchCore / Grid / Bayesian 각각 별도 러너 모듈 또는 함수로 분리
- [ ] Analyze 탭에 “분석 모드” 콤보 추가 시 Physics/DREAM/PatchCore 전환 가능
- [ ] 사용자 설정 경로: `%APPDATA%/motionanalyzer/` (이미 반영)
- [ ] 모델 저장 경로: `%APPDATA%/motionanalyzer/models/` 등 고정 경로 사용 권장

---

이 문서는 최종 결과물인 오프라인 윈도우 GUI의 완성도와 일관성을 위해 준수할 규칙을 정의합니다.
