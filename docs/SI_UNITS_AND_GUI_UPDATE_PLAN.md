# SI 단위계 및 분석·시각화·GUI 업데이트 계획

**문서 목적**: 오프라인 Windows 분석 프로그램의 실제 사용자 시나리오를 전제로, 물리량을 철저히 SI 단위계로 통일하고, 픽셀 스케일(mm/px)을 GUI에서 입력받아 분석·시각화 전 과정에 반영하는 기능 및 GUI 개선 계획을 설계한다.

**최종 갱신**: 2026-02-19

---

## 1. 현재 상태 요약 (실행·코드 기반)

### 1.1 실행 환경

- **데스크톱 GUI**: `python -m motionanalyzer.desktop_gui` 로 실행 (Tkinter 기반 오프라인 Windows GUI).
- **EXE**: `dist/` 에 미빌드 상태일 수 있음. 동일 기능은 Python 실행으로 검증 가능.

### 1.2 사용자 시나리오 (Analyze 탭 기준)

1. **Input/Output**: 입력 번들 경로, 출력 경로, FPS 입력, 분석 모드(physics / dream / patchcore / ensemble) 선택.
2. **Run Analysis** 클릭 → `run_analysis(input_dir, output_dir, fps=fps_val)` 호출.
3. **스케일(픽셀당 길이)**: 현재 **GUI에서 입력받지 않음**. `meters_per_pixel`은 **metadata.json 에만** 의존하며, 메타데이터가 없으면 SI 컬럼/요약이 생성되지 않음.
4. **결과**: Summary 텍스트, vector_map.png, vectors.csv, summary.json 등. 벡터 맵은 좌표가 **픽셀(px)** 이고, 속도/가속도 레이블만 조건부로 m/s, km/s² 사용.

### 1.3 코드상 스케일·단위 흐름

| 구간 | 현재 동작 | 한계 |
|------|-----------|------|
| **입력** | `load_bundle()` → `_read_meters_per_pixel(input_dir)` 로 metadata.json 에서만 m/px 읽음 | 실측 데이터는 메타데이터 없음. 사용자가 알고 있는 “mm/px” 값을 넣을 수 없음. |
| **분석** | `compute_vectors(..., meters_per_pixel=...)` → 있으면 speed_si, acceleration_si 등 추가 | 스케일이 없으면 SI 컬럼 자체가 없음. |
| **크랙 모델** | `compute_crack_risk(..., meters_per_pixel=...)` 사용하나, impact cap 등은 **px/s²** 기준 (`impact_cap_px_s2`) | 물리 모델 완전성 측면에서 가속도 cap 도 SI(m/s²)로 입력·표시하는 것이 일관됨. |
| **시각화** | `plot_full_vector_map` / `create_full_vector_map_figure`: **x, y 는 항상 px**. use_si 일 때만 vel/acc 단위를 m/s, km/s² 로 표기 | 그래프 축은 여전히 “X (px)”, “Y (px)”. 실제 미터 값으로 보이지 않음. |
| **출력** | summary.txt 에 mean_speed_m_s 등은 meters_per_pixel 있을 때만 기록 | 스케일 없으면 SI 필드 없음. |

정리하면:

- **스케일 입력**: GUI 에 “픽셀당 mm” 또는 “m/px” 입력이 없음.
- **분석 파이프라인**: 사용자 입력 스케일을 받아 `run_analysis` 에 전달하는 경로가 없음.
- **시각화**: 축·좌표가 픽셀 단위; SI 로 통일하려면 **축을 미터(m)** 로 하고, 주석·범례도 SI 로 통일해야 함.
- **물리 모델**: 내부적으로 px 단위 cap 을 쓰고 있어, “완전한 SI” 관점에서는 사용자에게 m/s² 등으로 입력받고 변환하는 단계가 있으면 좋음.

---

## 2. 목표 원칙

1. **SI 단위계 일원화**: 최종 사용자에게 보이는 수치·그래프·파일 출력은 **길이: m, 시간: s, 속도: m/s, 가속도: m/s²** 등 SI 로 통일.
2. **픽셀 스케일은 “처리 과정”에서만**: 픽셀(px) 단위는 내부 처리용이며, 사용자에게는 **“픽셀당 mm” 또는 “m/px”** 를 한 번 입력받아, 이후 분석·시각화·요약은 모두 SI 로 전달.
3. **GUI ↔ 분석 연계**: Analyze 탭에서 스케일을 입력하면, run_analysis → compute_vectors → compute_crack_risk → plot_full_vector_map / create_full_vector_map_figure 까지 **한 번에 시뮬레이션 가능한 단일 파이프라인**으로 흐르도록 설계.
4. **출력 이미지**: 벡터 맵 등 결과 이미지의 **축은 미터(m)**, 주석·범례는 **m/s, m/s²(또는 km/s²)** 등 SI 단위만 사용.

---

## 3. 기능·GUI 업데이트 설계

### 3.1 GUI: 스케일 입력 (Analyze 탭)

- **위치**: Input/Output 블록 안, FPS 입력 행 바로 아래(또는 동일 블록 내 논리적 위치).
- **항목 1 – “Scale (mm/px)”**
  - 사용자 입력: **픽셀 1개당 실측 길이 (mm)**. 예: 0.1 이면 1 px = 0.1 mm.
  - 변환: `meters_per_pixel = (mm_per_px * 1e-3)` (mm → m).
- **대안/추가 – “Meters per pixel (m/px)”**
  - 고급 사용자용으로 “m/px” 직접 입력 허용해도 됨. 내부적으로는 하나의 `meters_per_pixel` 로 통일.
- **동작**:
  - **비어 있거나 0**: 기존과 동일. `metadata.json` 에서만 `meters_per_pixel` 읽어서 사용하고, 없으면 None(픽셀 모드).
  - **값 입력**: 해당 값을 `meters_per_pixel` 로 사용하고, **metadata 에 있어도 오버라이드** (사용자 입력 우선).
- **연계**: `_on_run_analysis()` 에서
  - GUI “Scale (mm/px)” (또는 m/px) → `meters_per_pixel_override` 계산.
  - `run_analysis(..., meters_per_pixel_override=...)` 호출.

### 3.2 분석 파이프라인: run_analysis 확장

- **시그니처 확장**  
  `run_analysis(input_dir, output_dir, fps=None, crack_params=None, meters_per_pixel_override=None)`  
  - `meters_per_pixel_override`: `float | None`.  
    - `None`: 기존과 동일. `load_bundle()` 반환값의 `meters_per_pixel` 사용(metadata 유무에 따라 None 또는 값).  
    - `float` (>0): 메타데이터와 무관하게 이 값을 **전 구간**에서 사용.
- **내부 흐름**  
  1. `df, fps_val, m_from_meta = load_bundle(input_dir, fps=fps)`  
  2. `meters_per_pixel = meters_per_pixel_override if meters_per_pixel_override is not None and meters_per_pixel_override > 0 else m_from_meta`  
  3. `compute_vectors(..., meters_per_pixel=meters_per_pixel)`  
  4. `compute_crack_risk(..., meters_per_pixel=meters_per_pixel)`  
  5. `summarize(..., meters_per_pixel=meters_per_pixel)`  
  6. `export_analysis(...)` (이미 summary.meters_per_pixel 사용 중)
- **결과**: 사용자가 GUI 에서 mm/px 만 입력하면, 분석 전 과정이 동일한 `meters_per_pixel` 로 시뮬레이션되며, summary/vectors 에 SI 필드가 채워짐.

### 3.3 시각화: 축·단위를 SI(m) 로 통일

- **목표**: `meters_per_pixel` 이 설정된 경우, **그래프 축은 실측 길이(m)**, 모든 텍스트는 SI 단위.
- **구현 요약**  
  - **좌표 변환**:  
    - `x_m = x_px * meters_per_pixel`, `y_m = y_px * meters_per_pixel`  
    - scatter / quiver / LineCollection 등 **모든 그리기**는 `(x_m, y_m)` 기준으로 수행. (화면상 비율 유지를 위해 `set_aspect('equal')` 유지.)
  - **축 레이블**:  
    - `ax.set_xlabel("X (m)")`, `ax.set_ylabel("Y (m)")`.
  - **제목·범례·주석**:  
    - 속도: m/s, 가속도: m/s² (또는 필요 시 km/s²).  
    - “px” 표기는 제거.  
    - scale_note 는 “scale = … m/px” 형태로 유지해도 됨.
- **대상 함수**  
  - `plot_full_vector_map()`  
  - `create_full_vector_map_figure()` (GUI 임베딩용)
- **use_si 가 False 인 경우(스케일 없음)**: 현재처럼 축은 “X (px)”, “Y (px)”, 속도/가속도는 px/s, px/s² 유지.

### 3.4 출력 파일·요약 표시

- **summary.txt / summary.json**: 이미 `meters_per_pixel` 이 있으면 mean_speed_m_s 등 SI 필드 기록. 변경 없이, “스케일 입력 시 항상 이 필드가 채워지도록” 하면 됨.
- **벡터 맵 PNG**: 위 3.3 적용 시, 저장되는 이미지 자체가 “X (m), Y (m)”, “m/s, m/s²” 로 통일됨.
- **GUI Summary 영역**:  
  - `meters_per_pixel` 이 있을 때는 **SI 값(mean_speed_m_s, max_speed_m_s, mean_acceleration_m_s2, max_acceleration_m_s2)** 를 기본으로 표시하고,  
  - px 단위는 “참고” 또는 접기 형태로 표시하도록 할 수 있음 (선택 사항).

### 3.5 크랙 모델(물리 모델) 보강 (선택)

- **현재**: `impact_cap_px_s2` 등이 px/s² 단위로 하드코딩/설정.
- **개선 방향**:  
  - 사용자/설정에서 **가속도 상한을 m/s²** 로 입력받고,  
  - 내부에서 `meters_per_pixel` 로 px/s² 로 변환해 기존 `impact_cap_px_s2` 에 넣어 사용.  
  - 또는 CrackModelParams 에 `impact_cap_m_s2` 를 추가하고, `compute_crack_risk` 진입 시 `meters_per_pixel` 이 있으면 m/s² → px/s² 변환 후 cap 적용.
- 이렇게 하면 “물리 모델이 완벽하게 SI 기준으로 실행”된다는 요구를 만족시킬 수 있음.

### 3.6 Compare / Time Series / ML 탭

- **Compare**: summary 에 이미 SI 필드가 있으면, delta 도 자동으로 SI 단위로 해석 가능. 비교 결과 표시 시 “(m/s)”, “(m/s²)” 등 단위 표기만 명시하면 됨.
- **Time Series / ML**: 입력으로 쓰는 vectors/summary 에 이미 SI 컬럼이 포함되므로, 스케일 입력을 Analyze 에서만 하면 추가 변경 최소화 가능.

---

## 4. 구현 순서 제안 및 진행

1. **run_analysis 확장** — ✅ 구현됨  
   - `meters_per_pixel_override` 인자 추가 및 내부에서 `meters_per_pixel` 결정 로직.
2. **Analyze 탭 GUI** — ✅ 구현됨  
   - “Scale (mm/px)” 입력 위젯 추가, `_on_run_analysis` 에서 파싱 후 `run_analysis(..., meters_per_pixel_override=...)` 호출.
3. **시각화 SI 축** — ✅ 구현됨  
   - `plot_full_vector_map`, `create_full_vector_map_figure` 에서 use_si 일 때 좌표를 m 로 변환, 축 "X (m)", "Y (m)", 주석/범례 SI 통일.
4. **Summary 표시**  
   - GUI Summary 에서 meters_per_pixel 있을 때 SI 값 우선 표시 (선택).
5. **크랙 모델 SI cap**  
   - impact 등 cap 을 m/s² 로 입력받고 내부에서 px 변환 (선택).

---

## 5. 검증 시나리오

1. **스케일 없음**: Scale (mm/px) 비움 → metadata 없음 → 기존과 동일 (px 축, px/s, px/s²).
2. **스케일만 입력**: Scale = 0.1 (mm/px) → metadata 없어도 전체 분석·벡터맵·요약이 m, m/s, m/s² 로 동작.
3. **메타데이터 + GUI 입력**: metadata 에 m/px 있음 + GUI 에 값 입력 → GUI 입력이 우선, 일관된 스케일로 파이프라인 실행.
4. **출력 이미지**: vector_map.png 를 열어 축이 “X (m)”, “Y (m)” 이고, 범례/주석이 m/s, m/s² 인지 확인.

---

## 6. 참고 (단위 관계)

- **1 mm = 1e-3 m**
- **mm/px → m/px**: `meters_per_pixel = (mm_per_px) * 1e-3`
- **속도**: v_px_s → v_m_s = v_px_s * meters_per_pixel
- **가속도**: a_px_s2 → a_m_s2 = a_px_s2 * meters_per_pixel
- **곡률/curvature_like**: 1/px 단위이므로, SI 곡률(1/m) 로 쓰려면 `curvature_si = curvature_like * meters_per_pixel` (선택 적용).

이 계획에 따라 구현하면, “픽셀당 mm 값을 사용자에게 받고, 처리 과정을 SI 로 시뮬레이션하며, 결과 이미지까지 미터·m/s·m/s² 로 통일”하는 목표를 충족할 수 있다.
