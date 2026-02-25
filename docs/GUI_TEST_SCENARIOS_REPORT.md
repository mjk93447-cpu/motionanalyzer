# GUI 테스트 시나리오 실행 리포트

**실행일**: 2026-02-25  
**대상**: motionanalyzer-gui.exe + DS-260223-ml-fp-20k-60f (ml_dataset_fp_focused)

---

## 1. 준비 완료 사항

### 1.1 합성 데이터 (DS-260223-ml-fp-20k-60f)

- **경로**: `data/synthetic/ml_dataset_fp_focused/`
- **생성**: `python scripts/generate_ml_dataset.py --scale fp_focused --out data/synthetic/ml_dataset_fp_focused`

| 유형 | 개수 | 프레임 | 설명 |
|------|------|--------|------|
| **normal** | ~16k | 60 | 정상 벤딩 |
| **crack_in_bending** | ~1.05k | 60 | 벤딩 중 크랙 |
| **thick_panel** | ~2.8k | 60 | 두꺼운 패널 |
| **pre_damaged** | 150 | 60 | 사전 손상 |

- **폴더 구조**:
  ```
  ml_dataset_fp_focused/
  ├── normal/
  │   ├── normal_0001/
  │   ├── normal_0002/
  │   └── ...
  ├── crack_in_bending/
  │   ├── crack_0001/
  │   └── ...
  ├── thick_panel/
  ├── pre_damaged/
  └── manifest.json
  ```

### 1.2 EXE 빌드

- **경로**: `dist/motionanalyzer-gui.exe`
- **타입**: 경량 또는 ML 포함 (`-IncludeML`)

---

## 2. CLI 시뮬레이션 결과

`scripts/run_gui_test_scenarios.py` 로 실행:

| 시나리오 | 내용 | 결과 |
|----------|------|------|
| **1** | normal_0001 분석 (metadata 스케일) | mean_speed, mean_speed_m_s |
| **2** | normal_0001 + Scale 0.1 mm/px 오버라이드 | SI 단위 적용 |
| **3** | crack_0001 분석 | max_crack_risk |
| **4** | normal vs crack Compare | delta_max_acceleration 등 |
| **5** | normal_0050, normal_0100 배치 분석 | 60프레임 정상 처리 |

---

## 3. GUI에서 수행할 테스트 시나리오

### 시나리오 A: 정상 벤딩 단일 분석

1. **Analyze 탭** 열기
2. **Input bundle path**: `data/synthetic/ml_dataset_fp_focused/normal/normal_0001`
3. **Output analysis path**: `exports/vectors/gui_test_scenarios/normal_0001`
4. **FPS**: 30
5. **Scale (mm/px)**: 비움 또는 `0.1`
6. **Run Analysis** 클릭

### 시나리오 B: Scale(mm/px) 입력 후 SI 단위 분석

1. **Input**: `data/synthetic/ml_dataset_fp_focused/normal/normal_0001`
2. **Output**: `exports/vectors/gui_test_scenarios/normal_0001_scale01`
3. **Scale (mm/px)**: `0.1`
4. **Run Analysis**

### 시나리오 C: 비정상(크랙) 분석

1. **Input**: `data/synthetic/ml_dataset_fp_focused/crack_in_bending/crack_0001`
2. **Output**: `exports/vectors/gui_test_scenarios/crack_0001`
3. **Scale (mm/px)**: `0.1`
4. **Run Analysis**

### 시나리오 D: Compare (정상 vs 비정상)

1. **Compare 탭** 열기
2. **Base summary**: `exports/vectors/gui_test_scenarios/normal_0001_scale01/summary.json`
3. **Candidate summary**: `exports/vectors/gui_test_scenarios/crack_0001/summary.json`
4. **Compare** 클릭

---

## 4. 검증 체크리스트

- [ ] EXE 실행 시 GUI 정상 표시
- [ ] Analyze: DS-002 경로 선택, FPS 30, Scale 0.1
- [ ] Run Analysis 후 Summary, Vector Map 표시
- [ ] Compare 탭에서 두 summary 비교
- [ ] Time Series Analysis 탭에서 Change Point Detection (선택)

---

## 5. 출력 파일 위치

- **분석 결과**: `exports/vectors/gui_test_scenarios/<bundle_name>/`
- **테스트 스크립트**: `scripts/run_gui_test_scenarios.py`
