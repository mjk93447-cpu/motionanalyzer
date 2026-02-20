# GUI 테스트 시나리오 실행 리포트

**실행일**: 2026-02-19  
**대상**: motionanalyzer-gui.exe (최신 빌드) + FPCB 테스트 스위트

---

## 1. 준비 완료 사항

### 1.1 합성 데이터 생성

- **스크립트**: `scripts/prepare_fpcb_test_suite.py`
- **위치**: `data/synthetic/fpcb_test_suite/`

| 유형 | 개수 | 프레임 | FPS | 설명 |
|------|------|--------|-----|------|
| **정상** | 200 | 60 | 30 | 정상 범위 내 랜덤 변수 (points 200–260, noise 0.12–0.38, panel 210–250 px 등) |
| **비정상(크랙)** | 10 | 60 | 30 | 6× full_crack, 2× mild_crack (pre_damage), 2× snap_crack (uv_overcured) |

- **폴더 구조**:
  ```
  fpcb_test_suite/
  ├── normal/
  │   ├── normal_001/   (frame_00000.txt .. frame_00059.txt, fps.txt, frame_metrics.csv, metadata.json)
  │   ├── normal_002/
  │   └── ... normal_200/
  └── crack/
      ├── crack_01_full_crack/
      ├── crack_02_full_crack/
      ├── ...
      ├── crack_07_mild_crack/
      └── crack_09_snap_crack/
  ```

### 1.2 EXE 빌드

- **경로**: `dist/motionanalyzer-gui.exe`
- **타입**: 경량 버전 (ML 기능 없음, Physics 모드 사용)

---

## 2. CLI 시뮬레이션 결과 (GUI와 동일 로직)

`scripts/run_gui_test_scenarios.py` 로 실행한 시나리오:

| 시나리오 | 내용 | 결과 |
|----------|------|------|
| **1** | normal_001 분석 (metadata 스케일) | mean_speed 59.2 px/s, mean_speed_m_s 0.0052 m/s |
| **2** | normal_001 + Scale 0.1 mm/px 오버라이드 | SI 단위 적용, max_acceleration_m_s2 0.37 m/s² |
| **3** | crack_01_full_crack 분석 | max_crack_risk 0.98 |
| **4** | normal vs crack Compare | delta_max_acceleration +13601 (크랙 시 가속도 급증) |
| **5** | normal_050, normal_100 배치 분석 | 60프레임 정상 처리 |

---

## 3. GUI에서 수행할 테스트 시나리오

### 시나리오 A: 정상 벤딩 단일 분석

1. **Analyze 탭** 열기
2. **Input bundle path**: `data/synthetic/fpcb_test_suite/normal/normal_001` (또는 Browse로 선택)
3. **Output analysis path**: `exports/vectors/fpcb_test_suite/normal_001`
4. **FPS**: 30
5. **Scale (mm/px)**: 비움 (metadata 사용) 또는 `0.1` 입력
6. **Analysis mode**: physics
7. **Run Analysis** 클릭
8. **확인**: Summary에 mean_speed, mean_speed_m_s 등 표시, Vector Map 표시

### 시나리오 B: Scale(mm/px) 입력 후 SI 단위 분석

1. **Input**: `data/synthetic/fpcb_test_suite/normal/normal_001`
2. **Output**: `exports/vectors/fpcb_test_suite/normal_001_si`
3. **Scale (mm/px)**: `0.1` 입력 (1 px = 0.1 mm)
4. **Run Analysis**
5. **확인**: 벡터 맵 축이 X (m), Y (m), 속도/가속도 단위 m/s, m/s²(또는 km/s²)

### 시나리오 C: 비정상(크랙) 분석

1. **Input**: `data/synthetic/fpcb_test_suite/crack/crack_01_full_crack`
2. **Output**: `exports/vectors/fpcb_test_suite/crack_01`
3. **Scale (mm/px)**: `0.1`
4. **Run Analysis**
5. **확인**: max crack risk 높음, 벡터 맵에 crack risk 주석 표시

### 시나리오 D: Compare (정상 vs 비정상)

1. **Compare 탭** 열기
2. **Base summary**: `exports/vectors/fpcb_test_suite/normal_001_si/summary.json`
3. **Candidate summary**: `exports/vectors/fpcb_test_suite/crack_01/summary.json`
4. **Compare** 클릭
5. **확인**: Delta 표시 (delta_max_acceleration 등)

### 시나리오 E: 여러 정상 샘플 순차 분석

1. normal_050, normal_100, normal_150 순서로 Analyze 실행
2. 각각 다른 output 폴더에 저장
3. 정상 범위 내에서 mean_speed 등 변동 확인

---

## 4. 검증 체크리스트

- [ ] EXE 실행 시 GUI 정상 표시
- [ ] Analyze: 입력 경로 선택, FPS 30, Scale 0.1 입력
- [ ] Run Analysis 후 Summary, Vector Map 표시
- [ ] Scale 입력 시 벡터 맵 축 X (m), Y (m) 확인
- [ ] Compare 탭에서 두 summary 비교
- [ ] Time Series Analysis 탭에서 Change Point Detection (선택)

---

## 5. 출력 파일 위치

- **분석 결과**: `exports/vectors/fpcb_test_suite/<video_name>/`
  - vectors.csv, summary.json, summary.txt, vector_map.png
- **테스트 스크립트**: `scripts/run_gui_test_scenarios.py`
