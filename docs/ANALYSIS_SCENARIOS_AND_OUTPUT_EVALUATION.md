# 분석 시나리오·출력 이미지 평가 및 중장기 개발 전략 확장

**작성일**: 2026년 2월 19일  
**연계 문서**: `DEVELOPMENT_ROADMAP_FINAL.md`, `SI_UNITS_AND_GUI_UPDATE_PLAN.md`, `PROJECT_GOALS.md`

### 프로젝트 최종 목표 (우선순위)

| 우선순위 | 목표 | 시나리오 매핑 |
|----------|------|---------------|
| **1** | 벤딩 중 크랙 감지 (시계열·국소적) | 시나리오 2, 4 — CPD, 충격파·진동 |
| **2** | 이미 크랙된 패널 감지 (전체적 패턴) | 시나리오 3, 7, 8 — Compare, ML, 유형별 |

---

## 1. 출력 결과 이미지 실행 및 설명

### 1.1 정상 벤딩 (normal_001) — metadata 스케일

**경로**: `exports/vectors/fpcb_test_suite/normal_001/vector_map.png`

| 항목 | 값 |
|------|-----|
| scale | 8.75e-05 m/px (metadata에서 자동) |
| 축 | X (m), Y (m) — SI 단위 |
| 프레임/포인트 | 60 frames, 253 points |
| mean_speed_m_s | 0.0052 m/s |
| max_acceleration_m_s2 | 0.325 m/s² |

**이미지 설명**:
- 아치형/U형 곡선 패턴으로 FPCB 정상 벤딩 진행
- 인덱스별 색상으로 속도(m/s), 보색으로 가속도(km/s²) 표시
- 축이 미터(m)로 표기되어 SI 단위 적용
- scale=8.75e-05 m/px로 픽셀–물리량 변환 명시

---

### 1.2 정상 벤딩 (normal_001) — Scale 0.1 mm/px 사용자 입력

**경로**: `exports/vectors/fpcb_test_suite/normal_001_scale01/vector_map.png`

| 항목 | 값 |
|------|-----|
| scale | 1.00e-04 m/px (0.1 mm/px 사용자 입력) |
| mean_speed_m_s | 0.0059 m/s |
| max_acceleration_m_s2 | 0.371 m/s² |
| max_crack_risk | 0.35 |

**이미지 설명**:
- X축 0.080~0.100 m, Y축 0.055~0.075 m 범위
- 속도·가속도 벡터가 굽힘 경로를 따라 배치
- max crack risk 지점에 주석 표시 (정상이므로 낮은 P(crack))
- 사용자 입력 스케일이 metadata를 오버라이드하여 일관된 SI 출력

---

### 1.3 비정상(크랙) 벤딩 (crack_01_full_crack)

**경로**: `exports/vectors/fpcb_test_suite/crack_01_full_crack/vector_map.png`

| 항목 | 값 |
|------|-----|
| scale | 1.00e-04 m/px |
| mean_speed_m_s | 0.0046 m/s |
| max_acceleration_m_s2 | **1.73 m/s²** |
| max_crack_risk | **0.98** |

**이미지 설명**:
- 아치형 패턴 내부에 **max crack risk 지점** 주석 (x≈0.092 m, y≈0.063 m)
- 가속도 스파이크로 인해 급격한 변화 구간 강조
- 정상 대비 max_acceleration 약 4.7배, max_crack_risk 0.98로 높은 이상 신호
- Px/m 역변환 정보로 픽셀–물리량 변환 검증 가능

---

## 2. 정상 vs 비정상 수치 비교

| 지표 | normal_001 | crack_01_full_crack | Delta |
|------|------------|---------------------|-------|
| mean_speed_m_s | 0.0059 | 0.0046 | -0.0013 |
| max_speed_m_s | 0.026 | 0.073 | +0.047 |
| max_acceleration_m_s2 | 0.37 | 1.73 | +1.36 |
| max_crack_risk | 0.35 | 0.98 | +0.63 |

**해석**: 크랙 시 max_acceleration가 크게 증가하고 max_crack_risk가 0.98로 높아, 이상 판정에 유리한 지표가 됨.

---

## 3. 다양한 사용자 분석 수요 시나리오 설계

### 시나리오 1: 정상 공정 검증 (QC)

**목적**: 라인별 정상 범위 내 동작 확인  
**입력**: 정상 200개 중 샘플 (normal_001~050)  
**절차**: Analyze → Scale 0.1 mm/px → Physics 모드  
**평가**: mean_speed_m_s, max_acceleration_m_s2가 사내 기준 내인지 확인  
**개선**: 기준값(threshold) GUI 입력, 초과 시 경고 표시

---

### 시나리오 2: 비정상(크랙) 탐지

**목적**: 크랙 발생 여부 및 위치 파악  
**입력**: 의심 샘플 (crack_01~10)  
**절차**: Analyze → Scale 0.1 mm/px → Physics 모드  
**평가**: max_crack_risk > 0.7이면 이상, 벡터 맵에서 crack risk 주석 위치 확인  
**개선**: crack risk 임계값 설정 UI, 이상 시 자동 하이라이트

---

### 시나리오 3: 정상 vs 비정상 비교 (Compare)

**목적**: 정상·비정상 간 차이 정량화  
**입력**: Base=normal_001, Candidate=crack_01  
**절차**: Compare 탭 → summary.json 선택 → Compare  
**평가**: delta_max_acceleration, delta_max_crack_risk 등  
**개선**: Delta 기준(예: +50% 이상) 초과 시 경고, 시각화 강화

---

### 시나리오 4: 크랙 발생 시점 탐지 (Change Point)

**목적**: 크랙이 발생한 프레임 구간 추정  
**입력**: crack_01_full_crack  
**절차**: Time Series Analysis → acceleration_max → CUSUM/Window-based  
**평가**: 변화점이 crack_center_ratio(≈72%) 근처 프레임에 있는지  
**개선**: 예상 구간 입력, 정확도(Recall@k) 리포트

---

### 시나리오 5: 스케일 검증 및 미세 조정

**목적**: 픽셀당 mm 값이 실제와 맞는지 확인  
**입력**: 동일 데이터, Scale 0.08 / 0.1 / 0.12 mm/px  
**절차**: 각 스케일로 분석 후 mean_speed_m_s, max_acceleration_m_s2 비교  
**평가**: 알려진 참조값(예: 레이저/캘리브레이션)과 일치도  
**개선**: 스케일 추천(예: 패널 길이 px 입력 → mm/px 자동 계산)

---

### 시나리오 6: 배치 분석 (다수 영상)

**목적**: 200개 정상 + 10개 크랙 일괄 분석  
**입력**: fpcb_test_suite 전체  
**절차**: CLI `analyze-bundle` 또는 GUI 반복  
**평가**: 정상/크랙 분류 ROC AUC, 처리 시간  
**개선**: 배치 모드 GUI, 결과 요약 테이블/차트

---

### 시나리오 7: ML 모델 학습 및 추론

**목적**: DREAM/PatchCore로 이상 탐지  
**입력**: 정상 200개(학습), 크랙 10개(검증)  
**절차**: ML & Optimization → normal/crack 선택 → Train → Analyze에서 추론  
**평가**: ROC AUC, PR AUC, F1  
**개선**: 학습 진행률, 검증 메트릭 실시간 표시

---

### 시나리오 8: 크랙 유형별 분석

**목적**: full_crack / mild_crack / snap_crack 구분  
**입력**: crack_01~10 (유형 혼합)  
**절차**: 각각 분석 후 max_crack_risk, curvature_concentration, acceleration 패턴 비교  
**평가**: 유형별 시그니처(예: shockwave 강도, 진동 지속 시간)  
**개선**: 유형 자동 분류, 유형별 임계값

---

## 4. 이미지 및 분석 결과 평가

### 4.1 강점

- **SI 단위 통일**: 축 X (m), Y (m), 속도 m/s, 가속도 km/s²
- **스케일 명시**: scale=… m/px로 변환 계수 표시
- **crack risk 시각화**: max crack risk 지점 주석
- **인덱스별 색상**: 포인트 추적 용이

### 4.2 개선 필요

| 항목 | 현재 | 개선 방향 |
|------|------|-----------|
| 가속도 단위 | km/s² (값이 작게 보임) | m/s² 또는 mm/s²로 통일 검토 |
| 주석 가독성 | 작은 폰트 | 폰트 크기/대비 조정 |
| 스케일 검증 | 수동 | 캘리브레이션/참조값 대조 기능 |
| 배치 결과 | 개별 파일 | 요약 대시보드/테이블 |

### 4.3 Scale 미세 조정 가이드

1. **참조값 활용**: 알려진 패널 길이(mm) ÷ 영상 내 px 길이 → mm/px
2. **다중 스케일 비교**: 0.08, 0.1, 0.12 mm/px로 분석 후 물리량 일관성 확인
3. **metadata 우선**: 합성 데이터는 metadata에 m/px 포함 → 실측 시 사용자 입력 우선
4. **단위 일관성**: mm/px 입력 시 m/px = mm/px × 1e-3 자동 변환

---

## 5. 더 나은 분석 결과를 위한 방법

### 5.1 단기 (1–2주)

- **스케일 추천**: 패널 길이(mm) + px 입력 → mm/px 자동 계산
- **임계값 UI**: crack_risk, max_acceleration 기준값 설정
- **Compare 시각화**: Delta 막대 그래프, 기준선 표시

### 5.2 중기 (1–2개월)

- **캘리브레이션 모드**: 체커보드/레이저 기반 mm/px 추정
- **배치 분석 GUI**: 폴더 선택 → 일괄 분석 → 요약 테이블
- **유형별 프로파일**: full/mild/snap crack 시그니처 DB 및 자동 분류

### 5.3 장기 (3개월+)

- **실제 데이터 검증**: 합성 vs 실측 성능 비교
- **Few-shot Fine-tuning**: 소량 크랙 데이터로 모델 보정
- **시각화 개선**: 3D 벡터장, 시계열 애니메이션

---

## 6. 중장기 개발 전략 문서 연계 (DEVELOPMENT_ROADMAP_FINAL 이어서)

### Phase D: 사용자 시나리오 기반 고도화 (신규, 4–6주)

**목표**: 다양한 분석 수요 시나리오 지원 및 출력 품질 개선

#### D.1 스케일 및 SI 단위 고도화 (1주)

- [ ] 스케일 추천: 패널 길이(mm) + px 입력 → mm/px 자동 계산
- [ ] 가속도 표시 단위 옵션: m/s², mm/s², km/s²
- [ ] 출력 이미지 주석 가독성 개선 (폰트, 대비)

#### D.2 임계값 및 기준값 UI (1주)

- [ ] crack_risk, max_acceleration 기준값 설정
- [ ] 초과 시 Summary/벡터 맵 경고 표시
- [ ] Compare 탭 Delta 기준선 및 경고

#### D.3 배치 분석 및 요약 (2주)

- [ ] 배치 분석 모드: 폴더 선택 → 일괄 분석
- [ ] 결과 요약 테이블 (CSV/Excel)
- [ ] 정상/비정상 분류 ROC·PR 곡선 (배치 결과 기반)

#### D.4 크랙 유형별 프로파일 (2주)

- [ ] full_crack / mild_crack / snap_crack 시그니처 분석
- [ ] 유형별 임계값/가중치 권장값
- [ ] 유형 자동 분류 (선택)

### Phase E: 실제 데이터 전환 준비 (대기 → 실제 데이터 확보 시)

- [ ] 캘리브레이션 도구 (체커보드/레이저)
- [ ] 합성 vs 실측 성능 비교 리포트
- [ ] Few-shot Fine-tuning (Phase C.1 연계)

---

## 7. 참고 문서

- `docs/DEVELOPMENT_ROADMAP_FINAL.md`: Phase A/B/C 로드맵
- `docs/SI_UNITS_AND_GUI_UPDATE_PLAN.md`: SI 단위 및 스케일 입력 설계
- `docs/GUI_TEST_SCENARIOS_REPORT.md`: GUI 테스트 시나리오
- `docs/PHASE_B_INSIGHTS.md`: Phase B 인사이트
