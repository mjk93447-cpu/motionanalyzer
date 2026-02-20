# ver7 Dev Plan - EXE Functionality Upgrade

Last updated: 2025-02-12

## 목표

GitHub Actions에서 생성되는 exe 파일의 기능을 개선하여, Run Analysis 실행 시 벡터맵 시각화를 GUI에 표시하고, 결과 파일에 물리학 단위를 포함한 정확한 분석 이미지를 추가한다.

## 주요 업데이트 항목

### 1. GUI 벡터맵 표시

- Run Analysis 실행 시 결과가 벡터맵 형태로 GUI에 시각화되어 표시
- 줌이 자유자재로 가능한 인터랙티브 뷰어
- 수천 개의 점과 화살표를 겹침 없이 구분 가능하도록 투명도 및 시각화 최적화

### 2. 결과 파일에 이미지 추가

- 입력 프레임 점좌표를 하나의 2D 그래프에 표시
- 프레임당 이동변위 및 시간간격(FPS 기반)으로 속도 벡터 계산 → 화살표 plotting
- 프레임별 시간당 속도 벡터 변화로 가속도 벡터 계산 → 속도와 구분되는 형태의 화살표 plotting
- 이미지 크기 최대화, 점과 화살표는 최대한 가늘게

### 3. 물리학 데이터 처리

- FPS로부터 프레임당 시간차(초, s) 계산
- 속도: px/s (픽셀/초)
- 가속도: px/s² (픽셀/초²)
- 결과값에 단위 명시

### 4. 충격·크랙 분석 표현

- 가장 큰 충격이 발생한 것으로 판단되는 부분을 이미지에 작게 표시
- 이해를 돕는 수치 및 크랙 예상 확률 표현

## 기술 스펙

| 항목 | 내용 |
|------|------|
| 좌표계 | 원본 입력 txt 파일의 점좌표 그대로 (x, y px) |
| 속도 단위 | px/s |
| 가속도 단위 | px/s² |
| 시간 간격 | dt = 1 / fps (초) |
| 시각화 | velocity arrow + acceleration arrow (구분 가능한 스타일) |
| 줌 | matplotlib NavigationToolbar 또는 동등한 줌/팬 지원 |

## 완료 조건

- [x] GUI에서 Run Analysis 후 벡터맵 이미지 표시
- [x] 줌/팬 가능한 뷰어 (matplotlib NavigationToolbar2Tk)
- [x] export 시 vector_map.png 생성
- [x] velocity / acceleration 화살표 구분 (색상·두께 상이)
- [x] max impact 영역 및 crack probability 주석
- [x] summary에 단위 표시 (mean_speed_px_s, max_acceleration_px_s2 등)
- [ ] 버전업 후 main 브랜치 푸시, GitHub Actions exe 빌드
