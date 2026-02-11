# FPCB Bending Physics/Chemistry Knowledge Base

## Purpose

OLED 공정의 FPCB 벤딩 모션 분석/합성데이터 생성/이상탐지를 위해, 개발 중 즉시 재사용 가능한 물리/화학 지식을 정리한다.

## 대상 구조

- 다층 유기필름(PI, adhesive, coverlay 등) + 구리 배선(copper trace) 복합체
- side-view 관측 기준으로 공정 중 형상 변화: 직선 -> 호 -> U 유사 형상
- 해석 대상: 좌표 궤적, 속도/가속도, 곡률, 국부 응력/변형률 surrogate

## Core Physics (Solid/Continuum)

- **Neutral axis**
  - 층구성/탄성계수 분포에 따라 중립축 이동
  - 재료 불균일/접착층 비대칭은 인장/압축 분포를 바꾼다
- **Curvature and strain**
  - 기본 근사: `epsilon ~ t / (2R)` 또는 `epsilon ~ z * kappa`
  - `kappa = 1/R`, `z`는 중립축으로부터 거리
- **Bending stiffness**
  - 등가 강성 `EI` 증가(두꺼움, 과경화)는 동일 구동에서 작은 곡률
  - 손상/균열은 국부적으로 강성을 낮추어 곡률 집중(hotspot) 유도
- **Rate and damping**
  - 실제 공정은 quasi-static에 가까우나, 구동 프로파일과 점탄성 때문에 지연/히스테리시스 존재
  - 시간 축에서 각도 증가율/가속도 피크를 품질지표로 사용 가능

## Core Chemistry/Materials (Organic + Interface)

- **PI/접착층의 가교도(crosslink density)**
  - UV/열 경화가 과도하면 탄성률 상승, 취성(brittleness) 증가, 연신율 저하
- **Interfacial adhesion**
  - 계면 결함은 벤딩 시 박리 또는 비정상 곡률 전이로 나타남
- **Copper trace fatigue/crack**
  - 반복 혹은 고변형 구간에서 국부 균열 위험 증가
  - 합성/실데이터 비교 시 곡률 집중과 궤적 불연속을 crack surrogate로 본다

## Failure Signatures to Model

- **Normal**: 부드러운 단조 벤딩, 과도한 곡률 집중 없음
- **Crack**: 특정 위치에서 급격한 힌지형 굽힘(곡률 집중 급증)
- **Pre-damage**: 약한 비대칭 + 중간 수준 곡률 집중
- **Thickened panel**: 전체적으로 잘 안 접힘(최종 각도 저하)
- **UV over-cured**: 초기 지연, 후반 급격한 snap-like 변형 가능

## Observable Metrics (from point trajectories)

- 프레임별 총 벤딩각 `theta(t)`
- 곡률분포 `kappa(s, t)` 및 집중도 `max(kappa)/mean(kappa)`
- 위치/속도/가속도 벡터 통계
- 변형률 surrogate `epsilon_hat(t)` (상대 지표)
- 시나리오별 판정 경계값(규칙 기반 또는 ML 기반)

## Practical Modeling Guidance for AI Coding

- 단순 랜덤 곡선 대신 `구동입력 + 응답 + 국부강성맵` 구조를 우선 사용
- 모델 파라미터는 시나리오별로 분리하고 메타데이터로 저장
- 생성 데이터는 항상 `frame_metrics.csv`, `metadata.json` 같이 남겨 재현 가능성 확보
- 검증은 "성공/실패"만이 아니라 어떤 물리 시그니처가 미달인지 메시지로 반환

## Limits and Cautions

- 현재 모델은 2D side-view surrogate이며 3D 응력장 FEA 대체가 아님
- 절대 응력/변형률 값은 재료 상수 측정치 없이 정량 신뢰를 주장하면 안 됨
- 사내 실데이터 기반 캘리브레이션 이후 threshold를 재조정해야 함
