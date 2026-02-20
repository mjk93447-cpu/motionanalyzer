# FPCB Synthetic Model (Side View)

## 목적

OLED 공정의 FPCB 벤딩 모션을 현실과 유사하게 재현하는 합성 데이터 생성 기준을 정의한다.

## 공정 기반 가정

- 관찰 시점: side view
- 초기 형상: 약 230 px 수평 직선
- 종료 형상: 옆으로 눕는 U형에 가까운 고곡률 형상
- 프레임 순서: 실제 공정 시간 순서와 동일

## 물리 모델 요약

생성기는 패널 중심선을 1D 빔(rod)으로 근사한다.

1. 전체 벤딩 진행
   - 공정 구동 입력(drive)으로 총 회전각 `theta_total`이 시간에 따라 증가
   - 감쇠(damping)와 응답(response_alpha)로 실제 공정의 지연/완만함 반영
2. 국부 곡률 분포
   - 기본 곡률 `kappa0 = theta_total / L`
   - 손상/재료 편차는 곡률 가중치로 반영
3. 변형률 근사
   - `epsilon ~= t / (2R)` 사용
   - 코드에서는 `epsilon ~= 0.5 * thickness * kappa` 형태로 추정

## 시나리오 정의

- `normal`
  - 정상 공정. 마지막에 큰 벤딩각(거의 U형) 달성
- `crack`
  - 특정 구간의 강성 급감 -> 곡률 집중(힌지형 꺾임)
- `pre_damage`
  - 경미한 사전 손상 -> 비대칭 + 중간 수준 곡률 집중
- `thick_panel`
  - 두께 증가(유효 강성 증가) -> 최종 벤딩각 감소
- `uv_overcured`
  - UV 과경화로 초기 지연 + 후반 snap-like 급격 굽힘

## 생성 산출물

`motionanalyzer gen-synthetic` 실행 시:

- `frame_*.txt` : 프레임별 좌표
- `fps.txt` : FPS
- `frame_metrics.csv` : 프레임별 벤딩/곡률/추정변형률
- `metadata.json` : 시나리오/가정/파라미터

## 선검증 기준

`motionanalyzer validate-synthetic`은 시나리오별 특징을 자동 체크한다.

- 공통: 충분한 최종 벤딩각, 벤딩 진행의 단조성
- crack: 곡률 집중 지표 높음
- thick_panel: 최종 벤딩각/추정변형률 낮음
- uv_overcured: 후반 snap 시그니처(각도 2차차분 피크) 존재

## 참고 조사 포인트

모델 구성 시 아래 공학적 사실을 반영했다.

- FPCB 굽힘에서 중립축/반경 기반 변형률 근사 `epsilon = t/(2R)`
- 구리 회로는 굽힘 반복/고변형에서 균열 위험 증가
- 과경화(높은 가교도) 재료는 유연성 저하 및 brittle 거동 가능
