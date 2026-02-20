# motionanalyzer 프로젝트 최종 목표

**작성일**: 2026년 2월 19일  
**역할**: 모든 개발·문서의 최상위 목표 정의

---

## 1. 핵심 원칙: 통합 NG 정의

**결정적 사실**: 크랙은 벤딩 과정에서 발생할 가능성이 가장 높다. 벤딩 중 크랙이 발생했든, 이미 크랙 징후가 있었든, **결과적으로 벤딩 후 크랙이 있는 패널은 모두 NG**로 정의하고 탐지해야 한다.

| 구분 | 설명 | 최종 결과 |
|------|------|-----------|
| 벤딩 중 크랙 | 과경화·과도한 궤적 등으로 벤딩 도중 균열 | NG |
| 이미 크랙된 패널 투입 | 손상된 패널이 벤딩 라인에 투입 | NG |

**개발 우선순위**: 우수한 결과를 얻는 데 집중하기 위해, **벤딩 중 크랙 감지(목표 1)**를 먼저 최대한 고도화한다.

---

## 2. 목표 1: 벤딩 중 크랙 감지 (최우선)

### 2.1 정의

벤딩 과정 **도중** 다양한 원인에 의해 발생하는 크랙을 **실시간에 가깝게** 감지하는 것이다.  
**최종 목표**: Precision-Recall Score 최대화.

### 2.2 감지 전략: 국소 + 전체 패턴 복합

탐지 정확도를 높이기 위해 **국소적 감지**와 **전체 패턴 감지**를 복합적으로 사용한다.

#### 국소적 감지 (Local)

크랙 발생 직전·직후의 **미세한 물리적 현상**:

| 감지 대상 | 설명 | 관련 특징 |
|-----------|------|-----------|
| **충격파(shockwave)** | 크랙 발생 시 가속도 스파이크 | acceleration_max, shockwave_amplitude |
| **진동(micro-vibration)** | 균열 후 미세 진동 | FFT, spectral_entropy, vibration_frequency |
| **크랙 부위 벌어짐** | 국소적 이격·길이 변화 | strain_surrogate, curvature_concentration |
| **직전/직후 속도 변화** | 균열 전후 순간적 속도 변화 | velocity, acceleration 시계열 |

#### 전체 패턴 감지 (Global)

크랙 징후·크랙 후 변화된 물성에 따른 **벡터 패턴**:

| 감지 대상 | 설명 | 관련 특징 |
|-----------|------|-----------|
| **크랙 징후 (과경화 등)** | UV over-cure로 인한 후반 스냅·충격파 | uv_delay_ratio, uv_snap_gain |
| **크랙 후 물성 변화** | 균열 후 달라진 궤적·곡률·가속도 분포 | bend_angle, curvature_concentration, 통계 특징 |
| **전체 시퀀스 이상 패턴** | 정상 대비 미묘한 시계열·공간 패턴 차이 | DREAM/PatchCore 기반 이상 점수 |

### 2.3 기술적 접근

- **CPD**: CUSUM, Window-based, PELT — 크랙 발생 시점 추정
- **DREAM / PatchCore**: 국소+전체 특징을 활용한 이상 점수 — Precision-Recall 최대화
- **Temporal 모델**: LSTM/GRU — 시계열 구조 보존
- **특징**: crack_risk, strain surrogate, impact surrogate, curvature_concentration, spectral_entropy

### 2.4 개발 우선순위

1. **DREAM·PatchCore를 목표 1에 적극 활용** — 벤딩 중 크랙 vs 정상 구분, PR 최대화
2. 국소 특징 강화 — 충격파·진동·곡률 집중
3. 전체 패턴 특징 — 과경화 징후, 크랙 후 물성 반영
4. CPD 정확도 — 시점 추정 보조
5. 앙상블 — 국소+전체·다중 모델 결합

---

## 3. 목표 2: 이미 크랙된 패널 감지 (보조)

### 3.1 정의

**이미 손상된 FPCB**가 벤딩 라인에 투입되는 경우, 손상으로 인한 물성·구조 차이가 **전체 궤적·물리 패턴**에 미묘한 차이를 만든다. 이 차이를 감지한다.  
최종적으로 역시 **NG**로 분류된다.

### 3.2 특성

- 전체 궤적·패턴 기반 (국소가 아닌 전역)
- DREAM, PatchCore, Ensemble로 미세 패턴 구분

### 3.3 개발 우선순위

목표 1에서 우수한 결과를 확보한 후, 목표 2 고도화를 진행한다.

---

## 4. 우선순위에 따른 개발 방향

### 4.1 목표 1 최우선 (벤딩 중 크랙)

1. **DREAM·PatchCore**: normal vs crack_in_bending — **Precision-Recall 최대화**
2. **국소+전체 복합**: 충격파·진동·곡률 집중 + 전체 시퀀스 이상 패턴
3. **CPD**: 크랙 시점 추정 정확도
4. **Temporal**: 시계열 모델 보완

### 4.2 목표 2 다음 (이미 크랙된 패널)

1. DREAM/PatchCore: pre_damaged vs normal
2. 앙상블·고급 특징

---

## 5. 문서 연계

- `README.md`: 프로젝트 개요에 본 목표 반영
- `DEVELOPMENT_STRATEGY_AND_WORK_ORDER.md`: Phase별 우선순위, PR 최대화 전략
- `docs/SYNTHETIC_DATA_SPEC.md`: 합성 규칙·시나리오
- `.cursor/rules/fpcb-domain-knowledge.mdc`: 도메인 가이드라인
