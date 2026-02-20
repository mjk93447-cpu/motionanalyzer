# 합성 데이터 규격 및 규칙

**작성일**: 2026년 2월 19일  
**용도**: ML 학습·평가용 합성 데이터 생성 규칙, 태그, 시나리오 정의  
**목표**: 벤딩 중 크랙 감지 Precision-Recall 최대화 (목표 1 최우선)

---

## 1. 데이터 규모

| 클래스 | 개수 | 목표 | 시나리오 |
|--------|------|------|----------|
| **normal** | 1000 | - | normal |
| **crack_in_bending** | 80 | 목표 1 (벤딩 중 크랙) | crack (50), uv_overcured (30) |
| **pre_damaged_panel** | 20 | 목표 2 (이미 크랙된 패널) | pre_damage (20) |
| **thick_panel** | 20 | 변형 (경계 케이스) | thick_panel (20) |

**총 1120건**, 각 60프레임, 30 fps, 2초.

---

## 2. 시나리오 매핑

| 시나리오 | 목표 | 물리 특성 |
|----------|------|-----------|
| normal | - | 정상 범위 내 변동 |
| crack | 목표 1 | 충격파, 진동, 곡률 집중 |
| uv_overcured | 목표 1 | 과경화, 후반 스냅, 충격파 |
| pre_damage | 목표 2 | 미묘한 물성 차이, 전체 궤적 변화 |

---

## 3. 태그 스키마 (metadata.json)

```json
{
  "goal": "goal1" | "goal2" | "normal",
  "scenario": "normal" | "crack" | "uv_overcured" | "pre_damage",
  "label": 0 | 1,
  "crack_frame": -1 | int,
  "split": "train" | "val" | "test",
  "dataset_id": "normal_0001" | "crack_0001" | "predam_0001"
}
```

- **goal**: 목표 연계 (goal1=벤딩 중 크랙, goal2=이미 크랙된 패널)
- **label**: 0=정상, 1=비정상
- **crack_frame**: 크랙 발생 프레임 (목표 1만 유효, -1은 해당 없음)
- **split**: train 70%, val 15%, test 15% (클래스별 비율 유지)

---

## 4. 합성 규칙 (변경 시 재생성)

### 4.1 Normal

- points_per_frame: 200–260
- noise_std: 0.12–0.38
- panel_length_px: 210–250
- pixels_per_mm: 8–12

### 4.2 Crack (목표 1) — 국소+전체 감지용

- scenario: crack
- crack_gain > 0, **shockwave, vibration** (국소)
- crack_center_ratio: 0.65–0.80 (변동)
- crack_frame ≈ int(crack_center_ratio * (frames-1))
- 크랙 부위 벌어짐, 곡률 집중

### 4.3 UV over-cured (목표 1) — 크랙 징후(과경화)

- scenario: uv_overcured
- uv_delay_ratio, uv_snap_gain
- 후반 스냅 → 충격파 (전체 패턴)

### 4.4 Pre-damage (목표 2)

- scenario: pre_damage
- crack_gain 낮음, pre_damage_skew
- 전체 궤적·물성 미묘 차이

---

## 5. manifest.json

```json
{
  "version": "1.0",
  "created_at": "ISO8601",
  "total_count": 1100,
  "normal": 1000,
  "crack_in_bending": 80,
  "pre_damaged_panel": 20,
  "splits": { "train": 770, "val": 165, "test": 165 },
  "entries": [
    {
      "path": "normal/normal_0001",
      "goal": "normal",
      "label": 0,
      "split": "train"
    }
  ]
}
```
