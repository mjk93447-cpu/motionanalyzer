# GPU 활용 전략 (RTX 2070 Super · Windows exe)

**작성일**: 2026년 2월 19일  
**배경**: 오피스 PC(RTX 2070 Super)에서 CUDA로 ML 가속, 보안 네트워크로 Cursor 불가 → **Windows exe 내에서 모든 ML 작업 수행** 필요.

---

## 전략 변경 (2026-02-19)

### 우선순위 조정

| 단계 | 시점 | 작업 |
|------|------|------|
| **1** | **실제 데이터 확보 전** | 로컬에서 합성데이터 규모 조절하여 크랙검출 성능 검증 |
| **2** | **실제 데이터 확보 후** | GPU 사용 부분 개발 (RTX 2070 Super, exe 내 CUDA) |

- **현재**: 로컬 CPU·합성데이터로 confusion matrix, 벡터맵, 인사이트 도출
- **실제 데이터 확보 시**: GPU·exe·오피스 PC 워크플로우 본격화

---

## 1. RTX 2070 Super 호환성 검토

### 1.1 스펙

| 항목 | RTX 2070 Super | RTX 3060 (참고) |
|------|----------------|-----------------|
| 아키텍처 | Turing (TU104) | Ampere |
| CUDA 코어 | 2,560 | 3,584 |
| VRAM | 8GB GDDR6 | 12GB GDDR6 |
| Compute Capability | 7.5 | 8.6 |
| CUDA 지원 | 10.2+ | 11.x, 12.x |

### 1.2 ML 작업 가능 여부

**결론: RTX 2070 Super로 CUDA 기반 ML 작업을 충분히 빠르게 수행할 수 있다.**

| 작업 | VRAM 사용 | RTX 2070 Super |
|------|-----------|----------------|
| DREAM (작은 AE + Discriminator) | ~100–300MB | ✅ 충분 |
| PatchCore (sklearn k-NN, CPU) | - | ✅ CPU로 실행 |
| Temporal (LSTM/GRU) | ~200–500MB | ✅ 충분 |
| 합성 데이터 1100건 | CPU | ✅ 병렬화로 가속 |

- DREAM: 입력 차원 50–100, hidden 64→32→16, latent 8 수준의 소형 모델 → 8GB VRAM의 1% 미만
- PyTorch CUDA 11.8/12.1 빌드와 호환 (Compute Capability 7.5 지원)

### 1.3 권장 사항 변경

- **이전**: RTX 3060 이상 권장
- **현재**: **RTX 2070 Super로 충분** (DREAM·Temporal 등 현재 모델 규모 기준)

---

## 2. 오피스 PC 제약 및 exe 전략

### 2.1 제약

| 항목 | 내용 |
|------|------|
| 네트워크 | 보안 네트워크만 사용 가능 |
| Cursor | 사용 불가 |
| 개발 환경 | 로컬에서 코드 편집 후 exe 빌드·배포 |

### 2.2 핵심 요구사항

**Windows exe 프로그램 내에서 모든 ML 작업이 가능해야 함.**

- 합성 데이터 생성
- DREAM·PatchCore·Temporal 학습
- CPD (Change Point Detection)
- 평가·리포트 생성

### 2.3 현재 desktop_gui (exe 후보) 기능

| 기능 | GUI 탭 | exe 내 가능 |
|------|--------|-------------|
| DREAM 학습 | ML & Optimization | ✅ |
| PatchCore 학습 | ML & Optimization | ✅ |
| Ensemble 학습 | ML & Optimization | ✅ |
| Temporal 학습 | ML & Optimization | ✅ |
| CPD (CUSUM/Window/PELT) | Time Series Analysis | ✅ |
| 분석 (Physics/DREAM/PatchCore) | Analyze | ✅ |
| **합성 데이터 생성** | **없음** | ❌ 추가 필요 |
| **목표별 평가 파이프라인** | **없음** | ❌ 추가 필요 |

### 2.4 exe 내 전체 ML 가능 여부

**가능하다.** 다음을 GUI에 추가하면 된다.

1. **합성 데이터 생성**: `generate_ml_dataset.py` 로직을 GUI 탭 또는 메뉴로 노출
2. **목표 평가**: `evaluate_goal1_ml.py`, `evaluate_goal2_ml.py` 로직을 GUI에서 실행
3. **DREAM device**: `torch.cuda.is_available()` 시 `device="cuda"` 사용

---

## 3. PyInstaller + PyTorch CUDA exe

### 3.1 가능 여부

**가능하다.** 단, 다음 조건이 필요하다.

| 조건 | 설명 |
|------|------|
| 대상 PC | NVIDIA GPU + 최신 드라이버 (CUDA runtime 포함) |
| PyTorch | `torch` + CUDA 빌드 (`cu118` 또는 `cu121`) |
| PyInstaller | `--hidden-import`로 torch 하위 모듈 명시 |

### 3.2 주의사항

- CUDA DLL은 exe에 번들하지 않고, **시스템에 설치된 NVIDIA 드라이버**를 사용
- 오피스 PC에 RTX 2070 Super가 있으면 드라이버는 일반적으로 설치되어 있음
- exe 빌드는 **CUDA가 설치된 개발 PC**에서 수행

### 3.3 빌드 예시

```bash
# CUDA 지원 PyTorch 설치 후
pip install pyinstaller
pyinstaller --onefile --windowed ^
  --hidden-import=torch ^
  --hidden-import=torch._C ^
  -n motionanalyzer_gui ^
  src/motionanalyzer/desktop_gui.py
```

---

## 4. 전략 요약

### 4.1 오피스 PC (RTX 2070 Super) 활용

| 단계 | 작업 | 환경 |
|------|------|------|
| 1 | exe 빌드 (로컬/CI) | Cursor 사용 가능한 PC |
| 2 | exe 배포 | USB, 내부망 등 |
| 3 | 오피스 PC에서 exe 실행 | 보안 네트워크, Cursor 불가 |
| 4 | GUI에서 합성 데이터 생성·학습·평가 | exe 내부, GPU 자동 사용 |

### 4.2 코드 수정 사항

| # | 작업 | 목적 |
|---|------|------|
| 1 | DREAM device 자동 감지 | `cuda` 사용 가능 시 GPU 사용 |
| 2 | GUI에 "합성 데이터 생성" 기능 | exe 내 데이터 생성 |
| 3 | GUI에 "목표 평가" 기능 | exe 내 평가 파이프라인 |
| 4 | PyInstaller spec (PyTorch+CUDA) | exe 빌드 설정 |

### 4.3 VM/원격 전략 (선택)

- Cursor 사용 가능한 PC에서 개발
- 오피스 PC에는 **exe만 배포**하여 ML 실행
- VM/원격 서버는 **개발·대규모 실험**용으로만 사용

---

## 5. 문서 연계

- `GPU_TODO_LIST.md`: exe 내 ML·GUI 확장 TODO
- `DEVELOPMENT_STRATEGY_AND_WORK_ORDER.md`: Phase에 exe·GPU 반영
