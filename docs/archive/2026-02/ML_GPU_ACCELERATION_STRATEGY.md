# ML GPU 가속화 중기 전략

**목표**: 합성 데이터 생성 및 분석 파이프라인의 속도를 비약적으로 개선하여 100k 데이터셋을 실용적 시간 내에 처리

**최종 갱신**: 2026-02-20

---

## 1. 현재 병목 분석

| 단계 | 소요 시간 (추정) | 병목 원인 |
|------|------------------|-----------|
| **합성 데이터 생성** (100k) | 수 시간 | 단일 프로세스, 순차 루프, CPU 전용 |
| **특징 추출** (prepare_training_data) | 30분~1시간 | 데이터셋별 순차 로드, FFT/통계 연산 |
| **DREAM 학습** | 20~40분 | PyTorch CPU 기본, batch_size=32 |
| **PatchCore 학습** | 10~20분 | scikit-learn k-NN, CPU 전용 |
| **총 파이프라인** | 4~6시간+ | — |

---

## 2. 중기 전략 로드맵

### Phase 1: Jupyter + GPU 환경 구축 (1~2일)

1. **Jupyter Notebook 설치**
   - Cursor IDE에 Jupyter 확장 활성화
   - `notebooks/` 디렉터리 생성 및 커널 설정

2. **GPU 가상환경 구성**
   - Python 3.11 + CUDA 12.x 호환 PyTorch
   - `pip install torch --index-url https://download.pytorch.org/whl/cu121`
   - `pip install jupyter ipykernel`

3. **환경 검증**
   - `torch.cuda.is_available()` 확인
   - 간단한 GPU 벤치마크 실행

### Phase 2: 데이터 생성 병렬화 (2~3일)

1. **멀티프로세싱 적용**
   - `generate_ml_dataset.py`에 `--workers N` 옵션
   - `concurrent.futures.ProcessPoolExecutor`로 샘플별 병렬 생성
   - 100k 생성: 4~8 workers → 예상 1/4~1/8 시간

2. **배치 단위 생성**
   - 샘플을 청크로 나누어 worker에 분배
   - 메모리 사용량 제어

### Phase 3: 특징 추출 병렬화 (2~3일)

1. **병렬 load_dataset + extract_features**
   - `prepare_training_data`에 `n_jobs` 파라미터
   - `joblib.Parallel` 또는 `ProcessPoolExecutor` 적용
   - 100k 샘플: 8 workers → 예상 1/6~1/8 시간

2. **캐싱**
   - 추출된 특징을 Parquet/feather로 저장
   - 재실행 시 캐시 로드

### Phase 4: DREAM GPU 최적화 (1~2일)

1. **명시적 CUDA 사용**
   - `device="cuda"` 강제 (가능 시)
   - `batch_size` 증가 (64~256)

2. **DataLoader 최적화**
   - `num_workers`, `pin_memory` 설정
   - Mixed precision (AMP) 검토

### Phase 5: PatchCore 가속 (선택, 1~2일)

1. **FAISS-GPU** (k-NN 대체)
   - `faiss-gpu`로 메모리 뱅크 검색 가속
   - 대규모 데이터셋에서 10x+ 속도 향상

2. **cuML** (RAPIDS)
   - `cuml.neighbors.NearestNeighbors` 사용
   - NVIDIA GPU 필요

---

## 3. Jupyter 노트북 구조

```
notebooks/
├── 00_setup_gpu_env.ipynb      # GPU 환경 검증
├── 01_generate_ml_dataset.ipynb # 병렬 데이터 생성 (Phase 2)
├── 02_extract_features.ipynb   # 병렬 특징 추출 (Phase 3)
├── 03_train_dream_patchcore.ipynb # DREAM/PatchCore 학습 (Phase 4)
└── 04_full_pipeline_100k.ipynb # 100k 전체 파이프라인
```

---

## 4. 예상 효과

| 단계 | 현재 | 목표 | 개선 |
|------|------|------|------|
| 데이터 생성 100k | ~4h | ~45min | ~5x |
| 특징 추출 100k | ~1h | ~10min | ~6x |
| DREAM 학습 | ~30min | ~5min | ~6x |
| PatchCore 학습 | ~15min | ~3min | ~5x |
| **총 파이프라인** | **~6h** | **~1h** | **~6x** |

---

## 5. 의존성 (GPU 환경)

```txt
# requirements-gpu.txt (또는 pyproject.toml [project.optional-dependencies].gpu)
torch>=2.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
joblib>=1.3.0
# CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. 참고

- [MASTER_RULES.md](../MASTER_RULES.md): 문서 계층
- [fpcb-domain-knowledge.mdc](../.cursor/rules/fpcb-domain-knowledge.mdc): FPCB 도메인 가드레일
