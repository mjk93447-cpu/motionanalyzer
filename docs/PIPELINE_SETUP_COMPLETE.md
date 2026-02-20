# GPU 파이프라인 완전 세팅 가이드

**최종 갱신**: 2026-02-20

---

## 1. 한 번에 세팅 (권장)

```powershell
cd c:\motionanalyzer
.\scripts\setup_gpu_env.ps1
```

이 스크립트는 다음을 수행합니다:
- `.venv-gpu` 가상환경 생성
- motionanalyzer + ML 의존성 설치
- PyTorch (CUDA 12.1) 설치
- Jupyter, ipykernel, joblib 설치
- Jupyter 커널 `motionanalyzer-gpu` 등록

---

## 2. 수동 세팅

### 2.1 가상환경

```powershell
cd c:\motionanalyzer
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
```

### 2.2 패키지 설치

```powershell
pip install -e ".[ml]"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
python -m ipykernel install --user --name motionanalyzer-gpu --display-name "Python (motionanalyzer GPU)"
```

### 2.3 GPU 검증

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## 3. 100k 파이프라인 실행

### 3.1 전체 파이프라인 (한 번에)

```powershell
.\scripts\run_full_pipeline.ps1
```

### 3.2 단계별 실행

```powershell
# 1. 100k 데이터 생성 (4 workers)
python scripts/generate_ml_dataset.py --scale 100k --workers 4

# 2. ML 분석 (DREAM + PatchCore)
python scripts/analyze_crack_detection.py

# 3. 논문 리포트 재생성
python scripts/generate_final_report_docx.py
```

---

## 4. Jupyter 노트북

1. Cursor에서 `notebooks/00_setup_gpu_env.ipynb` 열기
2. 커널 선택: **Python (motionanalyzer GPU)**
3. 100k 파이프라인: `notebooks/04_full_pipeline_100k.ipynb`

---

## 5. 예상 소요 시간

| 단계 | 100k (workers=4) | 비고 |
|------|------------------|------|
| 데이터 생성 | ~45~60분 | normal 75k 병렬 |
| 특징 추출 | ~15~30분 | 순차 |
| DREAM 학습 | ~5~15분 | GPU 시 batch_size=128 |
| PatchCore | ~5~10분 | CPU |
| **총** | **~1~2시간** | |

---

## 6. 출력 경로

- 데이터: `data/synthetic/ml_dataset/`
- 분석: `reports/crack_detection_analysis/`
- 논문: `reports/deliverables/FPCB_Crack_Detection_Final_Report.docx`
