# Jupyter + GPU 환경 설정 가이드

**목적**: Cursor IDE에서 Jupyter 노트북을 사용하여 GPU 가속 ML 파이프라인 실행

---

## 1. 사전 요구사항

- **NVIDIA GPU** (RTX 시리즈 권장)
- **CUDA 12.x** 드라이버 설치
- **Python 3.11+**

---

## 2. 가상환경 및 패키지 설치

### 2.1 가상환경 생성

```powershell
cd c:\motionanalyzer
python -m venv .venv-gpu
.\.venv-gpu\Scripts\Activate.ps1
```

### 2.2 패키지 설치 (GPU)

```powershell
# motionanalyzer + ML 의존성
pip install -e ".[ml]"

# PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Jupyter
pip install -r requirements-gpu.txt
```

### 2.3 Jupyter 커널 등록

```powershell
python -m ipykernel install --user --name motionanalyzer-gpu --display-name "Python (motionanalyzer GPU)"
```

---

## 3. Cursor IDE에서 Jupyter 사용

### 3.1 확장 설치

1. Cursor에서 `Ctrl+Shift+X` (확장)
2. "Jupyter" 검색 후 **Jupyter** (Microsoft) 설치
3. 필요 시 **Python** 확장도 설치

### 3.2 커널 선택

1. `notebooks/00_setup_gpu_env.ipynb` 열기
2. 우측 상단 "Select Kernel" 클릭
3. **Python (motionanalyzer GPU)** 선택

### 3.3 GPU 검증

`00_setup_gpu_env.ipynb`의 셀을 순서대로 실행하여 다음을 확인:

- `torch.cuda.is_available()` → `True`
- `torch.cuda.get_device_name(0)` → GPU 이름 출력

---

## 4. 노트북 구조

| 노트북 | 용도 |
|--------|------|
| `00_setup_gpu_env.ipynb` | GPU 환경 검증 |
| `04_full_pipeline_100k.ipynb` | 100k 데이터 전체 파이프라인 (특징 추출 → DREAM/PatchCore 학습) |

---

## 5. 트러블슈팅

### CUDA를 찾을 수 없음

- NVIDIA 드라이버가 CUDA 12.x와 호환되는지 확인
- `nvidia-smi`로 GPU 인식 여부 확인

### Jupyter 커널이 보이지 않음

```powershell
jupyter kernelspec list
# motionanalyzer-gpu가 목록에 있어야 함
```

### 메모리 부족

- `--max-train` 옵션으로 학습 샘플 수 제한
- `batch_size` 축소 (64 → 32)
