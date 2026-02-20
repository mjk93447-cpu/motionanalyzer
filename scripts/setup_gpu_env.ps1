# GPU 환경 완전 세팅 스크립트
# 실행: .\scripts\setup_gpu_env.ps1

$ErrorActionPreference = "Stop"
$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Set-Location $repoRoot

Write-Host "=== MotionAnalyzer GPU 환경 세팅 ===" -ForegroundColor Cyan
Write-Host "Repo: $repoRoot"

# 1. 가상환경 생성
$venvPath = Join-Path $repoRoot ".venv-gpu"
if (-not (Test-Path $venvPath)) {
    Write-Host "`n[1/5] Creating venv .venv-gpu..." -ForegroundColor Yellow
    python -m venv $venvPath
} else {
    Write-Host "`n[1/5] venv .venv-gpu exists" -ForegroundColor Green
}

# 2. 활성화 및 motionanalyzer 설치
Write-Host "`n[2/5] Installing motionanalyzer + ML deps..." -ForegroundColor Yellow
& "$venvPath\Scripts\pip.exe" install -e ".[ml]" -q

# 3. PyTorch CUDA
Write-Host "`n[3/5] Installing PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
& "$venvPath\Scripts\pip.exe" install torch --index-url https://download.pytorch.org/whl/cu121 -q

# 4. GPU requirements
Write-Host "`n[4/5] Installing Jupyter, joblib..." -ForegroundColor Yellow
& "$venvPath\Scripts\pip.exe" install -r requirements-gpu.txt -q

# 5. Jupyter kernel 등록
Write-Host "`n[5/5] Registering Jupyter kernel..." -ForegroundColor Yellow
& "$venvPath\Scripts\python.exe" -m ipykernel install --user --name motionanalyzer-gpu --display-name "Python (motionanalyzer GPU)"

# 검증
Write-Host "`n=== GPU 검증 ===" -ForegroundColor Cyan
& "$venvPath\Scripts\python.exe" -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"@

Write-Host "`n=== 세팅 완료 ===" -ForegroundColor Green
Write-Host "활성화: .\.venv-gpu\Scripts\Activate.ps1"
Write-Host "100k 생성: python scripts/generate_ml_dataset.py --scale 100k --workers 4"
Write-Host "분석: python scripts/analyze_crack_detection.py"
