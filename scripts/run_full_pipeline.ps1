# 100k 데이터셋 구축 및 ML 분석 전체 파이프라인
# 실행: .\scripts\run_full_pipeline.ps1

$ErrorActionPreference = "Stop"
$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv-gpu\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "=== MotionAnalyzer 100k Pipeline ===" -ForegroundColor Cyan
Write-Host "Python: $python"

# 1. 100k 데이터 생성 (4 workers)
Write-Host "`n[1/4] Generating 100k dataset (workers=4)..." -ForegroundColor Yellow
& $python scripts/generate_ml_dataset.py --scale 100k --workers 4
if ($LASTEXITCODE -ne 0) { throw "Dataset generation failed" }

# 2. ML 분석 (DREAM + PatchCore)
Write-Host "`n[2/4] Running crack detection analysis..." -ForegroundColor Yellow
& $python scripts/analyze_crack_detection.py
if ($LASTEXITCODE -ne 0) { throw "Analysis failed" }

# 3. 논문 리포트 재생성
Write-Host "`n[3/4] Regenerating paper report..." -ForegroundColor Yellow
& $python scripts/generate_final_report_docx.py
if ($LASTEXITCODE -ne 0) { throw "Report generation failed" }

Write-Host "`n=== Pipeline 완료 ===" -ForegroundColor Green
Write-Host "Output: reports/deliverables/FPCB_Crack_Detection_Final_Report.docx"
Write-Host "        reports/crack_detection_analysis/"
