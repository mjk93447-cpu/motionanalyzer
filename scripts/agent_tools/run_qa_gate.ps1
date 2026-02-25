# QA 게이트 일괄 실행 (개발 단위 시작 전)
# 사용: .\scripts\agent_tools\run_qa_gate.ps1

$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = "$root\.venv-gpu\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv-gpu 필요. setup_gpu_env.ps1 실행" }
Set-Location $root
& $py scripts/evaluate_synthetic_dataset_quality.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/validate_enhanced_dream.py
exit $LASTEXITCODE
