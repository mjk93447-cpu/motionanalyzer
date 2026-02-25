# S08: QA 게이트 검증
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = "$root\.venv-gpu\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv-gpu required" }
Set-Location $root
& $py scripts/evaluate_synthetic_dataset_quality.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/validate_enhanced_dream.py
exit $LASTEXITCODE
