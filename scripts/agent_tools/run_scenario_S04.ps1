# S04: 고급 특징 과적합 관리
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/analyze_advanced_features_overfitting.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/validate_advanced_features.py
exit $LASTEXITCODE
