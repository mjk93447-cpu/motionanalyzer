# S02: CPD 정확도 향상
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/validate_cpd_optimization.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/evaluate_goal1_cpd.py
exit $LASTEXITCODE
