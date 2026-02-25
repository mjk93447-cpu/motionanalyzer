# S10: Goal1/Goal2 ML 평가
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/evaluate_goal1_ml.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/evaluate_goal2_ml.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/evaluate_goals_summary.py
exit $LASTEXITCODE
