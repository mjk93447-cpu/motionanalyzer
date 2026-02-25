# S01: Temporal 모델 개선
# Usage: .\scripts\agent_tools\run_scenario_S01.ps1

$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/validate_temporal_synthetic.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/benchmark_phase_b_comprehensive.py
exit $LASTEXITCODE
