# S09: 앙상블 가중치 최적화
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/benchmark_phase_b_comprehensive.py
exit $LASTEXITCODE
