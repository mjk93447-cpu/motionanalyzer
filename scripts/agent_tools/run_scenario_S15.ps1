# S15: 논문 재현성 검증 (벤치마크 + 파이프라인)
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/benchmark_phase_b_comprehensive.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Benchmark OK. Full pipeline optional: .\scripts\run_full_pipeline.ps1"
exit 0
