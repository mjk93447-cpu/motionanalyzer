# S05: 전체 파이프라인 (100k)
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root
& "$root\scripts\run_full_pipeline.ps1"
exit $LASTEXITCODE
