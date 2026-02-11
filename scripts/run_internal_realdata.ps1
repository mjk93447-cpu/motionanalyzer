Param(
    [string]$ExePath = ".\dist\motionanalyzer-cli.exe",
    [string]$InputDir = "data/raw/session_real_001",
    [string]$OutputDir = "exports/vectors/real_session_001",
    [string]$BaselineSummary = "exports/vectors/baseline/summary.json",
    [string]$RunLogTxt = "internal_eval/logs/internal_run_latest.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    throw "Executable not found at '$ExePath'. Build or download exe first."
}

& $ExePath internal-realdata-run `
    --input-dir $InputDir `
    --output-dir $OutputDir `
    --baseline-summary $BaselineSummary `
    --run-log-txt $RunLogTxt

Write-Host "Internal realdata run finished. Log: $RunLogTxt"
