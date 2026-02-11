Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$InputDir = "data/raw/session_real_001"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Run bootstrap first."
}

Write-Host "==> Step 1/3: Prepare internal evaluation folders and templates"
& $PythonExe -m motionanalyzer.cli prepare-internal

Write-Host "==> Step 2/3: Run real-data preflight checks"
& $PythonExe -m motionanalyzer.cli preflight-realdata `
    --input-dir $InputDir `
    --report-path "reports/preflight/internal_preflight_latest.json"

Write-Host "==> Step 3/3: Done"
Write-Host "Use internal_eval/logs/result_template.csv for redacted sharing logs."
