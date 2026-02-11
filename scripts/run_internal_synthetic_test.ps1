Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$Scenario = "normal",
    [string]$SessionName = "internal_sim_001"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Run bootstrap first."
}

$inputDir = "data/synthetic/$SessionName"
$outputDir = "exports/vectors/$SessionName"

Write-Host "==> 1/4 Generate synthetic data ($Scenario)"
& $PythonExe -m motionanalyzer.cli gen-synthetic `
    --scenario $Scenario `
    --output-dir $inputDir `
    --frames 120 `
    --points-per-frame 230 `
    --fps 30

Write-Host "==> 2/4 Validate synthetic signature"
& $PythonExe -m motionanalyzer.cli validate-synthetic `
    --input-dir $inputDir `
    --scenario $Scenario

Write-Host "==> 3/4 Analyze bundle"
& $PythonExe -m motionanalyzer.cli analyze-bundle `
    --input-dir $inputDir `
    --output-dir $outputDir

Write-Host "==> 4/4 Complete"
Write-Host "Summary: $outputDir/summary.json"
