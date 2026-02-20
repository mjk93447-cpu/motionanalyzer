<#
.SYNOPSIS
    Run EXE to analyze synthetic data and generate visualizations.

.DESCRIPTION
    Uses motionanalyzer-cli.exe to:
    1. Analyze normal and crack synthetic datasets
    2. Compare results
    3. Run full FPCB pipeline (generate -> analyze -> plot)

.EXAMPLE
    .\scripts\run_exe_synthetic_analysis.ps1
#>

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Set-Location $projectRoot

$exePath = Join-Path $projectRoot "dist\motionanalyzer-cli.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] EXE not found: $exePath"
    Write-Host "Please build first: .\scripts\build_exe.ps1"
    exit 1
}

Write-Host "==> Using EXE: $exePath"
Write-Host ""

# 1. Analyze normal synthetic data
Write-Host "[1/4] Analyzing normal synthetic data..."
& $exePath analyze-bundle `
    --input-dir "data\synthetic\examples\normal" `
    --output-dir "exports\vectors\exe_normal"
Write-Host "[OK] Normal analysis complete"
Write-Host ""

# 2. Analyze crack synthetic data
Write-Host "[2/4] Analyzing crack synthetic data..."
& $exePath analyze-bundle `
    --input-dir "data\synthetic\examples\crack" `
    --output-dir "exports\vectors\exe_crack"
Write-Host "[OK] Crack analysis complete"
Write-Host ""

# 3. Compare normal vs crack
Write-Host "[3/4] Comparing normal vs crack..."
$baseSummary = "exports\vectors\exe_normal\summary.json"
$candidateSummary = "exports\vectors\exe_crack\summary.json"
if ((Test-Path $baseSummary) -and (Test-Path $candidateSummary)) {
    & $exePath compare-runs `
        --base-summary $baseSummary `
        --candidate-summary $candidateSummary
    Write-Host "[OK] Compare complete"
} else {
    Write-Host "[WARN] Skipping compare (summary files not found)"
}
Write-Host ""

# 4. Run full FPCB pipeline (generate high-fidelity data, analyze, plot)
Write-Host "[4/4] Running FPCB pipeline (generate -> analyze -> plot)..."
& $exePath run-fpcb-pipeline `
    --data-dir "data\synthetic\fpcb_high_fidelity" `
    --export-vectors-dir "exports\vectors\fpcb_high_fidelity" `
    --plots-dir "exports\plots"
Write-Host "[OK] FPCB pipeline complete"
Write-Host ""

Write-Host "==> Analysis and visualization complete!"
Write-Host ""
Write-Host "Generated files:"
Write-Host "  - exports\vectors\exe_normal\vectors.csv, summary.json, vector_map.png"
Write-Host "  - exports\vectors\exe_crack\vectors.csv, summary.json, vector_map.png"
Write-Host "  - exports\vectors\fpcb_high_fidelity\vector_map.png"
Write-Host "  - exports\plots\fpcb_metrics.png"
Write-Host ""
