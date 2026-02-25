<#
.SYNOPSIS
    Prepare everything needed for local EXE testing.

.DESCRIPTION
    1. Ensures synthetic datasets exist (generates if missing)
    2. Builds GUI EXE (motionanalyzer-gui.exe)
    3. Builds CLI EXE (motionanalyzer-cli.exe)
    4. Creates standard export directories

.PARAMETER SkipBuild
    Skip EXE build (only prepare data and dirs)

.PARAMETER SkipData
    Skip synthetic data generation (use existing data)

.EXAMPLE
    .\scripts\prepare_exe_test.ps1
    .\scripts\prepare_exe_test.ps1 -SkipBuild
#>

Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$SkipBuild = $false,
    [switch]$SkipData = $false
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "==> EXE Local Test Preparation"
Write-Host "    Project root: $projectRoot"
Write-Host ""

# Resolve Python
$resolvedPython = $null
if ($PythonExe -match "[\\/]" -or $PythonExe -match "^\.+") {
    if (-not (Test-Path $PythonExe)) {
        Write-Host "[WARN] Python not found at '$PythonExe'. Using 'python' from PATH."
        $resolvedPython = "python"
    } else {
        $resolvedPython = $PythonExe
    }
} else {
    $resolvedPython = $PythonExe
}

# 1. Standard directories
Write-Host "[1/4] Creating export directories..."
$null = New-Item -ItemType Directory -Force -Path "exports\vectors", "exports\plots", "reports\compare", "logs"
Write-Host "      OK: exports\vectors, exports\plots, reports\compare"
Write-Host ""

# 2. Synthetic data (DS-260223-ml-fp-20k-60f)
if (-not $SkipData) {
    $dsCheck = "data\synthetic\ml_dataset_fp_focused\normal\normal_0001"
    if (-not (Test-Path "$dsCheck\frame_00001.txt")) {
        Write-Host "[2/4] DS-260223-ml-fp-20k-60f (ml_dataset_fp_focused) not found."
        Write-Host "      Generate with: python scripts/generate_ml_dataset.py --scale fp_focused --out data/synthetic/ml_dataset_fp_focused"
        Write-Host "      Or use DS-260220-ml-100k-100k-60f: data\synthetic\ml_dataset_100k_v2"
        exit 1
    }
    Write-Host "[2/4] Synthetic data exists: data\synthetic\ml_dataset_fp_focused (DS-260223-ml-fp-20k-60f)"
} else {
    Write-Host "[2/4] Skipping data check (-SkipData)"
}
Write-Host ""

# 3. Build GUI EXE
if (-not $SkipBuild) {
    Write-Host "[3/4] Building GUI EXE (motionanalyzer-gui.exe)..."
    & "$scriptDir\build_exe.ps1" -PythonExe $resolvedPython
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] GUI EXE build failed"
        exit 1
    }
    Write-Host "      OK: dist\motionanalyzer-gui.exe"
} else {
    if (Test-Path "dist\motionanalyzer-gui.exe") {
        Write-Host "[3/4] GUI EXE exists: dist\motionanalyzer-gui.exe"
    } else {
        Write-Host "[3/4] GUI EXE not found. Run: .\scripts\build_exe.ps1"
    }
}
Write-Host ""

# 4. Build CLI EXE
if (-not $SkipBuild) {
    Write-Host "[4/4] Building CLI EXE (motionanalyzer-cli.exe)..."
    & "$scriptDir\build_cli_exe.ps1" -PythonExe $resolvedPython
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] CLI EXE build failed"
        exit 1
    }
    Write-Host "      OK: dist\motionanalyzer-cli.exe"
} else {
    if (Test-Path "dist\motionanalyzer-cli.exe") {
        Write-Host "[4/4] CLI EXE exists: dist\motionanalyzer-cli.exe"
    } else {
        Write-Host "[4/4] CLI EXE not found. Run: .\scripts\build_cli_exe.ps1"
    }
}
Write-Host ""

Write-Host "==> Preparation complete!"
Write-Host ""
Write-Host "Next: Run tests (see docs\EXE_LOCAL_TEST_GUIDE.md)"
Write-Host "  .\scripts\run_exe_synthetic_analysis.ps1   # CLI batch test"
Write-Host "  .\dist\motionanalyzer-gui.exe             # GUI manual test"
Write-Host ""
