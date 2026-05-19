# Build turnkey ML GUI: balanced pretrain in release/models + bundled EXE + models zip.
Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$SkipTrain,
    [switch]$SkipExe
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$modelsDir = Join-Path $root "release\models"
$required = @("draem_model.pt", "patchcore_model.npz", "bundle_manifest.json")

if (-not $SkipTrain) {
    foreach ($n in $required) {
        if (-not (Test-Path (Join-Path $modelsDir $n))) {
            Write-Host "==> Training balanced pretrain (missing $n)"
            & $PythonExe scripts/train_balanced_pretrain.py
            break
        }
    }
}

foreach ($n in $required) {
    if (-not (Test-Path (Join-Path $modelsDir $n))) {
        throw "Missing release/models/$n — run: python scripts/train_balanced_pretrain.py"
    }
}

if (-not $SkipExe) {
    Write-Host "==> Building ML GUI EXE with bundled models"
    & "$PSScriptRoot\build_exe.ps1" -PythonExe $PythonExe -IncludeML -BundleModels
}

Write-Host "==> Packaging models zip"
& "$PSScriptRoot\export_release_model_bundle.ps1" -SourceDir $modelsDir

Write-Host "==> Verify turnkey"
& "$PSScriptRoot\verify_turnkey_setup.ps1" -ModelsDir $modelsDir

Write-Host ""
Write-Host "Turnkey outputs:"
Write-Host "  dist/motionanalyzer-gui-ml.exe"
Write-Host "  dist/motionanalyzer-models.zip"
Write-Host "  release/models/ (pretrain weights + manifest)"
