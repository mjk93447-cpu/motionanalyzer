# Verify pretrained bundle + smoke inference (CI post-build).
Param(
    [string]$ModelsDir = "",
    [string]$BundleDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
}

$env:PYTHONPATH = "src"
if ($ModelsDir) {
    $env:MOTIONANALYZER_MODELS_DIR = (Resolve-Path $ModelsDir).Path
}

$models = if ($env:MOTIONANALYZER_MODELS_DIR) { $env:MOTIONANALYZER_MODELS_DIR } else { Join-Path $env:APPDATA "motionanalyzer\models" }
if (-not (Test-Path $models)) {
    $models = Join-Path $root "release\models"
}

$required = @("draem_model.pt", "patchcore_model.npz", "bundle_manifest.json")
foreach ($f in $required) {
    $p = Join-Path $models $f
    if (-not (Test-Path $p)) {
        Write-Host "[FAIL] Missing $p"
        exit 1
    }
    Write-Host "[OK] $f"
}

$bundle = $BundleDir
if (-not $bundle) {
    $candidates = @(
        "data\synthetic\ml_pretrain_balanced_3k_60f\normal\normal_0001",
        "data\synthetic\ml_default_1k_60f\normal\normal_0001",
        "data\synthetic\ml_fp_focused_20k_60f\normal\normal_0001"
    )
    foreach ($c in $candidates) {
        $p = Join-Path $root $c
        if (Test-Path $p) {
            $bundle = $p
            break
        }
    }
}

if (-not $bundle -or -not (Test-Path $bundle)) {
    Write-Host "[WARN] No synthetic bundle for smoke test; skipping predict_bundle"
    exit 0
}

$smoke = @"
import os, sys
from pathlib import Path
sys.path.insert(0, 'src')
os.environ['MOTIONANALYZER_MODELS_DIR'] = r'$models'
from motionanalyzer.services.ml_inference import predict_bundle
r = predict_bundle(Path(r'$bundle'), 'patchcore', dataset_level_max=True)
print('smoke_ok', r.get('dataset_is_anomaly'), r.get('dataset_score'))
"@
& $python -c $smoke
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] predict_bundle smoke test"
    exit 1
}
Write-Host "[OK] predict_bundle smoke"
exit 0
