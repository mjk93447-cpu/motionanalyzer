# Normal FP Rate improvement batch - run steps until target achieved
# Each step: ~10-15 min (full) or ~3-5 min (max-train 2000)
# Target: Normal FP Rate <= 0.1%

$ErrorActionPreference = "Stop"
$repo = "c:\motionanalyzer"
$base = "$repo\data\synthetic\ml_dataset_fp_focused"
$maxTrain = 2000  # reduce for faster iteration

Set-Location $repo

if (-not (Test-Path "$base\manifest.json")) {
    Write-Host "[ERROR] manifest.json not found in $base" -ForegroundColor Red
    exit 1
}

$steps = @(
    @{ Label = "Step 1: normal-fp-max=0"; Args = "--normal-fp-max 0" },
    @{ Label = "Step 2: +margin=0.2"; Args = "--normal-fp-max 0 --threshold-margin 0.2" },
    @{ Label = "Step 3: percentile=99.9"; Args = "--threshold-percentile 99.9" },
    @{ Label = "Step 4: percentile=99.9 +margin=0.1"; Args = "--threshold-percentile 99.9 --threshold-margin 0.1" }
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Normal FP Improvement Batch (target <= 0.1%)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Base: $base | max-train: $maxTrain"
Write-Host ""

foreach ($s in $steps) {
    Write-Host "[$($s.Label)]" -ForegroundColor Yellow
    $cmd = "python scripts/analyze_crack_detection.py --base-dir $base --dataset-level-eval --max-train $maxTrain $($s.Args)"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "  [WARN] Exit code $LASTEXITCODE" -ForegroundColor Yellow }
    $json = Get-Content "$repo\reports\crack_detection_analysis\analysis.json" -Raw | ConvertFrom-Json
    $ens = $json.models.Ensemble
    $fp = $ens.fp
    $n = $json.n_normal
    $rate = [math]::Round(100.0 * $fp / $n, 4)
    Write-Host "  Ensemble FP=$fp n_normal=$n => Normal FP Rate = $rate%"
    if ($rate -le 0.1) {
        Write-Host ""
        Write-Host "[OK] Target achieved: Normal FP rate <= 0.1%" -ForegroundColor Green
        exit 0
    }
    Write-Host ""
}

Write-Host "[INFO] Max steps reached. Consider Phase 3 (class weight, loss)." -ForegroundColor Yellow
exit 1
