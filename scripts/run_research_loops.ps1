# 연구 개발 루프 3회 이상 실행
# 상황분석 → 성능개선개발 → 테스트 → 평가분석 → 추가개발
# Usage: .\scripts\run_research_loops.ps1
#        .\scripts\run_research_loops.ps1 -MaxLoops 3 -SkipLoop2  # 100k 스킵

param(
    [int]$MaxLoops = 4,
    [switch]$SkipLoop2,
    [switch]$Quick,   # max-train 2000 for faster runs
    [switch]$NoEarlyExit  # Loop 1: run all 6 steps (P1 full comparison)
)

$ErrorActionPreference = "Stop"
$repo = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $repo
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

$loop1Out = Join-Path $repo "reports\loop1_threshold_comparison.json"
$ts = Get-Date -Format "yyyyMMdd_HHmm"

Write-Host "=== Research Development Loops (>=3) ===" -ForegroundColor Cyan
Write-Host "MaxLoops: $MaxLoops | Skip100k: $SkipLoop2 | Quick: $Quick`n"

# Loop 1: Threshold strategy comparison (fp_focused)
Write-Host "[LOOP 1] Threshold strategy verification (fp_focused)..." -ForegroundColor Yellow
$args1 = @("scripts/run_normal_fp_improvement_loop.py", "--base-dir", "data/synthetic/ml_dataset_fp_focused")
if ($Quick) { $args1 += @("--max-train", "2000") }
if ($NoEarlyExit) { $args1 += @("--no-early-exit") }
& $py $args1
$r1 = $LASTEXITCODE
if ($r1 -eq 0) { Write-Host "  Loop 1: Target achieved (FP rate <= 0.1%)" -ForegroundColor Green }
else { Write-Host "  Loop 1: Max steps reached or error (exit $r1)" -ForegroundColor Gray }

# Loop 2: 100k cross-validation
if (-not $SkipLoop2) {
    Write-Host "`n[LOOP 2] 100k inference-only evaluation..." -ForegroundColor Yellow
    & $py scripts/evaluate_100k_inference_only.py
    $r2 = $LASTEXITCODE
    if ($r2 -eq 0) { Write-Host "  Loop 2: 100k eval complete" -ForegroundColor Green }
    else { Write-Host "  Loop 2: Error (exit $r2)" -ForegroundColor Red }
} else {
    Write-Host "`n[LOOP 2] Skipped (--SkipLoop2)" -ForegroundColor Gray
}

# Loop 3: Ablation (DREAM vs PatchCore vs Ensemble) - already in analysis.json
Write-Host "`n[LOOP 3] Ablation (DREAM/PatchCore/Ensemble) - check analysis.json" -ForegroundColor Yellow
$analysisPath = Join-Path $repo "reports\crack_detection_analysis\analysis.json"
if (Test-Path $analysisPath) {
    $a = Get-Content $analysisPath | ConvertFrom-Json
    $ens = $a.models.Ensemble
    Write-Host "  Ensemble: TN=$($ens.tn) FP=$($ens.fp) FN=$($ens.fn) TP=$($ens.tp)" -ForegroundColor Cyan
} else {
    Write-Host "  analysis.json not found" -ForegroundColor Gray
}

# Summary
Write-Host "`n=== Loop Summary ===" -ForegroundColor Cyan
Write-Host "  Loop 1: Threshold comparison - run_normal_fp_improvement_loop"
Write-Host "  Loop 2: 100k eval - evaluate_100k_inference_only"
Write-Host "  Loop 3: Ablation - see analysis.json models"
Write-Host "`nNext: If research complete, run paper pipeline:"
Write-Host "  .\scripts\archive_legacy_deliverables.ps1"
Write-Host "  .\scripts\run_paper_pipeline.ps1"
Write-Host "`nSee docs/RESEARCH_AND_PAPER_MASTER_PLAN.md for full plan." -ForegroundColor Gray
