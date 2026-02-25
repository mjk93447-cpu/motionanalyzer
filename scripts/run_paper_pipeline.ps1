# FPCB Final Paper Pipeline
# Order: 1) Analysis (optional skip) 2) Paper Banana figures (optional) 3) Word 4) PPT
# Usage: .\scripts\run_paper_pipeline.ps1
#        .\scripts\run_paper_pipeline.ps1 -SkipAnalysis -SkipPaperBanana  # Word+PPT only

param(
    [switch]$SkipAnalysis,
    [switch]$SkipPaperBanana,
    [switch]$Run100k
)

$ErrorActionPreference = "Stop"
$repo = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $repo

$py = "python"
if (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
elseif (Get-Command py -ErrorAction SilentlyContinue) { $py = "py" }

Write-Host "=== FPCB Paper Pipeline ===" -ForegroundColor Cyan
Write-Host "Repo: $repo`n"

# 1. Analysis (fp_focused)
if (-not $SkipAnalysis) {
    Write-Host "[1/4] Running crack detection analysis (fp_focused)..." -ForegroundColor Yellow
    & $py scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --normal-fp-max 0
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[1/4] Skip analysis (use existing analysis.json)" -ForegroundColor Gray
}

# 2. (Optional) 100k inference
if ($Run100k) {
    Write-Host "[1b] Running 100k inference-only evaluation..." -ForegroundColor Yellow
    & $py scripts/evaluate_100k_inference_only.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 3. Paper Banana figures (optional)
if (-not $SkipPaperBanana) {
    $figDir = Join-Path $repo "reports\deliverables\figures"
    if (-not (Test-Path $figDir)) { New-Item -ItemType Directory -Path $figDir -Force | Out-Null }
    $methodInput = Join-Path $repo "docs\paperbanana_inputs\fpcb_methodology.txt"
    if ((Test-Path $methodInput) -and (Get-Command paperbanana -ErrorAction SilentlyContinue)) {
        Write-Host "[2/4] Generating methodology figure (Paper Banana)..." -ForegroundColor Yellow
        paperbanana generate --input $methodInput --caption "FPCB bending crack detection pipeline" --output (Join-Path $figDir "fig_methodology.png")
    } elseif (-not (Get-Command paperbanana -ErrorAction SilentlyContinue)) {
        Write-Host "[2/4] Paper Banana not installed (pip install paperbanana); skip figures" -ForegroundColor Gray
    } else {
        Write-Host "[2/4] Skip Paper Banana (no input or not configured)" -ForegroundColor Gray
    }
} else {
    Write-Host "[2/4] Skip Paper Banana" -ForegroundColor Gray
}

# 4. Word
Write-Host "[3/4] Generating Word report..." -ForegroundColor Yellow
& $py scripts/generate_final_report_docx.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 5. PPT
Write-Host "[4/4] Generating PPT..." -ForegroundColor Yellow
& $py scripts/generate_final_report_ppt.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nDone. Output:" -ForegroundColor Green
Write-Host "  reports/deliverables/FPCB_Crack_Detection_Final_Report.docx"
Write-Host "  reports/deliverables/FPCB_Crack_Detection_Final_Report.pptx"
Write-Host "  See docs/PAPER_WRITING_AND_PAPERBANANA_PLAN.md for full plan."
