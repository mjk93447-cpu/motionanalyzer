# FP-focused pipeline: fp_focused dataset -> analysis (dataset-level, zero-FP) -> report
$ErrorActionPreference = "Stop"
$repo = "c:\motionanalyzer"
$base = "$repo\data\synthetic\ml_dataset_fp_focused"
$manifest = "$base\manifest.json"

Set-Location $repo

# Wait for manifest
Write-Host "Waiting for fp_focused dataset (manifest.json)..." -ForegroundColor Yellow
while (-not (Test-Path $manifest)) {
    Start-Sleep -Seconds 60
    $n = 0
    if (Test-Path "$base\normal") { $n += (Get-ChildItem "$base\normal" -Directory -EA 0 | Measure).Count }
    if (Test-Path "$base\crack_in_bending") { $c = (Get-ChildItem "$base\crack_in_bending" -Directory -EA 0 | Measure).Count; $n += $c }
    Write-Host "  normal+crack=$n"
}
Write-Host "Manifest found. Running analysis..." -ForegroundColor Green

# Analysis with FP minimization
python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused `
    --dataset-level-eval --zero-fp-priority --min-precision 0.99

# Report
python scripts/generate_final_report_docx.py

Write-Host "`nDone. Report: reports/deliverables/FPCB_Crack_Detection_Final_Report.docx" -ForegroundColor Cyan
