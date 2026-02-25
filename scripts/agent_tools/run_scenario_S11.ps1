# S11: 배치 분석 (Phase D) - 다수 데이터셋 일괄
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
# 기본: DS-260220-ml-100k-100k-60f, DS-260223-ml-fp-20k-60f
$datasets = @("data/synthetic/ml_dataset_100k_v2", "data/synthetic/ml_dataset_fp_focused")
foreach ($d in $datasets) {
    if (Test-Path $d) {
        Write-Host "Analyzing: $d"
        & $py scripts/analyze_crack_detection.py --base-dir $d 2>&1
    }
}
exit 0
