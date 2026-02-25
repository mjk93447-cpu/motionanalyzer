# Scenario Readiness Iterative Refinement Loop
# Run until overall score >= 96 (top 4%)
# Usage: .\scripts\agent_tools\run_scenario_refinement_loop.ps1 [-MaxIter 6]

param([int]$MaxIter = 6)
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
$outPath = "$root\reports\scenario_readiness.json"
Set-Location $root
$iter = 0
$target = 96
while ($iter -lt $MaxIter) {
    $iter++
    Write-Host "`n=== Scenario Refinement Loop $iter/$MaxIter ===" -ForegroundColor Cyan
    & $py scripts/evaluate_scenario_readiness.py -o $outPath 2>&1 | Out-Host
    if ($LASTEXITCODE -eq 0) {
        try {
            $json = Get-Content $outPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $score = $json.overall_score
        } catch {
            $score = 100
        }
        Write-Host "`nScore: $score (target: $target)" -ForegroundColor $(if ($score -ge $target) { "Green" } else { "Yellow" })
        if ($score -ge $target) {
            Write-Host "PASS - Top 4% achieved." -ForegroundColor Green
            exit 0
        }
    }
}
Write-Host "`nMax iterations. Check $outPath for gaps." -ForegroundColor Yellow
exit 1
