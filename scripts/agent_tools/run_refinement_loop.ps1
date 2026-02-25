# Iterative Refinement Loop - Run until agent setup score >= 96%
# Usage: .\scripts\agent_tools\run_refinement_loop.ps1 [-MaxIter 5] [-Quick]

param(
    [int]$MaxIter = 5,
    [switch]$Quick = $false
)

$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" }
      elseif (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" }
      else { "python" }
$outPath = "$root\reports\agent_setup_evaluation.json"

Set-Location $root
$iter = 0
$target = 96

while ($iter -lt $MaxIter) {
    $iter++
    Write-Host "`n=== Refinement iteration $iter/$MaxIter ===" -ForegroundColor Cyan
    $opts = @("scripts/evaluate_agent_setup.py", "-o", $outPath)
    if ($Quick) { $opts += "--quick" }
    & $py @opts 2>&1 | Out-Host
    if ($LASTEXITCODE -eq 0) {
        $json = Get-Content $outPath -Raw | ConvertFrom-Json
        $pct = $json.percentile_equivalent
        Write-Host "`nScore: $pct% (target: $target%)" -ForegroundColor $(if ($pct -ge $target) { "Green" } else { "Yellow" })
        if ($pct -ge $target) {
            Write-Host "PASS - Top 4% achieved." -ForegroundColor Green
            exit 0
        }
    }
}
Write-Host "`nMax iterations reached. Check $outPath for gaps." -ForegroundColor Yellow
exit 1
