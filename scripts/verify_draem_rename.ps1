# Verify Phase 1 DRAEM rename gate (excludes migration doc + MIGRATION_GUIDE legacy table).
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$exclude = @(
    "migrate_dream_to_draem.py",
    "verify_draem_rename.ps1",
    "MIGRATION_GUIDE.md",
    "backup-manifest.json"
)
$dirs = @("src", "tests", "scripts")
$hits = @()
foreach ($d in $dirs) {
    $path = Join-Path $root $d
    if (-not (Test-Path $path)) { continue }
    Get-ChildItem -Path $path -Recurse -File -Include *.py,*.ps1 |
        Where-Object { $exclude -notcontains $_.Name } |
        ForEach-Object {
            $m = Select-String -Path $_.FullName -Pattern '\bdream\b|\bDREAM\b' -AllMatches -ErrorAction SilentlyContinue
            if ($m) { $hits += $m }
        }
}
if ($hits.Count -gt 0) {
    Write-Host "FAIL: dream/DREAM still found in src/tests/scripts:" -ForegroundColor Red
    $hits | ForEach-Object { Write-Host $_.Path ":" $_.LineNumber $_.Line.Trim() }
    exit 1
}
Write-Host "OK: no dream/DREAM in src, tests, scripts (excluded migration files)." -ForegroundColor Green
python -m pytest tests/test_draem.py tests/test_gui_runners.py -q --tb=short
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "OK: DRAEM unit tests passed." -ForegroundColor Green
