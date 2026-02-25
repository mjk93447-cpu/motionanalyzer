# GitHub 활용 세팅 효율성 평가 (0-100점)
$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$total = 0
$maxTotal = 100

# 1. 백업 스크립트 (25점)
if (Test-Path "scripts\git_backup.ps1") { $total += 25 } else { $total += 0 }

# 2. 체크포인트 스크립트 (20점)
if (Test-Path "scripts\git_checkpoint.ps1") { $total += 20 } else { $total += 0 }

# 3. 워크플로우 스크립트 (15점)
if (Test-Path "scripts\git_workflow.ps1") { $total += 15 } else { $total += 0 }

# 4. GitHub Actions 아티팩트 (20점)
$wf = 0
if (Test-Path ".github\workflows\build-windows-exe.yml") {
    $c = Get-Content ".github\workflows\build-windows-exe.yml" -Raw
    if ($c -match "upload-artifact") { $wf += 10 }
    if ($c -match "retention-days") { $wf += 10 }
}
$total += $wf

# 5. 원격 (10점)
$r = git remote get-url origin 2>&1
if ($r -match "github") { $total += 10 }

# 6. 문서 (10점)
if (Test-Path "GITHUB_SETUP.md") { $total += 5 }
if (Test-Path "docs\GITHUB_WORKFLOW_COMPLETE.md") { $total += 5 }

Write-Host "=== GitHub Setup Evaluation ==="
Write-Host "  Score: $total/100"
Write-Host ""
exit 0
