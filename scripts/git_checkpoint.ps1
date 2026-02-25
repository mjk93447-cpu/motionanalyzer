<#
.SYNOPSIS
    중간 체크포인트: 현재 브랜치에 빠른 커밋

.PARAMETER Message
    커밋 메시지 (필수)

.EXAMPLE
    .\scripts\git_checkpoint.ps1 -Message "feat: add dataset naming rule"
#>

param([Parameter(Mandatory=$true)][string]$Message)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$status = git status --porcelain 2>&1
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "[OK] No changes to commit."
    exit 0
}

git add -A
git commit -m $Message
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Checkpoint committed."
    $branch = git rev-parse --abbrev-ref HEAD
    Write-Host "  Branch: $branch"
} else {
    Write-Host "[ERROR] Commit failed."
    exit 1
}
