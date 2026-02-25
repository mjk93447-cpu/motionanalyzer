<#
.SYNOPSIS
    중간 백업: 현재 작업을 backup/YYYY-MM-DD-HHmm 브랜치에 커밋·푸시

.DESCRIPTION
    별도 지시 없이 중간 백업을 수행. main과 분리된 backup 브랜치에 저장.

.PARAMETER Message
    커밋 메시지 (기본: "backup: YYYY-MM-DD HH:mm")

.PARAMETER SyncFirst
    백업 전 origin/main에서 pull --rebase 수행 (원격 변경 반영)

.PARAMETER PushMain
    백업 병합 후 main을 origin에 push

.EXAMPLE
    .\scripts\git_backup.ps1
    .\scripts\git_backup.ps1 -Message "WIP: dataset cleanup"
    .\scripts\git_backup.ps1 -SyncFirst -PushMain
#>

Param(
    [string]$Message = "",
    [switch]$SyncFirst,
    [switch]$PushMain
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$timestamp = Get-Date -Format "yyyy-MM-dd-HHmm"
$branchName = "backup/$timestamp"
$commitMsg = if ($Message) { $Message } else { "backup: $timestamp" }

# Optional: sync with remote before backup
if ($SyncFirst) {
    git fetch origin 2>&1 | Out-Null
    git pull --rebase origin main 2>&1 | Out-Null
}

# Check for changes
$status = git status --porcelain 2>&1
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "[OK] No changes to backup. Working tree clean."
    exit 0
}

# Stash if on main with uncommitted (we'll create new branch from current)
$currentBranch = git rev-parse --abbrev-ref HEAD 2>&1
$hasRemote = git remote get-url origin 2>$null
if (-not $hasRemote) {
    Write-Host "[WARN] No remote 'origin'. Backup will be local only."
}

# Create backup branch, commit changes there, merge back to current
$err = $null; git checkout -b $branchName 2>&1 | Out-Null; $ec = $LASTEXITCODE
if ($ec -ne 0) { git checkout $branchName 2>&1 | Out-Null }
git add -A
git status --short
git commit -m $commitMsg 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] No changes to commit."
    git checkout $currentBranch 2>$null
    git branch -D $branchName 2>$null
    exit 0
}
git push -u origin $branchName 2>&1
$pushOk = ($LASTEXITCODE -eq 0)
git checkout $currentBranch 2>$null
git merge $branchName -m "merge backup: $timestamp" --no-edit 2>&1
if ($pushOk) { Write-Host "[OK] Backup pushed to origin/$branchName" }
else { Write-Host "[WARN] Push failed. Backup is local: $branchName" }
Write-Host "  Merged backup into $currentBranch"

if ($PushMain) {
    git push origin $currentBranch 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] Pushed $currentBranch to origin" }
    else { Write-Host "[WARN] Push $currentBranch failed" }
}
