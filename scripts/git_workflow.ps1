<#
.SYNOPSIS
    GitHub 워크플로우: 백업, 브랜치, 커밋, 푸시를 한 번에

.DESCRIPTION
    별도 지시 없이 사용 가능한 통합 워크플로우.
    - backup: backup/YYYY-MM-DD-HHmm 브랜치에 푸시
    - commit: 현재 브랜치에 커밋·푸시
    - branch: 새 feature 브랜치 생성·체크아웃
    - status: 현재 상태 요약

.PARAMETER Action
    backup | commit | branch | status

.PARAMETER Message
    커밋 메시지 (commit 시 필수)

.PARAMETER BranchName
    새 브랜치 이름 (branch 시 필수)

.EXAMPLE
    .\scripts\git_workflow.ps1 -Action backup
    .\scripts\git_workflow.ps1 -Action commit -Message "feat: add MCP"
    .\scripts\git_workflow.ps1 -Action branch -BranchName "feature/dataset-cleanup"
#>

Param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("backup","commit","branch","status")]
    [string]$Action,
    [string]$Message = "",
    [string]$BranchName = "",
    [switch]$SyncFirst,
    [switch]$PushMain
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

switch ($Action) {
    "backup" {
        & "$PSScriptRoot\git_backup.ps1" -Message $Message -SyncFirst:$SyncFirst -PushMain:$PushMain
    }
    "commit" {
        if (-not $Message) { Write-Host "[ERROR] -Message required for commit"; exit 1 }
        & "$PSScriptRoot\git_checkpoint.ps1" -Message $Message
        git push 2>&1
    }
    "branch" {
        if (-not $BranchName) { Write-Host "[ERROR] -BranchName required for branch"; exit 1 }
        git checkout -b $BranchName 2>&1
        Write-Host "[OK] Created and switched to $BranchName"
    }
    "status" {
        Write-Host "=== Git Status ==="
        git status
        Write-Host "`n=== Branches ==="
        git branch -a
        Write-Host "`n=== Remote ==="
        git remote -v
    }
}
