# Cursor 속도 최적화 3단계 순차 실행
# 외부 PowerShell에서 실행 - Cursor 종료 후에도 작업 계속됨

$LogPath = "$env:TEMP\cursor_optimization_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
function Log { param($m) $t = Get-Date -Format "HH:mm:ss"; "$t $m" | Tee-Object -FilePath $LogPath -Append }

Log "=========================================="
Log "  Cursor 속도 최적화 3단계 시작"
Log "=========================================="

# [1/3] CURSOR_FULL_RESET - 캐시 및 채팅 히스토리 삭제
Log "[1/3] Cursor 프로세스 종료 및 캐시 정리..."
Stop-Process -Name "Cursor*" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

$CursorDir = "$env:APPDATA\Cursor"
$dirs = @(
    "$CursorDir\User\workspaceStorage",
    "$CursorDir\Cache",
    "$CursorDir\CachedData",
    "$CursorDir\CachedExtensions",
    "$CursorDir\GPUCache",
    "$CursorDir\Code Cache",
    "$CursorDir\logs"
)
foreach ($d in $dirs) {
    if (Test-Path $d) {
        Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
        Log "  삭제: $d"
    }
}
New-Item -ItemType Directory -Path "$CursorDir\User\workspaceStorage" -Force | Out-Null
New-Item -ItemType Directory -Path "$CursorDir\Cache" -Force | Out-Null
New-Item -ItemType Directory -Path "$CursorDir\logs" -Force | Out-Null
Log "[1/3] 완료."

# [2/3] NODE_OPTIONS 영구 설정
Log "[2/3] NODE_OPTIONS 환경 변수 설정 (8GB)..."
[System.Environment]::SetEnvironmentVariable("NODE_OPTIONS", "--max-old-space-size=8192", "User")
Log "[2/3] 완료. (새 터미널/Cursor에서 적용)"

# [3/3] RAM 디스크 설정 시도
Log "[3/3] RAM 디스크 설정 시도..."
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
try {
    & "$ScriptDir\CURSOR_RAMDISK_SETUP.ps1" 2>&1 | ForEach-Object { Log "  $_" }
} catch {
    Log "  RAM 디스크 스크립트 건너뜀: $_"
}
$imdisk = Get-Command imdisk -ErrorAction SilentlyContinue
if (-not $imdisk) {
    Log "  ImDisk 미설치 - RAM 디스크는 수동 설정 필요: https://imdisktoolkit.com/"
}
Log "[3/3] 완료."

Log "=========================================="
Log "  모든 최적화 완료. 로그: $LogPath"
Log "=========================================="
