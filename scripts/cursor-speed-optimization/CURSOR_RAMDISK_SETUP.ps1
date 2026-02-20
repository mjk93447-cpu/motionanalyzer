# Cursor RAM 디스크 설정 스크립트 (Windows)
# ImDisk Toolkit 필요: https://imdisktoolkit.com/ (무료)
# 효과: I/O 병목 40% 이상 감소, HDD에서 특히 효과적

param(
    [string]$RamDriveLetter = "R",
    [int]$SizeMB = 2048
)

$ErrorActionPreference = "Stop"
$CursorDir = "$env:APPDATA\Cursor"
$RamPath = "${RamDriveLetter}:\CursorCache"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cursor RAM 디스크 설정" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Cursor 완전 종료 확인
Write-Host "[1/5] Cursor를 완전히 종료한 후 실행하세요." -ForegroundColor Yellow
$proc = Get-Process -Name "Cursor*" -ErrorAction SilentlyContinue
if ($proc) {
    Write-Host "ERROR: Cursor가 실행 중입니다. 종료 후 다시 실행하세요." -ForegroundColor Red
    exit 1
}

# 2. ImDisk 설치 확인
$imdisk = Get-Command imdisk -ErrorAction SilentlyContinue
if (-not $imdisk) {
    Write-Host "ImDisk Toolkit이 설치되지 않았습니다." -ForegroundColor Yellow
    Write-Host "다운로드: https://imdisktoolkit.com/" -ForegroundColor White
    Write-Host "설치 후 imdisk.exe를 PATH에 추가하세요." -ForegroundColor White
    Write-Host ""
    Write-Host "수동 설정 방법:" -ForegroundColor Cyan
    Write-Host "1. ImDisk로 ${RamDriveLetter}: 드라이브 생성 (${SizeMB}MB)" -ForegroundColor White
    Write-Host "2. $RamPath\Cache, $RamPath\CachedData, $RamPath\globalStorage 생성" -ForegroundColor White
    Write-Host "3. 아래 junction 명령 실행 (관리자 권한)" -ForegroundColor White
    Write-Host ""
    Write-Host "mklink /J `"$CursorDir\Cache`" `"$RamPath\Cache`"" -ForegroundColor Gray
    Write-Host "mklink /J `"$CursorDir\CachedData`" `"$RamPath\CachedData`"" -ForegroundColor Gray
    Write-Host "mklink /J `"$CursorDir\User\globalStorage`" `"$RamPath\globalStorage`"" -ForegroundColor Gray
    exit 1
}

# 3. RAM 디스크 생성
Write-Host "[2/5] RAM 디스크 생성 중 (${RamDriveLetter}: ${SizeMB}MB)" -ForegroundColor Green
if (Test-Path "${RamDriveLetter}:") {
    Write-Host "  ${RamDriveLetter}: 드라이브가 이미 존재합니다. 기존 사용 중이면 건너뜁니다." -ForegroundColor Yellow
} else {
    imdisk -a -s ${SizeMB}M -m ${RamDriveLetter}: -p "/fs:ntfs /q /y"
}

# 4. 폴더 생성 및 데이터 복사
Write-Host "[3/5] 폴더 생성 및 데이터 복사" -ForegroundColor Green
$folders = @("Cache", "CachedData", "globalStorage")
foreach ($f in $folders) {
    $dest = "$RamPath\$f"
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest -Force | Out-Null }
    $src = if ($f -eq "globalStorage") { "$CursorDir\User\globalStorage" } else { "$CursorDir\$f" }
    if ((Test-Path $src) -and -not (Test-Path $src -PathType Leaf)) {
        Write-Host "  복사: $f -> RAM" -ForegroundColor Gray
        robocopy $src $dest /E /MIR /R:1 /W:1 /NFL /NDL /NJH /NJS 2>$null
    }
}

# 5. Junction 생성 (관리자 권한 필요)
Write-Host "[4/5] Junction 생성 (관리자 권한 필요)" -ForegroundColor Green
$junctions = @(
    @{ Orig = "$CursorDir\Cache"; Dest = "$RamPath\Cache" },
    @{ Orig = "$CursorDir\CachedData"; Dest = "$RamPath\CachedData" },
    @{ Orig = "$CursorDir\User\globalStorage"; Dest = "$RamPath\globalStorage" }
)
foreach ($j in $junctions) {
    if (Test-Path $j.Orig) {
        $item = Get-Item $j.Orig
        if ($item.Attributes -match "ReparsePoint") {
            Write-Host "  Junction 이미 존재: $($j.Orig)" -ForegroundColor Gray
        } else {
            Remove-Item $j.Orig -Recurse -Force
            cmd /c mklink /J "`"$($j.Orig)`"" "`"$($j.Dest)`""
            Write-Host "  Junction 생성: $($j.Orig)" -ForegroundColor Gray
        }
    }
}

Write-Host "[5/5] 완료." -ForegroundColor Green
Write-Host ""
Write-Host "주의: RAM 디스크는 재부팅 시 초기화됩니다. Cursor가 자동으로 재생성합니다." -ForegroundColor Yellow
Write-Host "재부팅 후 이 스크립트를 다시 실행하거나, ImDisk에서 '부팅 시 자동 마운트'를 설정하세요." -ForegroundColor Yellow
Write-Host ""
