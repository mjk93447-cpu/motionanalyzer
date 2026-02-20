<#
.SYNOPSIS
    Build motionanalyzer CLI EXE (console) for command-line analysis.

.DESCRIPTION
    Builds motionanalyzer-cli.exe using PyInstaller and motionanalyzer-cli.spec.
    The CLI EXE supports: analyze-bundle, compare-runs, run-fpcb-pipeline, etc.

.EXAMPLE
    .\scripts\build_cli_exe.ps1
#>

Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$resolvedPython = $null
if ($PythonExe -match "[\\/]" -or $PythonExe -match "^\.+") {
    if (-not (Test-Path $PythonExe)) {
        throw "Python executable not found at '$PythonExe'. Run bootstrap first."
    }
    $resolvedPython = $PythonExe
} else {
    $cmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Python command '$PythonExe' was not found in PATH."
    }
    $resolvedPython = $cmd.Source
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

Write-Host "==> Installing build dependencies"
& $resolvedPython -m pip install -e ".[build]"

Write-Host "==> Building CLI EXE (motionanalyzer-cli.exe)"
& $resolvedPython -m PyInstaller --noconfirm --clean motionanalyzer-cli.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyInstaller failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build complete: dist\motionanalyzer-cli.exe"
Write-Host ""
Write-Host "Usage examples:"
Write-Host "  .\dist\motionanalyzer-cli.exe analyze-bundle --input-dir data\synthetic\examples\normal --output-dir exports\vectors\exe_test"
Write-Host "  .\dist\motionanalyzer-cli.exe run-fpcb-pipeline"
Write-Host ""
