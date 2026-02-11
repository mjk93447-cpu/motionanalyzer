Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Run bootstrap first."
}

Write-Host "==> Installing build dependencies"
& $PythonExe -m pip install -e ".[build]"

Write-Host "==> Building Windows executable with PyInstaller"
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name motionanalyzer-cli `
    --collect-all motionanalyzer `
    src\motionanalyzer\cli.py

Write-Host "Build complete: dist\motionanalyzer-cli.exe"
