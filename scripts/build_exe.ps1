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

Write-Host "==> Installing build dependencies"
& $resolvedPython -m pip install -e ".[build]"

Write-Host "==> Building offline Windows desktop GUI executable with PyInstaller"
Write-Host "Note: Only native Windows GUI exe is built for offline use"
& $resolvedPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name motionanalyzer-gui `
    --windowed `
    --exclude-module tensorflow `
    --exclude-module keras `
    --exclude-module torch `
    --exclude-module PySide6 `
    --exclude-module pytest `
    --exclude-module streamlit `
    src\motionanalyzer\desktop_gui.py

Write-Host "Build complete:"
Write-Host " - dist\motionanalyzer-gui.exe"
