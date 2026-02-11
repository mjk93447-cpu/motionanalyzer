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

Write-Host "==> Building Windows executable with PyInstaller"
& $resolvedPython -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --name motionanalyzer-cli `
    --collect-all motionanalyzer `
    src\motionanalyzer\cli.py

Write-Host "Build complete: dist\motionanalyzer-cli.exe"
