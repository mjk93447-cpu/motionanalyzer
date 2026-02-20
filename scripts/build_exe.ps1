Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [switch]$IncludeML = $false
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

if ($IncludeML) {
    Write-Host "==> Installing ML dependencies (PyTorch CPU-only recommended)"
    & $resolvedPython -m pip install -e ".[ml]"
    Write-Host "==> Building offline Windows desktop GUI executable WITH ML support (PyTorch)"
    Write-Host "Note: EXE size will be larger (~200-500MB) due to PyTorch inclusion"
    $exeName = "motionanalyzer-gui-ml"
    $excludeTorch = ""
} else {
    Write-Host "==> Building offline Windows desktop GUI executable WITHOUT ML support"
    Write-Host "Note: DREAM/PatchCore features will not work. Use -IncludeML for ML features."
    $exeName = "motionanalyzer-gui"
    $excludeTorch = "--exclude-module torch"
}

Write-Host "==> Building with PyInstaller"
$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    "--name", $exeName,
    "--windowed",
    "--exclude-module", "tensorflow",
    "--exclude-module", "keras",
    "--exclude-module", "PySide6",
    "--exclude-module", "pytest",
    "--exclude-module", "streamlit",
    "src\motionanalyzer\desktop_gui.py"
)

if ($excludeTorch) {
    $pyinstallerArgs += "--exclude-module", "torch"
}

& $resolvedPython -m PyInstaller @pyinstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] PyInstaller failed with exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Build complete:"
Write-Host " - dist\$exeName.exe"
if ($IncludeML) {
    Write-Host ""
    Write-Host "ML-enabled EXE includes PyTorch and scikit-learn."
    Write-Host "DREAM and PatchCore models can be loaded and used for inference."
} else {
    Write-Host ""
    Write-Host "Lightweight EXE (ML features disabled)."
    Write-Host "To build with ML support, run: .\scripts\build_exe.ps1 -IncludeML"
}
