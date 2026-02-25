Param(
    [string]$PythonExe = "",
    [switch]$ML = $false,
    [string]$Host = "127.0.0.1",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
if (-not $PythonExe) {
    $venvGpu = Join-Path $repoRoot ".venv-gpu\Scripts\python.exe"
    $venvStd = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if ($ML -and (Test-Path $venvGpu)) {
        $PythonExe = $venvGpu
    } elseif (Test-Path $venvStd) {
        $PythonExe = $venvStd
    } elseif (Test-Path $venvGpu) {
        $PythonExe = $venvGpu
    } else {
        $PythonExe = "python"
    }
}

if ($PythonExe -match "[\\/]" -and -not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Run bootstrap or setup_gpu_env first."
}

& $PythonExe -m motionanalyzer.cli gui --host $Host --port $Port
