Param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at '$PythonExe'. Run bootstrap first."
}

& $PythonExe -m motionanalyzer.cli gui --host $Host --port $Port
