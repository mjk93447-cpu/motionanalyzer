# S07: EXE 빌드·배포
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
& "$root\scripts\build_exe.ps1" -PythonExe $py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& "$root\scripts\test_build_exe.ps1"
exit $LASTEXITCODE
