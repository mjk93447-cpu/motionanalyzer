# S06: 논문/리포트 작성
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" } else { "$root\.venv\Scripts\python.exe" }
Set-Location $root
& $py scripts/generate_final_report_docx.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py scripts/generate_final_report_ppt.py
exit $LASTEXITCODE
