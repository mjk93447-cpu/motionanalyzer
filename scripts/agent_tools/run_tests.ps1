# 에이전트용 테스트 실행 (결정적, 재현 가능)
# 사용: .\scripts\agent_tools\run_tests.ps1 [-Quick]
# -Quick: 실패 시 첫 실패에서 중단

param([switch]$Quick = $false)
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$py = if (Test-Path "$root\.venv-gpu\Scripts\python.exe") { "$root\.venv-gpu\Scripts\python.exe" }
      elseif (Test-Path "$root\.venv\Scripts\python.exe") { "$root\.venv\Scripts\python.exe" }
      else { "python" }
$opts = @("-m", "pytest", "tests/", "-q", "--tb=short")
if ($Quick) { $opts += "-x" }
& $py @opts
exit $LASTEXITCODE
