# AI 에이전트 핸드오프 검증 스크립트
# 목적: skills, shell tools, 환경, 파이프라인이 정상 동작하는지 객관적 지표로 검증
# 실행: .\scripts\verify_agent_handoff.ps1 [-Quick] [-OutputJson path]
# -Quick: ML/GPU 검증 생략, 기본 테스트만
# -OutputJson: 결과를 JSON 파일로 저장

param(
    [switch]$Quick = $false,
    [string]$OutputJson = ""
)

$ErrorActionPreference = "Continue"
$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Set-Location $repoRoot

$results = @{
    timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    repo = $repoRoot
    quick = $Quick
    checks = @()
    summary = @{ passed = 0; failed = 0; skipped = 0 }
}

function Add-Check {
    param([string]$Id, [string]$Name, [bool]$Passed, [string]$Message = "", [object]$Metrics = $null, [string]$Skipped = "")
    $r = @{
        id = $Id
        name = $Name
        passed = $Passed
        message = $Message
        metrics = $Metrics
    }
    if ($Skipped) { $r.skipped = $Skipped }
    $results.checks += $r
    if ($Skipped) { $results.summary.skipped++ }
    elseif ($Passed) { $results.summary.passed++ }
    else { $results.summary.failed++ }
}

# [1] 문서 존재 확인
$docs = @(
    "docs/AGENT_HANDOFF_QUICK_START.md",
    "docs/INDEX.md",
    "docs/PROJECT_GOALS.md",
    "docs/DEVELOPMENT_ROADMAP_FINAL.md",
    "docs/PHASE_B_INSIGHTS.md",
    "AGENTS.md"
)
$missing = $docs | Where-Object { -not (Test-Path (Join-Path $repoRoot $_)) }
if ($missing.Count -eq 0) {
    Add-Check -Id "docs" -Name "Core docs exist" -Passed $true -Metrics @{ count = $docs.Count }
} else {
    Add-Check -Id "docs" -Name "Core docs exist" -Passed $false -Message ("Missing: " + ($missing -join ', '))
}

# [2] Skills/Rules 존재
$cursorItems = @(
    ".cursor/skills/ai-coding-accelerator/SKILL.md",
    ".cursor/skills/agent-performance/SKILL.md",
    ".cursor/rules/motionanalyzer.mdc",
    ".cursor/rules/fpcb-domain-knowledge.mdc",
    ".cursor/rules/cursor-tools-optimization.mdc"
)
$missingRules = $cursorItems | Where-Object { -not (Test-Path (Join-Path $repoRoot $_)) }
if ($missingRules.Count -eq 0) {
    Add-Check -Id "cursor-config" -Name "Skills/Rules exist" -Passed $true -Metrics @{ count = $cursorItems.Count }
} else {
    Add-Check -Id "cursor-config" -Name "Skills/Rules exist" -Passed $false -Message ("Missing: " + ($missingRules -join ', '))
}

# [3] corpus-index
$corpusPath = Join-Path $repoRoot "indexes/corpus-index.json"
if ((Test-Path $corpusPath) -and (Get-Content $corpusPath -Raw | ConvertFrom-Json)) {
    Add-Check -Id "corpus-index" -Name "corpus-index.json valid" -Passed $true
} else {
    Add-Check -Id "corpus-index" -Name "corpus-index.json valid" -Passed $false -Message "File missing or JSON parse failed"
}

# [4] Python 환경 (venv 또는 venv-gpu)
$venvStd = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvGpu = Join-Path $repoRoot ".venv-gpu\Scripts\python.exe"
$python = $null
if (Test-Path $venvGpu) {
    $python = $venvGpu
    Add-Check -Id "venv" -Name "Venv (.venv-gpu)" -Passed $true -Metrics @{ path = ".venv-gpu" }
} elseif (Test-Path $venvStd) {
    $python = $venvStd
    Add-Check -Id "venv" -Name "Venv (.venv)" -Passed $true -Metrics @{ path = ".venv" }
} else {
    $python = "python"
    Add-Check -Id "venv" -Name "Venv" -Passed $false -Message "venv or venv-gpu missing. Run bootstrap or setup_gpu_env"
}

# [5] motionanalyzer 패키지 import
if ($python) {
    $srcPath = (Join-Path $repoRoot "src") -replace '\\', '/'
    $importCode = "import sys; sys.path.insert(0, '$srcPath'); import motionanalyzer; print('OK')"
    $importResult = & $python -c $importCode 2>&1
    if ($importResult -match "^OK") {
        Add-Check -Id "import" -Name "motionanalyzer import" -Passed $true
    } else {
        Add-Check -Id "import" -Name "motionanalyzer import" -Passed $false -Message ($importResult -join " ")
    }
}

# [6] pytest (prefer .venv for dev deps; else $python)
$pytestPython = if (Test-Path $venvStd) { $venvStd } else { $python }
$pytestStart = Get-Date
$pytestOut = & $pytestPython -m pytest tests/ -q --tb=no -x 2>&1
$pytestElapsed = ((Get-Date) - $pytestStart).TotalSeconds
$pytestExit = $LASTEXITCODE
if ($pytestExit -eq 0) {
    $match = [regex]::Match($pytestOut -join " ", "(\d+) passed")
    $passedCount = if ($match.Success) { [int]$match.Groups[1].Value } else { 0 }
    Add-Check -Id "pytest" -Name "pytest" -Passed $true -Metrics @{ passed = $passedCount; elapsed_sec = [math]::Round($pytestElapsed, 2) }
} else {
    Add-Check -Id "pytest" -Name "pytest" -Passed $false -Message ("exit=" + $pytestExit) -Metrics @{ elapsed_sec = [math]::Round($pytestElapsed, 2) }
}

# [7] CLI doctor
$doctorOut = & $python -m motionanalyzer.cli doctor 2>&1
if ($LASTEXITCODE -eq 0 -and $doctorOut -match "ready|OK") {
    Add-Check -Id "cli-doctor" -Name "CLI doctor" -Passed $true
} else {
    Add-Check -Id "cli-doctor" -Name "CLI doctor" -Passed $false -Message ($doctorOut -join " ")
}

# [8] gen-synthetic + validate-synthetic (smoke)
$tmpDir = Join-Path $env:TEMP "ma_verify_$(Get-Date -Format 'HHmmss')"
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
$genOut = & $python -m motionanalyzer.cli gen-synthetic --output-dir $tmpDir --frames 60 --points-per-frame 180 --scenario normal 2>&1
$genOk = $LASTEXITCODE -eq 0
if ($genOk) {
    $valOut = & $python -m motionanalyzer.cli validate-synthetic --input-dir $tmpDir --scenario normal 2>&1
    $valOk = $LASTEXITCODE -eq 0 -and $valOut -match "passed|validation"
}
Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
if ($genOk -and $valOk) {
    Add-Check -Id "synthetic-smoke" -Name "Synthetic gen/validate" -Passed $true
} else {
    Add-Check -Id "synthetic-smoke" -Name "Synthetic gen/validate" -Passed $false -Message ("gen=" + $genOk + " val=" + $valOk)
}

# [9] GPU/PyTorch (Quick이면 스킵)
if (-not $Quick -and (Test-Path $venvGpu)) {
    $cudaCode = 'import torch; print("cuda:", torch.cuda.is_available())'
    $cudaOut = & $venvGpu -c $cudaCode 2>&1
    if ($LASTEXITCODE -eq 0 -and $cudaOut -match 'cuda:') {
        $cudaOk = $cudaOut -match 'cuda: True'
        Add-Check -Id "cuda" -Name "PyTorch CUDA" -Passed $cudaOk -Metrics @{ available = $cudaOk }
    } else {
        Add-Check -Id "cuda" -Name "PyTorch CUDA" -Passed $false -Message ($cudaOut -join " ")
    }
} elseif (-not $Quick -and -not (Test-Path $venvGpu)) {
    Add-Check -Id "cuda" -Name "PyTorch CUDA" -Passed $false -Skipped "venv-gpu missing"
}

# [10] QA 게이트 스크립트 존재 및 실행 가능 (Quick이면 존재만 확인)
$qaScripts = @(
    "scripts/evaluate_synthetic_dataset_quality.py",
    "scripts/validate_enhanced_dream.py"
)
$qaOk = $true
foreach ($s in $qaScripts) {
    if (-not (Test-Path (Join-Path $repoRoot $s))) { $qaOk = $false; break }
}
if ($qaOk) {
    if (-not $Quick -and (Test-Path $venvGpu)) {
        $qaStart = Get-Date
        $qaOut = & $venvGpu scripts/evaluate_synthetic_dataset_quality.py 2>&1
        $qaElapsed = ((Get-Date) - $qaStart).TotalSeconds
        if ($LASTEXITCODE -eq 0) {
            Add-Check -Id "qa-gate" -Name "QA gate (quality)" -Passed $true -Metrics @{ elapsed_sec = [math]::Round($qaElapsed, 2) }
        } else {
            Add-Check -Id "qa-gate" -Name "QA gate (quality)" -Passed $false -Message ("exit=" + $LASTEXITCODE)
        }
    } else {
        Add-Check -Id "qa-gate" -Name "QA gate scripts" -Passed $true -Metrics @{ scripts = $qaScripts.Count }
    }
} else {
    Add-Check -Id "qa-gate" -Name "QA gate scripts" -Passed $false -Message "Scripts missing"
}

# [11] Shell 스크립트 존재
$shellScripts = @(
    "scripts/run_full_pipeline.ps1",
    "scripts/setup_gpu_env.ps1",
    "scripts/run_gui.ps1",
    "scripts/build_exe.ps1",
    "scripts/verify_agent_handoff.ps1"
)
$shellMissing = $shellScripts | Where-Object { -not (Test-Path (Join-Path $repoRoot $_)) }
if ($shellMissing.Count -eq 0) {
    Add-Check -Id "shell-tools" -Name "Shell tools exist" -Passed $true -Metrics @{ count = $shellScripts.Count }
} else {
    Add-Check -Id "shell-tools" -Name "Shell tools exist" -Passed $false -Message ("Missing: " + ($shellMissing -join ', '))
}

# [12] Agent tools 디렉터리 (선택)
$agentToolsDir = Join-Path $repoRoot "scripts/agent_tools"
if (Test-Path $agentToolsDir) {
    $toolFiles = (Get-ChildItem $agentToolsDir -File -ErrorAction SilentlyContinue).Count
    Add-Check -Id "agent-tools" -Name "Agent tools dir" -Passed $true -Metrics @{ files = $toolFiles }
} else {
    Add-Check -Id "agent-tools" -Name "Agent tools dir" -Passed $false -Message "scripts/agent_tools missing (optional)"
}

# 출력
$total = $results.summary.passed + $results.summary.failed + $results.summary.skipped
$score = if ($total -gt 0) { [math]::Round(100.0 * $results.summary.passed / $total, 1) } else { 0 }
$results.summary.score = $score
$results.summary.total = $total

Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host '  AI Agent Handoff Verification' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ('  Passed: ' + $results.summary.passed + ' | Failed: ' + $results.summary.failed + ' | Skipped: ' + $results.summary.skipped) -ForegroundColor White
$scoreColor = if ($score -ge 80) { 'Green' } elseif ($score -ge 60) { 'Yellow' } else { 'Red' }
Write-Host ('  Score: ' + $score + '/100') -ForegroundColor $scoreColor
Write-Host ''

foreach ($c in $results.checks) {
    $icon = if ($c.skipped) { '~' } elseif ($c.passed) { 'OK' } else { 'FAIL' }
    $color = if ($c.skipped) { 'Gray' } elseif ($c.passed) { 'Green' } else { 'Red' }
    Write-Host ('  [' + $icon + '] ' + $c.name) -ForegroundColor $color
    if ($c.message -and -not $c.passed) { Write-Host ('      ' + $c.message) -ForegroundColor Gray }
}

if ($OutputJson) {
    $results | ConvertTo-Json -Depth 5 | Set-Content $OutputJson -Encoding UTF8
    Write-Host ''
    Write-Host ('  JSON saved: ' + $OutputJson) -ForegroundColor Gray
}

Write-Host ''
exit $(if ($results.summary.failed -gt 0) { 1 } else { 0 })
