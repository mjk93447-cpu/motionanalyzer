$ErrorActionPreference = "Stop"

$repo = "c:\motionanalyzer"
$base = Join-Path $repo "data\synthetic\ml_dataset"
$logDir = Join-Path $repo "reports\progress"
$logPath = Join-Path $logDir "generation_monitor.log"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Get-Counts {
  $n = (Get-ChildItem (Join-Path $base "normal") -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
  $c = (Get-ChildItem (Join-Path $base "crack_in_bending") -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
  $p = (Get-ChildItem (Join-Path $base "pre_damaged") -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
  $t = (Get-ChildItem (Join-Path $base "thick_panel") -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
  $total = $n + $c + $p + $t
  return @{ normal=$n; crack=$c; predam=$p; thick=$t; total=$total }
}

function Write-Status($counts) {
  $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  $line = "$ts | normal=$($counts.normal) crack_in_bending=$($counts.crack) pre_damaged=$($counts.predam) thick_panel=$($counts.thick) total=$($counts.total)"
  $line | Tee-Object -FilePath $logPath -Append
}

Set-Location $repo

$stagnant = 0
$prevTotal = -1
"=== Start monitor (1 min interval) ===" | Tee-Object -FilePath $logPath -Append
while ($true) {
  $counts = Get-Counts
  Write-Status $counts

  if ($counts.total -le $prevTotal) {
    $stagnant += 1
  } else {
    $stagnant = 0
  }
  $prevTotal = $counts.total

  if ($stagnant -ge 5) {
    "WARNING: generation appears stagnant for >=5 minutes. Check generator process/log." | Tee-Object -FilePath $logPath -Append
    $stagnant = 0
  }

  if ($counts.normal -ge 80000 -and $counts.crack -ge 11000 -and $counts.predam -ge 5000 -and $counts.thick -ge 4000 -and $counts.total -ge 100000) {
    "Generation target reached. Proceeding to setup verification and analysis..." | Tee-Object -FilePath $logPath -Append
    break
  }
  Start-Sleep -Seconds 60
}

# Verify/repair GPU environment
$venvPython = Join-Path $repo ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  python -m venv (Join-Path $repo ".venv-gpu")
}

& $venvPython -m pip install -e ".[ml]"
& $venvPython -m pip install -r "requirements-gpu.txt"
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu121
& $venvPython -m ipykernel install --user --name motionanalyzer-gpu --display-name "Python (motionanalyzer GPU)"

# Run analysis and regenerate report
& $venvPython "scripts/analyze_crack_detection.py"
& $venvPython "scripts/generate_final_report_docx.py"

"=== Monitor pipeline completed ===" | Tee-Object -FilePath $logPath -Append
