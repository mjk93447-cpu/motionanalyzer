Param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment (.venv)"
& $PythonExe -m venv .venv

Write-Host "==> Activating virtual environment"
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip"
python -m pip install --upgrade pip

Write-Host "==> Installing project with dev dependencies"
python -m pip install -e ".[dev]"

Write-Host "==> Installing pre-commit hooks"
pre-commit install

Write-Host "==> Creating standard data directories"
motionanalyzer init-dirs

Write-Host "Bootstrap complete."
