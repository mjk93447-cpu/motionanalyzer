# Export pretrained models + bundle_manifest for GitHub Release / PyInstaller bundle.
Param(
    [string]$SourceDir = "",
    [string]$OutZip = "dist/motionanalyzer-models.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $SourceDir) {
    $repoModels = Join-Path $root "release\models"
    if (Test-Path (Join-Path $repoModels "bundle_manifest.json")) {
        $SourceDir = $repoModels
    } else {
        $appData = $env:APPDATA
        if (-not $appData) { throw "APPDATA not set; pass -SourceDir or train into release/models first." }
        $SourceDir = Join-Path $appData "motionanalyzer\models"
    }
}

$src = Resolve-Path $SourceDir -ErrorAction Stop
$releaseModels = Join-Path $root "release\models"
New-Item -ItemType Directory -Force -Path $releaseModels | Out-Null

$names = @(
    "draem_model.pt",
    "patchcore_model.npz",
    "bundle_manifest.json",
    "ensemble_config.json"
)
foreach ($n in $names) {
    $f = Join-Path $src $n
    if (Test-Path $f) {
        Copy-Item -Force $f (Join-Path $releaseModels $n)
        Write-Host "Copied $n"
    } else {
        Write-Host "Skip (missing): $n"
    }
}

$distDir = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $distDir | Out-Null
if (Test-Path $OutZip) { Remove-Item -Force $OutZip }
Compress-Archive -Path (Join-Path $releaseModels "*") -DestinationPath $OutZip -Force
Write-Host "Release bundle: $OutZip"
Write-Host "Also staged under: $releaseModels"
