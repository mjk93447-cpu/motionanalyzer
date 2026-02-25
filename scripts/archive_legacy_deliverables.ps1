# Archive existing deliverables before writing new paper
# Run before: run_paper_pipeline.ps1
# Usage: .\scripts\archive_legacy_deliverables.ps1

$ErrorActionPreference = "Stop"
$repo = (Get-Item $PSScriptRoot).Parent.FullName
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$dst = Join-Path $repo "reports\archive\legacy_deliverables\$ts"
New-Item -ItemType Directory -Path $dst -Force | Out-Null

$moved = 0
$items = @(
    @{Path="reports\deliverables\FPCB_Crack_Detection_Final_Report.docx"; Move=$true},
    @{Path="reports\deliverables\FPCB_Crack_Detection_Final_Report.pdf"; Move=$true},
    @{Path="reports\deliverables\FPCB_Crack_Detection_Final_Report.pptx"; Move=$true},
    @{Path="reports\crack_detection_analysis\insights_summary.png"; Move=$false},
    @{Path="reports\crack_detection_analysis\analysis.json"; Move=$false},
    @{Path="reports\crack_detection_analysis\insights.md"; Move=$false},
    @{Path="reports\crack_detection_analysis\confusion_matrix_dream.png"; Move=$false},
    @{Path="reports\crack_detection_analysis\confusion_matrix_patchcore.png"; Move=$false},
    @{Path="reports\crack_detection_analysis\confusion_matrix_ensemble.png"; Move=$false},
    @{Path="reports\crack_detection_analysis\vector_map_normal.png"; Move=$false},
    @{Path="reports\crack_detection_analysis\vector_map_crack.png"; Move=$false}
)
foreach ($item in $items) {
    $src = Join-Path $repo $item.Path
    if (Test-Path $src) {
        if ($item.Move) { Move-Item $src $dst -Force } else { Copy-Item $src $dst -Force }
        Write-Host "Archived: $($item.Path)" -ForegroundColor Green
        $moved++
    }
}
if (Test-Path (Join-Path $repo "reports\deliverables\videos")) {
    $vDst = Join-Path $dst "videos"
    New-Item -ItemType Directory -Path $vDst -Force | Out-Null
    Copy-Item (Join-Path $repo "reports\deliverables\videos\*") $vDst -Force -ErrorAction SilentlyContinue
    Write-Host "Archived: reports/deliverables/videos/" -ForegroundColor Green
    $moved++
}
Write-Host "`nArchived $moved items to $dst" -ForegroundColor Cyan
Write-Host "REFERENCE ONLY - Do not reuse numbers/figures in new paper. Use fresh analysis output." -ForegroundColor Yellow
