# Simple script to wait for build completion
# Get latest run ID from API
$response = Invoke-RestMethod -Uri "https://api.github.com/repos/mjk93447-cpu/motionanalyzer/actions/runs?branch=main&per_page=1" -Headers @{"Accept"="application/vnd.github.v3+json"}
$runId = $response.workflow_runs[0].id
Write-Host "Tracking latest build run ID: $runId"
$maxAttempts = 100
$attempt = 0

while ($attempt -lt $maxAttempts) {
    $attempt++
    Start-Sleep -Seconds 30
    
    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/mjk93447-cpu/motionanalyzer/actions/runs/$runId" -Headers @{"Accept"="application/vnd.github.v3+json"}
        $status = $response.status
        $conclusion = $response.conclusion
        
        Write-Host "[$attempt/$maxAttempts] Status: $status" -NoNewline
        if ($conclusion) {
            Write-Host " | Conclusion: $conclusion"
            if ($status -eq "completed") {
                if ($conclusion -eq "success") {
                    Write-Host "`n[SUCCESS] Build completed!"
                    exit 0
                } else {
                    Write-Host "`n[FAILURE] Build failed"
                    exit 1
                }
            }
        } else {
            Write-Host ""
        }
    } catch {
        Write-Host "[ERROR] Failed to check: $_"
        Start-Sleep -Seconds 10
    }
}

Write-Host "[TIMEOUT] Maximum attempts reached"
exit 1
