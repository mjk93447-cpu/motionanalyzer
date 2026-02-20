<#
.SYNOPSIS
    Track GitHub Actions build and create release when complete.

.DESCRIPTION
    Polls GitHub Actions API to check build status and creates release when build completes successfully.
    Requires GITHUB_TOKEN environment variable.

.PARAMETER Tag
    Release tag (e.g., v0.2.0)

.PARAMETER ReleaseNotesPath
    Path to release notes markdown file

.PARAMETER PollInterval
    Polling interval in seconds (default: 30)

.PARAMETER MaxWaitTime
    Maximum wait time in minutes (default: 30)
#>

Param(
    [Parameter(Mandatory=$true)]
    [string]$Tag,
    
    [Parameter(Mandatory=$true)]
    [string]$ReleaseNotesPath,
    
    [int]$PollInterval = 30,
    
    [int]$MaxWaitTime = 30
)

$ErrorActionPreference = "Stop"

$owner = "mjk93447-cpu"
$repo = "motionanalyzer"
$token = $env:GITHUB_TOKEN

if (-not $token) {
    Write-Host "[ERROR] GitHub token not found. Please set GITHUB_TOKEN environment variable."
    Write-Host ""
    Write-Host "To create a GitHub token:"
    Write-Host "  1. Go to https://github.com/settings/tokens"
    Write-Host "  2. Generate new token (classic) with 'repo' scope"
    Write-Host "  3. Set environment variable: `$env:GITHUB_TOKEN = 'your-token'"
    exit 1
}

$headers = @{
    "Authorization" = "token $token"
    "Accept" = "application/vnd.github.v3+json"
}

$startTime = Get-Date
$maxWaitTimeSpan = New-TimeSpan -Minutes $MaxWaitTime

Write-Host "Tracking GitHub Actions build for tag: $Tag"
Write-Host "Polling interval: $PollInterval seconds"
Write-Host "Maximum wait time: $MaxWaitTime minutes"
Write-Host ""

while ($true) {
    $elapsed = (Get-Date) - $startTime
    if ($elapsed -gt $maxWaitTimeSpan) {
        Write-Host "[ERROR] Maximum wait time exceeded ($MaxWaitTime minutes)"
        exit 1
    }

    try {
        $response = Invoke-RestMethod -Uri "https://api.github.com/repos/$owner/$repo/actions/runs?branch=main&per_page=1" -Headers $headers
        
        if ($response.workflow_runs.Count -eq 0) {
            Write-Host "[WARNING] No workflow runs found"
            Start-Sleep -Seconds $PollInterval
            continue
        }

        $run = $response.workflow_runs[0]
        $status = $run.status
        $conclusion = $run.conclusion
        $runId = $run.id
        $htmlUrl = $run.html_url

        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Build Status: $status" -NoNewline
        
        if ($conclusion) {
            Write-Host " | Conclusion: $conclusion"
        } else {
            Write-Host ""
        }

        if ($status -eq "completed") {
            if ($conclusion -eq "success") {
                Write-Host ""
                Write-Host "[SUCCESS] Build completed successfully!"
                Write-Host "Build URL: $htmlUrl"
                Write-Host ""
                
                # Create release
                Write-Host "Creating GitHub release..."
                
                if (-not (Test-Path $ReleaseNotesPath)) {
                    Write-Host "[ERROR] Release notes file not found: $ReleaseNotesPath"
                    exit 1
                }

                $releaseNotes = Get-Content $ReleaseNotesPath -Raw -Encoding UTF8

                $body = @{
                    tag_name = $Tag
                    name = $Tag
                    body = $releaseNotes
                    draft = $false
                    prerelease = $false
                } | ConvertTo-Json

                $releaseUrl = "https://api.github.com/repos/$owner/$repo/releases"

                try {
                    $releaseResponse = Invoke-RestMethod -Uri $releaseUrl -Method Post -Headers $headers -Body $body -ContentType "application/json"
                    
                    Write-Host "[SUCCESS] Release created successfully!"
                    Write-Host "Release URL: $($releaseResponse.html_url)"
                    Write-Host ""
                    Write-Host "Next steps:"
                    Write-Host "  1. Download EXE artifacts from Actions: $htmlUrl"
                    Write-Host "  2. Upload EXE files to the release: $($releaseResponse.html_url)"
                    exit 0
                } catch {
                    Write-Host "[ERROR] Failed to create release: $_"
                    if ($_.ErrorDetails.Message) {
                        Write-Host "Error details: $($_.ErrorDetails.Message)"
                    }
                    exit 1
                }
            } elseif ($conclusion -eq "failure") {
                Write-Host ""
                Write-Host "[FAILURE] Build failed!"
                Write-Host "Build URL: $htmlUrl"
                Write-Host ""
                Write-Host "Please check the build logs for details."
                exit 1
            } else {
                Write-Host ""
                Write-Host "[WARNING] Build completed with conclusion: $conclusion"
                Write-Host "Build URL: $htmlUrl"
                exit 1
            }
        } elseif ($status -eq "in_progress" -or $status -eq "queued") {
            Write-Host "  Waiting for build to complete... (elapsed: $([math]::Round($elapsed.TotalMinutes, 1)) minutes)"
            Start-Sleep -Seconds $PollInterval
        } else {
            Write-Host ""
            Write-Host "[WARNING] Unexpected build status: $status"
            Write-Host "Build URL: $htmlUrl"
            Start-Sleep -Seconds $PollInterval
        }
    } catch {
        Write-Host "[ERROR] Failed to check build status: $_"
        Start-Sleep -Seconds $PollInterval
    }
}
