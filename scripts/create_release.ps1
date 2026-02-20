<#
.SYNOPSIS
    Create GitHub Release using GitHub API.

.DESCRIPTION
    Creates a GitHub release for the specified tag with release notes.
    Requires GITHUB_TOKEN environment variable or GitHub CLI authentication.

.PARAMETER Tag
    Release tag (e.g., v0.2.0)

.PARAMETER ReleaseNotesPath
    Path to release notes markdown file

.PARAMETER Token
    GitHub personal access token (optional, uses GITHUB_TOKEN env var if not provided)
#>

Param(
    [Parameter(Mandatory=$true)]
    [string]$Tag,
    
    [Parameter(Mandatory=$true)]
    [string]$ReleaseNotesPath,
    
    [string]$Token = $env:GITHUB_TOKEN
)

$ErrorActionPreference = "Stop"

if (-not $Token) {
    Write-Host "[ERROR] GitHub token not found. Please set GITHUB_TOKEN environment variable or provide -Token parameter."
    Write-Host ""
    Write-Host "To create a GitHub token:"
    Write-Host "  1. Go to https://github.com/settings/tokens"
    Write-Host "  2. Generate new token (classic) with 'repo' scope"
    Write-Host "  3. Set environment variable: `$env:GITHUB_TOKEN = 'your-token'"
    exit 1
}

if (-not (Test-Path $ReleaseNotesPath)) {
    Write-Host "[ERROR] Release notes file not found: $ReleaseNotesPath"
    exit 1
}

$releaseNotes = Get-Content $ReleaseNotesPath -Raw -Encoding UTF8

$owner = "mjk93447-cpu"
$repo = "motionanalyzer"

$headers = @{
    "Authorization" = "token $Token"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    tag_name = $Tag
    name = $Tag
    body = $releaseNotes
    draft = $false
    prerelease = $false
} | ConvertTo-Json

$url = "https://api.github.com/repos/$owner/$repo/releases"

Write-Host "Creating GitHub release for tag: $Tag"
Write-Host "URL: $url"
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body -ContentType "application/json"
    
    Write-Host "[SUCCESS] Release created successfully!"
    Write-Host "Release URL: $($response.html_url)"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Wait for GitHub Actions build to complete"
    Write-Host "  2. Download EXE artifacts from Actions"
    Write-Host "  3. Upload EXE files to the release: $($response.html_url)"
} catch {
    Write-Host "[ERROR] Failed to create release: $_"
    Write-Host ""
    Write-Host "Response: $($_.Exception.Response)"
    if ($_.ErrorDetails.Message) {
        Write-Host "Error details: $($_.ErrorDetails.Message)"
    }
    exit 1
}
