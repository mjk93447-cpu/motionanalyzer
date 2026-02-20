<#
.SYNOPSIS
    Test EXE build and verify executables.

.DESCRIPTION
    Verifies that EXE files are built correctly and can be executed.
    Checks file existence, size, and basic functionality.

.PARAMETER ExeDir
    Directory containing EXE files (default: dist)

.PARAMETER SkipExecutionTest
    Skip actual execution test (useful for CI/CD)
#>

Param(
    [string]$ExeDir = "dist",
    [switch]$SkipExecutionTest = $false
)

$ErrorActionPreference = "Stop"

Write-Host "EXE Build Test and Verification"
Write-Host "=" * 60
Write-Host ""

$exeDirPath = Resolve-Path $ExeDir -ErrorAction SilentlyContinue
if (-not $exeDirPath) {
    Write-Host "[ERROR] Directory not found: $ExeDir"
    Write-Host "Please build EXE files first: .\scripts\build_exe.ps1"
    exit 1
}

$exeDirPath = $exeDirPath.Path
Write-Host "Checking EXE directory: $exeDirPath"
Write-Host ""

$allPassed = $true

# Check lightweight GUI EXE
$lightweightExe = Join-Path $exeDirPath "motionanalyzer-gui.exe"
if (Test-Path $lightweightExe) {
    $fileInfo = Get-Item $lightweightExe
    $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    Write-Host "[PASS] Lightweight GUI EXE found: $lightweightExe"
    Write-Host "       Size: $sizeMB MB"
    
    if ($sizeMB -gt 200) {
        Write-Host "[WARNING] File size is larger than expected (~50-100MB)"
    }
} else {
    Write-Host "[FAIL] Lightweight GUI EXE not found: $lightweightExe"
    $allPassed = $false
}

Write-Host ""

# Check ML-enabled GUI EXE
$mlExe = Join-Path $exeDirPath "motionanalyzer-gui-ml.exe"
if (Test-Path $mlExe) {
    $fileInfo = Get-Item $mlExe
    $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    Write-Host "[PASS] ML-enabled GUI EXE found: $mlExe"
    Write-Host "       Size: $sizeMB MB"
    
    if ($sizeMB -lt 200) {
        Write-Host "[WARNING] File size is smaller than expected (~200-500MB for ML version)"
    }
} else {
    Write-Host "[WARNING] ML-enabled GUI EXE not found: $mlExe"
    Write-Host "          This is optional. Build with: .\scripts\build_exe.ps1 -IncludeML"
}

Write-Host ""

# Execution test (if not skipped)
if (-not $SkipExecutionTest) {
    Write-Host "Execution Test"
    Write-Host "-" * 60
    
    if (Test-Path $lightweightExe) {
        Write-Host "Testing lightweight EXE (will open GUI window)..."
        Write-Host "Note: GUI window will open. Please close it manually to continue."
        
        try {
            $process = Start-Process -FilePath $lightweightExe -PassThru -WindowStyle Normal
            Start-Sleep -Seconds 2
            
            if (-not $process.HasExited) {
                Write-Host "[PASS] Lightweight EXE started successfully"
                Write-Host "       Process ID: $($process.Id)"
                Write-Host "       Please close the GUI window to continue..."
                
                # Wait for user to close (max 30 seconds)
                $process.WaitForExit(30000)
                if (-not $process.HasExited) {
                    Write-Host "[WARNING] GUI still running after 30 seconds"
                    Write-Host "          Stopping process..."
                    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                }
            } else {
                Write-Host "[FAIL] Lightweight EXE exited immediately (exit code: $($process.ExitCode))"
                $allPassed = $false
            }
        } catch {
            Write-Host "[FAIL] Failed to execute lightweight EXE: $_"
            $allPassed = $false
        }
    }
}

Write-Host ""
Write-Host "=" * 60
if ($allPassed) {
    Write-Host "[SUCCESS] All EXE build tests passed!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Test GUI functionality manually"
    Write-Host "  2. Commit changes: git add . && git commit -m 'Prepare v0.2.0 release'"
    Write-Host "  3. Create tag: git tag v0.2.0"
    Write-Host "  4. Push: git push origin main && git push origin v0.2.0"
    exit 0
} else {
    Write-Host "[FAILURE] Some EXE build tests failed"
    Write-Host "          Please fix issues before deployment"
    exit 1
}
