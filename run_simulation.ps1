# FedMed Federated Learning Simulation Runner for Windows PowerShell
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   FedMed Federated Learning Simulation (PowerShell)      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Activate .venv if available
$pythonExe = "python"
if ($PythonExe) {
    if (-not (Test-Path $PythonExe)) {
        throw "The supplied Python executable does not exist: $PythonExe"
    }
    $pythonExe = $PythonExe
    Write-Host "[Env] Using supplied Python: $pythonExe" -ForegroundColor Green
} elseif (Test-Path "$scriptDir\.venv\Scripts\python.exe") {
    $pythonExe = "$scriptDir\.venv\Scripts\python.exe"
    Write-Host "[Env] Using virtual environment: $pythonExe" -ForegroundColor Green
} else {
    Write-Host "[Env] Using default system Python: $pythonExe" -ForegroundColor Yellow
}

# Ensure runtime directories exist
New-Item -ItemType Directory -Force -Path "$scriptDir\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$scriptDir\results" | Out-Null

$serverLog = "$scriptDir\logs\server.log"
$client0Log = "$scriptDir\logs\client_0.log"
$client1Log = "$scriptDir\logs\client_1.log"

if (Test-Path $serverLog) { Remove-Item $serverLog -Force }
if (Test-Path $client0Log) { Remove-Item $client0Log -Force }
if (Test-Path $client1Log) { Remove-Item $client1Log -Force }

function Start-FedMedProcess {
    param(
        [string]$FilePath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$LogFile
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = "/c `"`"$FilePath`" $Arguments > `"$LogFile`" 2>&1`""
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    return [System.Diagnostics.Process]::Start($psi)
}

Write-Host "[1/4] Starting FedMed Server on 127.0.0.1:8080..." -ForegroundColor Cyan
$serverProcess = Start-FedMedProcess -FilePath $pythonExe -Arguments "-u src/server.py" -WorkingDirectory $scriptDir -LogFile $serverLog

# 2. Wait for Server port 8080 to become available using Test-NetConnection (up to 30 seconds)
Write-Host "[2/4] Waiting for server port 8080 to open..." -ForegroundColor Cyan
$serverReady = $false
$timeoutSeconds = 30
$elapsed = 0

while (-not $serverReady -and $elapsed -lt $timeoutSeconds) {
    if ($serverProcess.HasExited) {
        Write-Host "[ERROR] Server process exited prematurely with code $($serverProcess.ExitCode)!" -ForegroundColor Red
        if (Test-Path $serverLog) { Get-Content $serverLog }
        exit 1
    }
    Start-Sleep -Seconds 1
    $elapsed++

    try {
        $testConn = Test-NetConnection -ComputerName 127.0.0.1 -Port 8080 -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($testConn) {
            $serverReady = $true
        }
    } catch {
        # Fallback TCP socket check
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $iar = $tcp.BeginConnect("127.0.0.1", 8080, $null, $null)
            $wait = $iar.AsyncWaitHandle.WaitOne(500, $false)
            if ($wait -and $tcp.Connected) {
                $tcp.EndConnect($iar)
                $tcp.Close()
                $serverReady = $true
            } else {
                $tcp.Close()
            }
        } catch { }
    }
}

if (-not $serverReady) {
    Write-Host "[ERROR] Server failed to start on 127.0.0.1:8080 within $timeoutSeconds seconds." -ForegroundColor Red
    if (-not $serverProcess.HasExited) {
        $serverProcess.Kill()
    }
    if (Test-Path $serverLog) { Get-Content $serverLog }
    exit 1
}

Write-Host "[Server] FedMed server is live and accepting gRPC connections!" -ForegroundColor Green

# 3. Launch the two configured hospitals
Write-Host "[3/4] Launching Client 0 and Client 1..." -ForegroundColor Cyan
$client0Process = Start-FedMedProcess -FilePath $pythonExe -Arguments "-u src/client.py --client-id 0" -WorkingDirectory $scriptDir -LogFile $client0Log
$client1Process = Start-FedMedProcess -FilePath $pythonExe -Arguments "-u src/client.py --client-id 1" -WorkingDirectory $scriptDir -LogFile $client1Log

Write-Host "[4/4] Simulation running. Waiting for all processes to complete..." -ForegroundColor Cyan

# Wait for all background processes to terminate
$serverProcess.WaitForExit()
$client0Process.WaitForExit()
$client1Process.WaitForExit()

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "   Simulation Completed. Inspecting Process Statuses:     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Server Exit Code  : $($serverProcess.ExitCode)" -ForegroundColor $(if ($serverProcess.ExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Client 0 Exit Code: $($client0Process.ExitCode)" -ForegroundColor $(if ($client0Process.ExitCode -eq 0) { "Green" } else { "Red" })
Write-Host "Client 1 Exit Code: $($client1Process.ExitCode)" -ForegroundColor $(if ($client1Process.ExitCode -eq 0) { "Green" } else { "Red" })

if ($serverProcess.ExitCode -eq 0 -and $client0Process.ExitCode -eq 0 -and $client1Process.ExitCode -eq 0) {
    Write-Host "`n[SUCCESS] All federated learning rounds finished successfully across 2 clients!" -ForegroundColor Green
    if (Test-Path "$scriptDir\results\training_history.csv") {
        Write-Host "`n--- Training History (results/training_history.csv) ---" -ForegroundColor Yellow
        Get-Content "$scriptDir\results\training_history.csv"
    }
    exit 0
} else {
    Write-Host "`n[FAIL] One or more processes failed. Inspecting logs:" -ForegroundColor Red
    Write-Host "`n--- Server Log ---"
    if (Test-Path $serverLog) { Get-Content $serverLog }
    Write-Host "`n--- Client 0 Log ---"
    if (Test-Path $client0Log) { Get-Content $client0Log }
    Write-Host "`n--- Client 1 Log ---"
    if (Test-Path $client1Log) { Get-Content $client1Log }
    exit 1
}
