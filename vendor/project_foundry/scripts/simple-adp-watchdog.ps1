# Dot-source and call Start-SimpleADPMasterWatchdog before waiting for the
# launcher. The background job owns only the launcher PID supplied by caller.
function Start-SimpleADPMasterWatchdog {
    param(
        [Parameter(Mandatory)][int]$MasterPid,
        [Parameter(Mandatory)][int]$LauncherPid,
        [int]$GraceSeconds = 15,
        [int]$PollSeconds = 3
    )

    $jobScript = {
        param($masterPid, $launcherPid, $graceSeconds, $pollSeconds)

        function Get-SimpleADPDescendantIds([int]$RootPid) {
            $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
            $pending = [System.Collections.Generic.Queue[int]]::new()
            $seen = [System.Collections.Generic.HashSet[int]]::new()
            $pending.Enqueue($RootPid)
            while ($pending.Count -gt 0) {
                $parent = $pending.Dequeue()
                foreach ($process in $processes) {
                    if ([int]$process.ParentProcessId -ne $parent) { continue }
                    $child = [int]$process.ProcessId
                    if ($seen.Add($child)) { $pending.Enqueue($child) }
                }
            }
            return @($seen)
        }

        function Stop-SimpleADPProcessTree([int]$RootPid) {
            $descendants = @(Get-SimpleADPDescendantIds -RootPid $RootPid)
            foreach ($childPid in ($descendants | Sort-Object -Descending)) {
                Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue
            }
            Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
        }

        Start-Sleep -Seconds 3
        while ($true) {
            Start-Sleep -Seconds $pollSeconds
            $masterAlive = $false
            try {
                $master = Get-Process -Id $masterPid -ErrorAction Stop
                $masterAlive = -not $master.HasExited
            } catch { $masterAlive = $false }
            if (-not $masterAlive) {
                $stamp = Get-Date -Format 'HH:mm:ss'
                Write-Host "[$stamp][WATCHDOG] Master PID $masterPid gone. Stopping owned launcher PID $launcherPid..."
                Stop-Process -Id $launcherPid -ErrorAction SilentlyContinue
                Start-Sleep -Seconds $graceSeconds
                Stop-SimpleADPProcessTree -RootPid $launcherPid
                $stamp = Get-Date -Format 'HH:mm:ss'
                Write-Host "[$stamp][WATCHDOG] Orphan cleanup complete."
                break
            }
        }
    }

    $job = Start-Job -ScriptBlock $jobScript -ArgumentList $MasterPid, $LauncherPid, $GraceSeconds, $PollSeconds
    $sourceIdentifier = "SimpleADP.PowerShell.Exiting.$LauncherPid"
    $null = Register-EngineEvent -SourceIdentifier $sourceIdentifier -MessageData $LauncherPid -Action {
        Stop-Process -Id ([int]$event.MessageData) -Force -ErrorAction SilentlyContinue
    }
    $job | Add-Member -NotePropertyName SourceIdentifier -NotePropertyValue $sourceIdentifier -Force
    return $job
}

function Stop-SimpleADPMasterWatchdog {
    param([Parameter(Mandatory)]$WatchdogJob)
    try { Stop-Job -Job $WatchdogJob -ErrorAction SilentlyContinue } catch { }
    try { Remove-Job -Job $WatchdogJob -Force -ErrorAction SilentlyContinue } catch { }
    try {
        if ($WatchdogJob.SourceIdentifier) {
            Unregister-Event -SourceIdentifier $WatchdogJob.SourceIdentifier -ErrorAction SilentlyContinue
        }
    } catch { }
}
