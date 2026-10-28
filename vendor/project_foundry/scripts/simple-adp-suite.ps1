param(
    [string]$ResultsDir = 'benchmark-results/coordinated-suite',
    [string]$RunRoot = '',
    [string]$SourceRunRoot = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$SuiteArgs = @()
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot
. (Join-Path $PSScriptRoot 'simple-adp-watchdog.ps1')

$Python = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
if (-not $RunRoot) { $RunRoot = Join-Path $ResultsDir 'run_root' }
if (-not $SourceRunRoot) { $SourceRunRoot = Join-Path $ResultsDir 'source_run_root' }

function Test-RunnerTrue([string]$Value) {
    return $Value.Trim().ToLowerInvariant() -in @('1', 'true', 'yes', 'on', 'y')
}

function ConvertTo-WindowsCommandLineArgument([string]$Value) {
    $builder = [System.Text.StringBuilder]::new('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') { $slashes++; continue }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * ($slashes * 2 + 1)))
            [void]$builder.Append('"')
            $slashes = 0
            continue
        }
        if ($slashes -gt 0) { [void]$builder.Append(('\' * $slashes)); $slashes = 0 }
        [void]$builder.Append($character)
    }
    if ($slashes -gt 0) { [void]$builder.Append(('\' * ($slashes * 2))) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

$StopFile = Join-Path $env:TEMP 'simple-adp-thermal.stop'
$ThermalScript = Join-Path $PSScriptRoot 'simple-adp-thermal-guard.ps1'
$AwakeScript = Join-Path $PSScriptRoot 'strong_awake_guard.ps1'
$ThermalProcess = $null
$AwakeProcess = $null
$LauncherProcess = $null
$WatchdogJob = $null
$PreviousScheme = $null
$ExitCode = 1
$EnableThermal = $true
if (-not [string]::IsNullOrWhiteSpace($env:SIMPLE_ADP_ENABLE_THERMAL_GUARD)) {
    $EnableThermal = Test-RunnerTrue $env:SIMPLE_ADP_ENABLE_THERMAL_GUARD
}

Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
try {
    if (Get-Command powercfg -ErrorAction SilentlyContinue) {
        try {
            $active = (& powercfg /getactivescheme 2>$null | Out-String)
            if ($active -match '([0-9a-fA-F-]{36})') { $PreviousScheme = $Matches[1] }
        } catch { $PreviousScheme = $null }
    }
    if ($EnableThermal -and (Test-Path -LiteralPath $ThermalScript)) {
        $ThermalProcess = Start-Process -FilePath 'powershell.exe' -WindowStyle Minimized -PassThru -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$ThermalScript`"",
            '-StopFile', "`"$StopFile`""
        )
    }
    if (Test-Path -LiteralPath $AwakeScript) {
        $AwakeProcess = Start-Process -FilePath 'powershell.exe' -WindowStyle Minimized -PassThru -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$AwakeScript`""
        )
    }

    $env:PYTHONPATH = "$RepositoryRoot\src;$RepositoryRoot;$env:PYTHONPATH"
    $LogPath = Join-Path $RunRoot 'launcher.log'
    New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
    $Arguments = @('-u', '-m', 'simple_adp.runner.launcher', '--output', $ResultsDir,
        '--run-root', $RunRoot, '--source-run-root', $SourceRunRoot) + $SuiteArgs
    $ArgumentLine = ($Arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument ([string]$_) }) -join ' '
    $LauncherProcess = Start-Process -FilePath $Python -ArgumentList $ArgumentLine -WorkingDirectory $RepositoryRoot `
        -RedirectStandardOutput $LogPath -RedirectStandardError (Join-Path $RunRoot 'launcher.stderr.log') -PassThru
    $WatchdogJob = Start-SimpleADPMasterWatchdog -MasterPid $PID -LauncherPid $LauncherProcess.Id
    $LauncherProcess.WaitForExit()
    $ExitCode = $LauncherProcess.ExitCode
    if ($ExitCode -ne 0) { throw "simple-adp suite launcher exited with code $ExitCode" }
} finally {
    if ($EnableThermal -and $StopFile) {
        Set-Content -LiteralPath $StopFile -Value 'stop' -ErrorAction SilentlyContinue
    }
    if ($ThermalProcess -and -not $ThermalProcess.HasExited) {
        Stop-Process -Id $ThermalProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($AwakeProcess -and -not $AwakeProcess.HasExited) {
        Stop-Process -Id $AwakeProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($WatchdogJob) { Stop-SimpleADPMasterWatchdog -WatchdogJob $WatchdogJob }
    if ($LauncherProcess -and -not $LauncherProcess.HasExited) {
        Stop-Process -Id $LauncherProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($PreviousScheme -and (Get-Command powercfg -ErrorAction SilentlyContinue)) {
        try { & powercfg /setactive $PreviousScheme 2>$null | Out-Null } catch { }
    }
    Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
}

exit $ExitCode
