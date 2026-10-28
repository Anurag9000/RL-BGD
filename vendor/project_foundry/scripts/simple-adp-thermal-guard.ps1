param(
    [string]$StopFile = (Join-Path $env:TEMP 'simple-adp-thermal.stop'),
    [string]$LogFile = (Join-Path $env:TEMP 'simple-adp-thermal.log'),
    [double]$HotTempC = 90,
    [double]$CoolTempC = 85,
    [int]$HotCpuMax = 100,
    [int]$NormalCpuMax = 100,
    [int]$PollSeconds = 5
)

$ErrorActionPreference = 'SilentlyContinue'
$BaseScheme = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
$ExecutionStateType = @'
using System;
using System.Runtime.InteropServices;
public static class SimpleAdpExecutionState {
  [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
}
'@
Add-Type -TypeDefinition $ExecutionStateType -ErrorAction SilentlyContinue | Out-Null
$Continuous = [uint32]0x80000000
$SystemRequired = [uint32]0x00000001
$AwayModeRequired = [uint32]0x00000040

function Write-GuardLog([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $LogFile -Value $line
}

function Get-CpuTemperature {
    foreach ($query in @(
        { Get-CimInstance -Namespace 'root/LibreHardwareMonitor' -ClassName Sensor -ErrorAction Stop | Where-Object { $_.SensorType -eq 'Temperature' -and $_.Name -match 'CPU|Package' } | Select-Object -ExpandProperty Value },
        { Get-CimInstance -Namespace 'root/OpenHardwareMonitor' -ClassName Sensor -ErrorAction Stop | Where-Object { $_.SensorType -eq 'Temperature' -and $_.Name -match 'CPU|Package' } | Select-Object -ExpandProperty Value },
        { Get-CimInstance -Namespace 'root/wmi' -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | ForEach-Object { ($_.CurrentTemperature / 10.0) - 273.15 } }
    )) {
        try {
            $values = @(& $query | Where-Object { $_ -is [ValueType] } | ForEach-Object { [double]$_ })
            if ($values.Count -gt 0) { return ($values | Measure-Object -Maximum).Maximum }
        } catch { }
    }
    return $null
}

try {
    $schemes = (powercfg /list) -join "`n"
    $highPerformance = $null
    foreach ($line in ($schemes -split "`n")) {
        if ($line -match '([0-9a-fA-F-]{36}).*High performance') { $highPerformance = $Matches[1]; break }
    }
    if (-not $highPerformance) {
        powercfg /duplicatescheme $BaseScheme | Out-Null
        $created = (powercfg /list) -join "`n"
        foreach ($line in ($created -split "`n")) {
            if ($line -match '([0-9a-fA-F-]{36}).*High performance') { $highPerformance = $Matches[1]; break }
        }
    }
    if ($highPerformance) { powercfg /setactive $highPerformance | Out-Null }

    $settings = @(
        [pscustomobject]@{ Subgroup = 'SUB_VIDEO'; Setting = 'VIDEOIDLE'; Value = 5 },
        [pscustomobject]@{ Subgroup = 'SUB_SLEEP'; Setting = 'STANDBYIDLE'; Value = 0 },
        [pscustomobject]@{ Subgroup = 'SUB_SLEEP'; Setting = 'HIBERNATEIDLE'; Value = 0 },
        [pscustomobject]@{ Subgroup = 'SUB_DISK'; Setting = 'DISKIDLE'; Value = 0 },
        [pscustomobject]@{ Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '29f6c1db-86da-48c5-9fdb-f2b67b1f44da'; Value = 0 },
        [pscustomobject]@{ Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '9d7815a6-7ee4-497e-8888-515a05f02364'; Value = 0 },
        [pscustomobject]@{ Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '94ac6d29-73ce-41a6-809f-6363ba21b47e'; Value = 0 },
        [pscustomobject]@{ Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '7bc4a2f9-d8fc-4469-b07b-33eb785aaca0'; Value = 0 },
        [pscustomobject]@{ Subgroup = '238c9fa8-0aad-41ed-83f4-97be242c8f20'; Setting = '25dfa149-5dd1-4736-b5ab-e8a37b5b8187'; Value = 1 },
        [pscustomobject]@{ Subgroup = '4f971e89-eebd-4455-a8de-9e59040e7347'; Setting = '5ca83367-6e45-459f-a27b-476b1d01c9360'; Value = 0 },
        [pscustomobject]@{ Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = '893dee8e-2bef-41e0-89c6-b55d0929964c'; Value = 5 },
        [pscustomobject]@{ Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = 'bc5038f7-23e0-4960-96da-33abaf5935ec'; Value = $NormalCpuMax },
        [pscustomobject]@{ Subgroup = '54533251-82be-4824-96c1-47b60b740d00'; Setting = '94d3a615-a899-4ac5-ae2b-e4d8f634367f'; Value = 0 }
    )
    while (-not (Test-Path -LiteralPath $StopFile)) {
        if ($highPerformance) {
            foreach ($setting in $settings) {
                powercfg /setacvalueindex $highPerformance $setting.Subgroup $setting.Setting $setting.Value | Out-Null
            }
            powercfg /setactive $highPerformance | Out-Null
        }
        [SimpleAdpExecutionState]::SetThreadExecutionState($Continuous -bor $SystemRequired -bor $AwayModeRequired) | Out-Null
        $temperature = Get-CpuTemperature
        if ($null -ne $temperature) { Write-GuardLog ("cpu_temperature_c={0:N1}" -f $temperature) }
        Start-Sleep -Seconds $PollSeconds
    }
} finally {
    [SimpleAdpExecutionState]::SetThreadExecutionState($Continuous) | Out-Null
}
