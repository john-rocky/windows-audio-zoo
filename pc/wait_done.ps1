<#
One long SSH session from the Mac waits here for a chain: prints each new line of the chain's log as it lands and
returns when the done file exists (printing it). No WMI, a file read every 15 s; the Mac pipes the output to a log.
Several jobs in one session: -Done and -Log each take a comma-separated list; it returns when every done file exists.

  wait_done.ps1 -Done <file>[,<file>...] -Log <file>[,<file>...] [-MaxHours 12]
#>
param([Parameter(Mandatory = $true)][string]$Done, [Parameter(Mandatory = $true)][string]$Log, [int]$MaxHours = 12)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$deadline = (Get-Date).AddHours($MaxHours)
$logs = @($Log -split ",")
$dones = @($Done -split ",")
$pos = @{}
foreach ($l in $logs) { $pos[$l] = [int64]0 }
function Show-New {
    foreach ($l in $logs) {
        if (-not (Test-Path $l)) { continue }
        try {
            $fs = [IO.File]::Open($l, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
            if ($fs.Length -gt $pos[$l]) {
                $fs.Seek($pos[$l], [IO.SeekOrigin]::Begin) | Out-Null
                $buf = New-Object byte[] ($fs.Length - $pos[$l])
                $n = $fs.Read($buf, 0, $buf.Length)
                $pos[$l] += $n
                [Console]::Out.Write([Text.Encoding]::UTF8.GetString($buf, 0, $n))
                [Console]::Out.Flush()
            }
            $fs.Close()
        } catch { }
    }
}
while ((Get-Date) -lt $deadline) {
    Show-New
    if (-not @($dones | Where-Object { -not (Test-Path $_) }).Count) {
        Show-New
        foreach ($d in $dones) { "DONE $(([IO.File]::ReadAllText($d)).Trim()) ($d)" }
        exit 0
    }
    Start-Sleep -Seconds 15
}
"WAIT-TIMEOUT $(Get-Date -Format o)"
exit 1
