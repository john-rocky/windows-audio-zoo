<#
Untimed steps of one list (pc\steps\<name>.json, from the Mac), one after another; Windows PowerShell 5.1. Started
detached (pc\launch.ps1, without yield_run): a step stops for another lane's lock by being killed, not suspended (the
shared PC's rule: a suspended process keeps its RAM; 11-09 13:00, three suspended Z3 processes held 8.6 GB under another lane's
timing), and runs again from its start once busy.txt has been free 60 s in a row. A step with "lock": true is not
stopped: it runs under Z3's own lock.

  powershell -NoProfile -ExecutionPolicy Bypass -File pc\steps.ps1 -List <name>

A step: { name, exe ("py" = .venv-<venv>\Scripts\python.exe -I, "uv", "ps" = a .ps1 here, or a path), venv, args
(with {ROOT} {HOME} {FIX}), timeout_min, ram_gb, lock, stop_on_fail }.
A step with "lock": true (the queue's rule for a step over 8 GB that fits no window; its slot comes from the PC's
order) takes busy.txt once, after it has been free 60 s in a row, as "Z3 <name> (not a timing) until HH:MM" (its
timeout), checks the free RAM 10 s later (short: releases and waits for the next free lock), runs, and releases.
Logs: logs\<name>.log (one line per step), logs\<name>\<step>.txt (its output); logs\<name>.done = "OK" or
"FAIL <steps>" and the time.
#>
param([Parameter(Mandatory = $true)][string]$List, [string]$KeepUntil = "")
# -KeepUntil HH:mm: a list that ends holding Z3's lock leaves it held as "first pass (not a timing) until HH:mm" (no free
# gap before the first passes of Z3's slot in the PC's order); the Mac releases it
$KeepLockAs = if ($KeepUntil) { "first pass (not a timing) until $KeepUntil" } else { "" }
. "$PSScriptRoot\zlib.ps1"
$ErrorActionPreference = "Continue"
$listFile = Join-Path $Root "pc\steps\$List.json"
$ldir = Join-Path $Root "logs\$List"
New-Item -ItemType Directory -Force $ldir | Out-Null
$log = Join-Path $Root "logs\$List.log"
$done = Join-Path $Root "logs\$List.done"
if (Test-Path $done) { Remove-Item $done -Force }
if (-not (Test-Path -LiteralPath $listFile)) {   # a mistyped list name would otherwise run 0 steps and say OK
    Say $log "steps $($List): no list file pc\steps\$List.json"
    Write-Utf8 $done "FAIL no list file pc\steps\$List.json $(Get-Date -Format o)"
    exit 1
}
$steps = @(Get-Content $listFile -Raw -Encoding UTF8 | ConvertFrom-Json)
$steps = @($steps | ForEach-Object { $_ })   # PowerShell 5.1 reads a JSON list as one object
$Fix = Join-Path $Root "fixtures"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe" }
$env:UV_LINK_MODE = "copy"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "python"   # Z3's own interpreter: a per-app GPU preference names only Z3's exe
$env:UV_CACHE_DIR = Join-Path $Root "uvcache"
# a download that yield_run suspends dies of uv's 30 s read timeout on the resume (11-09 12:11: nvidia-cudnn-cu13 in
# pip-ort-cuda after another lane's 17-minute lock): a longer timeout and more retries
$env:UV_HTTP_TIMEOUT = "600"; $env:UV_HTTP_RETRIES = "8"
$env:HF_HUB_DISABLE_TELEMETRY = "1"; $env:ORT_DISABLE_TELEMETRY = "1"; $env:PYTHONUTF8 = "1"
Say $log "steps $List start: $($steps.Count) steps, pid $PID"
# the presence note of untimed work (the shared PC's rule: what and since when, rewritten as it goes, removed at the end)
$note = Join-Path $Pcup "active\Z3-$List.txt"
$since = Get-Date -Format o
$fail = @()
for ($i = 0; $i -lt $steps.Count; $i++) {
    $s = $steps[$i]
    $next = if ($i + 1 -lt $steps.Count) { $steps[$i + 1] } else { $null }
    Write-Utf8 $note "Z3 untimed steps '$List' (download / venv / export / reference; killed and run again around other lanes' locks), now $($s.name), since $since`r`nwritten $(Get-Date -Format HH:mm)"
    $a = @(@($s.args) | ForEach-Object { "$_".Replace("{ROOT}", $Root).Replace("{HOME}", $HomeDir).Replace("{FIX}", $Fix) })
    $exe = switch ($s.exe) {
        "py" { Join-Path $Root ".venv-$($s.venv)\Scripts\python.exe" }
        "uv" { $uv }
        "ps" { "powershell" }
        default { "$($s.exe)".Replace("{ROOT}", $Root) }
    }
    if ($s.exe -eq "py") { $a = @("-I") + $a }
    if ($s.exe -eq "ps") { $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "pc\$($a[0])")) + @($a | Select-Object -Skip 1) }
    $tmin = if ($s.timeout_min) { [int]$s.timeout_min } else { 60 }
    $held = $false
    if ($s.lock -and (Test-OursBusy)) {
        $held = $true   # kept from the list's previous lock step: the lock's text is rewritten, never a free gap
    } elseif ($s.lock) {
        while ($true) {
            $r = Take-BusyAfterFree "$List $($s.name) (not a timing) until $((Get-Date).AddMinutes($tmin + 2).ToString('HH:mm'))" $log 60 12
            if ($r -eq "TIMEOUT") { break }
            Start-Sleep -Seconds 10
            if (-not $s.ram_gb -or (Get-AvailGB) -ge [double]$s.ram_gb) { $held = $true; break }
            Say $log "$($s.name): $(Get-AvailGB) GB free under its $($s.ram_gb) GB: lock given back, trying at the next free lock"
            Release-Busy $log
            Start-Sleep -Seconds 120
        }
        if (-not $held) { $fail += $s.name; Say $log "$($s.name): no lock in 12 h"; continue }
    }
    if ($held) {
        Touch-Busy "$List $($s.name) (not a timing) until $((Get-Date).AddMinutes($tmin + 1).ToString('HH:mm'))"
    } elseif ($s.ram_gb -and [double]$s.ram_gb -gt 8) {
        [void](Wait-FreeRam ([double]$s.ram_gb) $log $s.name)
    }
    $out = Join-Path $ldir "$($s.name).txt"
    $quoted = @($a | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } })
    $sw = [Diagnostics.Stopwatch]::StartNew()
    for ($try = 1; $try -le 30; $try++) {
        if (-not $held) { Wait-OthersGone $log $s.name 60 }
        Say $log "$($s.name) start$(if ($try -gt 1) { " (try $try)" }): $exe $($quoted -join ' ')"
        $proc = Start-Process -FilePath $exe -ArgumentList ($quoted -join " ") -RedirectStandardOutput $out -RedirectStandardError "$out.err" -PassThru -WindowStyle Hidden
        $null = $proc.Handle
        $t0 = Get-Date; $code = $null; $stopped = $false
        while (-not $proc.WaitForExit(2000)) {
            if (-not $held -and (Test-OtherBusy)) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $proc.Id /T /F 2>&1 | Out-Null
                $proc.WaitForExit()
                $stopped = $true
                Say $log "$($s.name): killed for another lane's lock ($((Read-Busy).Text)) after $([int]((Get-Date) - $t0).TotalSeconds) s; runs again when it is gone"
                break
            }
            if (((Get-Date) - $t0).TotalMinutes -ge $tmin) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $proc.Id /T /F 2>&1 | Out-Null
                $proc.WaitForExit()
                $code = "timeout"
                break
            }
        }
        if (-not $stopped) { break }
    }
    if ($stopped) { $code = "killed 30 times for other lanes' locks" }
    elseif ($null -eq $code) { $code = $proc.ExitCode }
    if ($held -and -not ($next -and $next.lock)) {
        if ($KeepLockAs -and -not $next) { Touch-Busy $KeepLockAs; Say $log "lock kept as: $KeepLockAs" }
        else { Release-Busy $log }
    }
    Say $log "$($s.name) end: exit $code, $([math]::Round($sw.Elapsed.TotalMinutes, 1)) min"
    if ("$code" -ne "0") {
        $fail += $s.name
        Get-Content "$out.err" -Tail 5 -ErrorAction SilentlyContinue | ForEach-Object { Say $log "  $_" }
        if ($s.stop_on_fail) { Say $log "stop: $($s.name) failed"; break }
    }
}
Remove-Item $note -Force -ErrorAction SilentlyContinue
$st = if ($fail.Count) { "FAIL $($fail -join ',')" } else { "OK" }
Write-Utf8 $done "$st $(Get-Date -Format o)"
Say $log "steps $List done: $st"
