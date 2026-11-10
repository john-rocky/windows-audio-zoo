<#
Runs the units of one plan (pc\plans\<sitting>.json, written on the Mac by tools/make_plan.py) on this PC.
Windows PowerShell 5.1. Start it detached (pc\launch.ps1), never from an SSH session that may close.

  powershell -NoProfile -ExecutionPolicy Bypass -File chain.ps1 -Plan <plan.json> -Mode first|timed

A unit = one engine process of one cell (PLAN section 2: load, one untimed warm-up, one pass over the clip set):
{ cell, device (cpu|igpu|rtx|npu), venv, script, args (with {ROOT} {HOME} {FIX} {RUN}), timeout_min, ram_need_gb,
  engines_gate, role (control-warm|control-head|control-tail, optional), runs (default 1) }.
The chain adds "--out <run dir>\run<k>.json" to every unit's arguments.

-Mode first: one run per unit, no lock, active\Z3-<sitting>.txt says what runs; a unit that fails still writes its JSON.
  While another lane holds busy.txt nothing starts; a run in flight is killed when another lane's lock appears (its
  files removed; not suspended: a suspended process keeps its RAM) and runs again once the lock has been free 60 s.
-Mode timed: under the Z3 lock that the Mac took and gated before starting this (tools/sitting.sh). Before each run,
  the bench job's per-run quiet gate (zlib Wait-RunQuiet). The control rows (PLAN section 3, memory "timed sittings
  need a control row"): the plan's control-warm unit (not counted), control-head, the cells, control-tail. The known
  value is the plan's control_known_rtf, or (first sitting) the median of the head's runs when they agree within 5 %;
  a head off the known value by more than 5 % stops the sitting before any cell; the tail's verdict goes to
  runs\<sitting>\control.json. Duty: before a run, while another lane waits and the lock has been held 30 min or the
  run would end past 30 min of the hold, the lock is released for 16.5 min (window.txt), taken again once the other
  lane's lock has been gone 60 s, settled -SettleSeconds (180 s) and gated again. The lock's text says until when the
  run is expected. The sitting also settles 180 s inside the lock before control-warm (the queue's rule, line 5).
  Slots (no window.txt): the plan's slots [{ who, release_at, give_up_at }] (a fixed slot: a run that would end past
  release_at - 60 s does not start; the lock is held idle to release_at), and runs\slot_now.txt ("<who> <give up min>",
  written from the Mac at the user's word: a run in flight gets 8 min to end, then is stopped and run again after the
  slot). At a slot the lock is released, the other lane's lock waited for (until give_up) and for its return (free
  60 s), then taken back, settled 180 s and gated.
  end_by (the plan's ISO time): the PC's order gives the lock to another lane then (11-09: another lane at 21:07, other lanes'
  timings after it). A cell whose runs, a possible duty gap and the control tail would not end before it is not
  started, nor any cell after it; they go to control.json as left_cells (a later sitting runs them) and the tail runs.
  A plan with windows false (tools/make_plan.py --no-windows: the PC's order gives the sitting one block to end_by)
  opens no duty window at all; a slot_now.txt request still applies.
  No duty window opens before the control tail (the tail follows the last cell; the lock goes back after it). A
  take-back whose free minute, settle, gate and tail would not end before end_by is not made (nor a second gate after
  one that did not open): the sitting ends there without its tail row (control.json verdict), the rest left.
  At the end the lock stays ours for -HoldAtEndMin minutes so that the Mac pulls the records inside it (no connection
  after the lock is gone); the chain then releases it if it is still ours.
Every run writes <runs>\<sitting>\<cell>\run<k>.meta.json (the chain's view: gate, load before and after, peak working
set, GPU memory per adapter) next to the runner's run<k>.json; a run already on disk is skipped (resume).
#>
param(
    [Parameter(Mandatory = $true)][string]$Plan,
    [Parameter(Mandatory = $true)][ValidateSet("first", "timed")][string]$Mode,
    [int]$BetweenRunsSeconds = 20,
    [int]$HoldAtEndMin = 20,
    # the PC's rule (11-09): a timing right after another lane's heavy lock waits 180 s inside the lock
    # before its control (no wait: +54 % / +30 %; 180 s: within +4.7 %). Here: before control-warm and after every take-back.
    [int]$SettleSeconds = 180
)
. "$PSScriptRoot\zlib.ps1"
$p = Get-Content $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$sit = $p.sitting
$out = Join-Path $Root "runs\$sit"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "chain.log"
$done = Join-Path $out "chain.done"
if (Test-Path $done) { Remove-Item $done -Force }   # a done file of an earlier start is not this run's
$active = Join-Path $Pcup "active\$Label-$sit.txt"
$Fix = Join-Path $Root "fixtures"
$SlotNow = Join-Path $Root "runs\slot_now.txt"
$RetakeFreeSec = 60
$EndBy = if ($Mode -eq "timed" -and $p.end_by) { [datetime]::Parse($p.end_by) } else { $null }
$script:leftCells = @()
$chainStart = Get-Date -Format o
Say $log "chain start: plan $Plan mode $Mode, $(@($p.units).Count) units, pid $PID"
Write-JsonFile (Join-Path $out "env.$(Get-Date -Format yyyyMMddTHHmmss).json") (Get-HostEnv)

# adapter LUID -> kind, from the engine types of the GPU Engine counters (OFA / JPEG = NVIDIA, GSC / VideoProcessing =
# Intel GPU, Neural or Compute only = NPU); read once
function Get-LuidKinds {
    $m = @{}
    try {
        $names = (New-Object Diagnostics.PerformanceCounterCategory("GPU Engine")).GetInstanceNames()
        foreach ($n in $names) {
            if ($n -match "luid_(0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\d+_eng_\d+_engtype_(.*)$") {
                $l = $Matches[1]; $t = $Matches[2]
                if (-not $m.ContainsKey($l)) { $m[$l] = New-Object System.Collections.Generic.HashSet[string] }
                [void]$m[$l].Add($t)
            }
        }
    } catch { }
    $kinds = @{}
    foreach ($l in $m.Keys) {
        $t = @($m[$l])
        $k = if ($t -match "OFA|JPEG") { "nvidia" } elseif ($t -match "GSC|VideoProcessing") { "intel-gpu" } elseif (($t -join ",") -match "^(Neural|Compute)") { "npu?" } else { "?" }
        $kinds[$l] = [ordered]@{ kind = $k; engtypes = ($t | Sort-Object) -join "," }
    }
    return $kinds
}
$luids = Get-LuidKinds
Write-JsonFile (Join-Path $out "luids.json") $luids
Say $log "adapters: $(($luids.Keys | ForEach-Object { "$_=$($luids[$_].kind)" }) -join ', ')"

# What a first-pass cell gives way to: every other lane's lock on a GPU; on the NPU or the CPU only a timing lock
$script:CellOffGpu = $false
# first mode: any other lane's lock stops the run (killed, not suspended: a suspended process keeps its RAM, the shared
# PC's rule; 11-09 13:00 another lane's gate waited on 8.6 GB held by suspended Z3 processes); it runs again from its start
function Test-CellYield { return (Test-OtherBusy) }
function Stop-EngineTree([int]$EnginePid) {
    $ErrorActionPreference = "Continue"
    & "$env:SystemRoot\System32\taskkill.exe" /PID $EnginePid /T /F 2>&1 | Out-Null
}
function Read-SlotNow {
    if (-not (Test-Path $SlotNow)) { return $null }
    try {
        $t = ([IO.File]::ReadAllText($SlotNow)).Trim() -split '\s+'
        return [pscustomobject]@{ who = $t[0]; giveUpMin = $(if ($t.Count -gt 1) { [int]$t[1] } else { 15 }); at = (Get-Item $SlotNow).LastWriteTime }
    } catch { return $null }
}

# One engine process under a 0.5 s poll: its peak working set, its GPU memory per adapter every 2 s, the NVIDIA line
# every 5 s; killed with its tree after the unit's timeout. First mode: yields to another lane's lock (stopped, or
# suspended when $SuspendOnYield). Timed mode: a slot_now.txt request gives the run 8 min, then stops it (yielded).
function Invoke-Engine([string]$Exe, [string[]]$ArgList, [string]$Base, [int]$TimeoutMin, [bool]$YieldCheck = $false, [bool]$SuspendOnYield = $false) {
    $quoted = @($ArgList | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } })
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $Exe -ArgumentList ($quoted -join " ") -RedirectStandardOutput "$Base.stdout.txt" `
        -RedirectStandardError "$Base.stderr.txt" -PassThru -WindowStyle Hidden
    $null = $proc.Handle
    $peak = [int64]0; $gpu = @{}; $nvSamples = @(); $next2 = 2.0; $next5 = 5.0; $timedOut = $false
    $yielded = $false; $susp = 0.0; $suspN = 0; $slotStop = $null
    $engPid = $null; $engProc = $null; $engPath = $null
    try {
    while (-not $proc.HasExited) {
        # the venv's python.exe is a launcher that runs the interpreter as its child: watch the pid the runner prints
        if (-not $engPid) {
            try {
                $sfs = [IO.File]::Open("$Base.stdout.txt", [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
                $srd = New-Object IO.StreamReader($sfs); $stxt = $srd.ReadToEnd(); $srd.Close()
                if ($stxt -match 'ENGINE_PID (\d+)') {
                    $engPid = [int]$Matches[1]
                    $engProc = Get-Process -Id $engPid -ErrorAction SilentlyContinue
                    if ($engProc) { $null = $engProc.Handle; try { $engPath = $engProc.Path } catch { } }
                }
            } catch { }
        }
        $tgt = if ($engProc) { $engProc } else { $proc }
        try { $tgt.Refresh(); if ($tgt.PeakWorkingSet64 -gt $peak) { $peak = $tgt.PeakWorkingSet64 } } catch { }
        $t = $sw.Elapsed.TotalSeconds
        if ($t -ge $next2) {
            $gpu = Merge-GpuPeak $gpu (Get-ProcGpuMiB $(if ($engPid) { $engPid } else { $proc.Id }))
            $next2 += 2.0
            if ($Mode -eq "timed") {
                $sn = Read-SlotNow
                if ($sn -and -not $slotStop) { $slotStop = $sn.at.AddMinutes(8); Say $log "$Base slot request for $($sn.who): this run ends by $($slotStop.ToString('HH:mm:ss')) or is stopped" }
                if ($slotStop -and (Get-Date) -gt $slotStop) { $yielded = $true; Stop-EngineTree $proc.Id; break }
            }
        }
        if ($t -ge $next5) { $nvSamples += "$([int]$t)s $(Get-NvidiaLine)"; $next5 += 5.0 }
        if ($YieldCheck -and (Test-CellYield)) {
            if ($SuspendOnYield) {
                $hs = Suspend-Tree $proc.Id; $s0 = Get-Date
                Say $log "$Base suspended ($($hs.Count) processes) while another lane holds busy.txt ($((Read-Busy).Text))"
                do {
                    while (Test-CellYield) { Start-Sleep -Seconds 15 }
                    if ($script:CellRamWant -gt 0) { [void](Wait-FreeRam $script:CellRamWant $log "$Base resume") }
                } while (Test-CellYield)
                $rn = Resume-Tree $hs; $susp += ((Get-Date) - $s0).TotalSeconds; $suspN++
                Say $log "$Base resumed ($rn processes) after $([int]((Get-Date) - $s0).TotalSeconds) s"
                continue
            }
            $yielded = $true
            Stop-EngineTree $proc.Id
            break
        }
        if (($t - $susp) -ge $TimeoutMin * 60) {
            $timedOut = $true
            Stop-EngineTree $proc.Id
            break
        }
        Start-Sleep -Milliseconds 500
    }
    } finally {
        if (-not $proc.HasExited) { Stop-EngineTree $proc.Id }
    }
    $proc.WaitForExit()
    return [ordered]@{ exitCode = $(if ($timedOut -or $yielded) { $null } else { $proc.ExitCode }); timedOut = $timedOut; yielded = $yielded
        suspendedSeconds = [math]::Round($susp, 1); suspensions = $suspN; enginePid = $engPid; enginePath = $engPath
        elapsedSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 3); peakWorkingSetBytes = $peak; gpuMiBPeak = $gpu; nvidiaSamples = $nvSamples }
}

function Expand-Arg([string]$a, [string]$RunDir) {
    return $a.Replace("{HOME}", $HomeDir).Replace("{ROOT}", $Root).Replace("{FIX}", $Fix).Replace("{RUN}", $RunDir)
}

# A unit's "gpu_pref" (1 = power saving = the Intel GPU, 2 = high performance = the RTX): Windows' per-app GPU
# preference (HKCU\Software\Microsoft\DirectX\UserGpuPreferences, value name = the exe's full path, "GpuPreference=N;")
# written for the venv's launcher and for the interpreter it starts, BEFORE the engine process starts, and removed
# after it ends (the way an earlier LiteRT-LM job on this PC pointed its exe at a GPU; 11-10). LiteRT-LM 0.19.1's
# Python API has no device choice. Z3's runner (pc\run_asr_litertlm.py --gpu-pref) wrote the value under the junction's
# target (os.path.realpath) and every Intel GPU run of Z3 took the RTX; under the path the process runs by (uv's
# junction) it reached the Intel GPU, written before the start or inside the process (raw\z3b-probe). The run's meta
# records what was written and read back, and the "Selected adapter:" line of its log.
$GpuPrefSubKey = "Software\Microsoft\DirectX\UserGpuPreferences"
# Names "image" (default): the launcher and the interpreter as pyvenv.cfg's home names it, through uv's minor-version
# junction (...\cpython-3.12-...\python.exe: the image path Windows reports for the engine process, read 11-10);
# "real": the launcher and the junction's target (...\cpython-3.12.16-...\python.exe, the name Z3's runner wrote);
# "all": both.
function Get-VenvExes([string]$Venv, [string]$NameForm = "image") {
    $launcher = Join-Path $Root ".venv-$Venv\Scripts\python.exe"
    $found = @($launcher)
    $cfgPath = Join-Path $Root ".venv-$Venv\pyvenv.cfg"
    if (Test-Path -LiteralPath $cfgPath) {
        $homeLine = @(Get-Content -LiteralPath $cfgPath | Where-Object { $_ -match '^\s*home\s*=' })
        if ($homeLine.Count) {
            $pyDir = ($homeLine[0] -replace '^\s*home\s*=\s*', '').Trim()
            if ($NameForm -in "image", "all") { $found += (Join-Path $pyDir "python.exe") }
            if ($NameForm -in "real", "all") {
                $di = Get-Item -LiteralPath $pyDir -ErrorAction SilentlyContinue
                $tgt = if ($di -and $di.LinkType -and $di.Target) { @($di.Target)[0] } else { $pyDir }
                $found += (Join-Path $tgt "python.exe")
            }
        }
    }
    return @($found | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique)
}
function Set-GpuPrefBefore([int]$Value, [string[]]$Exes) {
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey($GpuPrefSubKey)
    try {
        foreach ($e in $Exes) { $key.SetValue($e, "GpuPreference=$Value;", [Microsoft.Win32.RegistryValueKind]::String) }
        $back = [ordered]@{}
        foreach ($e in $Exes) { $back[$e] = "$($key.GetValue($e))" }
    } finally { $key.Close() }
    return [ordered]@{ value = $Value; exes = @($Exes); readBack = $back; writtenAt = (Get-Date -Format o) }
}
function Clear-GpuPref([string[]]$Exes) {
    $gone = @()
    $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey($GpuPrefSubKey, $true)
    if (-not $key) { return $gone }
    try {
        foreach ($e in $Exes) { if ($null -ne $key.GetValue($e)) { $key.DeleteValue($e, $false); $gone += $e } }
    } finally { $key.Close() }
    return $gone
}
# a value a stopped chain left on this repo's exes would send every later GPU run of that interpreter to its GPU:
# removed at the chain's start (and logged)
$script:prefExesAll = @(@($p.units) | ForEach-Object { $_.venv } | Where-Object { $_ } | Select-Object -Unique | ForEach-Object { Get-VenvExes $_ "all" })
$leftPref = @(Clear-GpuPref $script:prefExesAll)
if ($leftPref.Count) { Say $log "removed a GpuPreference value left on: $($leftPref -join ', ')" }
function Get-SelectedAdapter([string]$Base) {
    $f = "$Base.stderr.txt"
    if (-not (Test-Path -LiteralPath $f)) { return $null }
    $m = @(Select-String -LiteralPath $f -Pattern 'Selected adapter: (.*)$' -ErrorAction SilentlyContinue)
    if (-not $m.Count) { return $null }
    return @($m | ForEach-Object { $_.Matches[0].Groups[1].Value.Trim() } | Select-Object -Unique)
}

# One run of a unit; returns the chain's meta
function Run-Once($u, [int]$k, [string]$base) {
    $yc = ($Mode -eq "first")
    $gate = $null
    if ($Mode -eq "timed") {
        $gate = Wait-RunQuiet $log -Engines:([bool]$u.engines_gate) -RamNeedGB $(if ($u.ram_need_gb) { [double]$u.ram_need_gb } else { 8 })
        if ($gate.quietGate -eq "ram-short") { throw "per-run gate: available RAM $($gate.availGB) GB under the $($gate.ramNeedGB) GB of $($u.cell) for 10 min in a row (holders in the log)" }
    }
    $before = [ordered]@{ load = (Measure-Load 1); nvidia = (Get-NvidiaLine); availGB = (Get-AvailGB) }
    $py = Join-Path $Root ".venv-$($u.venv)\Scripts\python.exe"
    $meta = [ordered]@{ cell = $u.cell; device = $u.device; role = $u.role; run = $k; mode = $Mode; sitting = $sit; start = (Get-Date -Format o); osBuild = (Get-OsBuild); desktopLocked = [bool](Get-Process -Name LogonUI -ErrorAction SilentlyContinue); power = (Get-PowerMode); gate = $gate; before = $before }
    $runDir = Split-Path -Parent $base
    $a = @("-u", (Join-Path $Root "pc\$($u.script)")) + @(@($u.args) | ForEach-Object { Expand-Arg "$_" $runDir }) + @("--out", "$base.json")
    $meta.command = "$py $($a -join ' ')"
    $prefRec = $null
    if ($u.gpu_pref) {
        $prefNames = if ($u.gpu_pref_names) { "$($u.gpu_pref_names)" } else { "image" }
        $prefRec = Set-GpuPrefBefore ([int]$u.gpu_pref) (Get-VenvExes $u.venv $prefNames)
        $prefRec["names"] = $prefNames
        $meta.gpuPreference = $prefRec
        Say $log "$($u.cell) run $k GpuPreference=$($u.gpu_pref) written before the start on: $(@($prefRec.exes) -join ', ')"
    }
    Say $log "$($u.cell) run $k start"
    try {
        $meta.engine = Invoke-Engine $py $a $base ([int]$u.timeout_min) $yc $false
    } finally {
        if ($prefRec) { $meta.gpuPreferenceRemoved = @(Clear-GpuPref @($prefRec.exes)); $meta.gpuPreferenceRemovedAt = (Get-Date -Format o) }
    }
    $meta.selectedAdapter = Get-SelectedAdapter $base
    if ($meta.selectedAdapter) { Say $log "$($u.cell) run $k Selected adapter: $(@($meta.selectedAdapter) -join ' | ')" }
    $meta.after = [ordered]@{ load = (Measure-Load 1); nvidia = (Get-NvidiaLine) }
    $meta.end = Get-Date -Format o
    return $meta
}

function Run-Unit($u, [int]$k) {
    $dir = Join-Path $out ($u.cell -replace '/', '\')   # <model>/<cell> -> <model>\<cell>
    New-Item -ItemType Directory -Force $dir | Out-Null
    $base = Join-Path $dir "run$k"
    if (Test-Path "$base.meta.json") { Say $log "$($u.cell) run $k on disk, skipped"; return }
    $script:CellOffGpu = [bool]($u.device -in "cpu", "npu")
    # first mode: every unit waits for its RAM (need + 2 GB) before it starts (11-09, the PC's order: untimed work side by
    # side stays inside the PC's free RAM; the per-run gate does it in timed mode)
    $script:CellRamWant = if ($Mode -eq "first") { [double]$u.ram_need_gb + 2 } else { 0 }
    for ($try = 1; $try -le 20; $try++) {
        if ($Mode -eq "first") {
            do {
                Wait-OthersGone $log "$($u.cell) run $k" 60
                if ($script:CellRamWant -gt 0) { [void](Wait-FreeRam $script:CellRamWant $log "$($u.cell) run $k") }
            } while (Test-CellYield)
        }
        $meta = Run-Once $u $k $base
        if (-not $meta.engine.yielded) { break }
        Say $log "$($u.cell) run $k stopped (another lane's lock or a slot); its files removed, run again after it"
        Remove-Item "$base.json", "$base.stdout.txt", "$base.stderr.txt" -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $dir "wav") -Recurse -Force -ErrorAction SilentlyContinue
        if ($Mode -eq "timed") { Invoke-Slot "unit $($u.cell) run $k" }
    }
    $meta.tries = $try
    Write-JsonFile "$base.meta.json" $meta
    $res = if (Test-Path "$base.json") { (Get-Content "$base.json" -Raw -Encoding UTF8 | ConvertFrom-Json).status } else { "no runner JSON" }
    Say $log "$($u.cell) run $k end: exit $($meta.engine.exitCode), $res, $($meta.engine.elapsedSeconds) s"
    Note-RunSec $u.cell $meta.engine.elapsedSeconds
    $script:runsSinceTake++
}

# ---- duty and slots (timed mode) ---------------------------------------------------------------------------------
$script:held = if ($Mode -eq "timed" -and (Test-OursBusy)) { (Get-Item $Busy).CreationTime } else { Get-Date }
$script:runsSinceTake = if (((Get-Date) - $script:held).TotalMinutes -ge 1) { 1 } else { 0 }
$script:runSec = @{}
function Note-RunSec([string]$Cell, $Sec) {
    try {
        $s = [double]$Sec
        if ($s -gt 0 -and (-not $script:runSec.ContainsKey($Cell) -or $s -gt $script:runSec[$Cell])) { $script:runSec[$Cell] = $s }
    } catch { }
}
# expected seconds of one run: this chain's own runs of the cell, else the plan's est_sec (the first pass), + 60 s
function Get-RunEstSec($u) {
    if ($script:runSec.ContainsKey($u.cell)) { return [double]$script:runSec[$u.cell] + 60 }
    if ($u.est_sec) { return [double]$u.est_sec + 60 }
    return 0
}
# end_by and a take-back (11-09): the free minute, the settle, the gate and the tail row must still end before end_by.
# Otherwise the sitting ends where it is (no take-back, no tail row): another lane's long block in a window, or a gate
# that does not open, never carries the take-back into a fixed slot after end_by.
function Test-NoTimeForTail {
    if (-not $EndBy) { return $false }
    $need = $RetakeFreeSec + $SettleSeconds + 60 + 120
    $tail = @($p.units | Where-Object { $_.role -eq "control-tail" } | Select-Object -First 1)
    if ($tail.Count) { $need += Get-RunEstSec $tail[0] }
    return ((Get-Date).AddSeconds($need) -gt $EndBy)
}
$script:endByStop = $false
function Stop-AtEndBy([string]$What) {
    $script:endByStop = $true
    Close-Window
    Release-Busy $log
    Say $log "end_by $($EndBy.ToString('HH:mm')): no take-back after $What (the free minute, settle, gate and the tail row would not end before it): the sitting ends here without its tail row"
}
function Take-Back([string]$What) {
    while ($true) {
        while ((Test-OtherBusy) -and -not (Test-NoTimeForTail)) { Start-Sleep -Seconds 5 }
        if (Test-NoTimeForTail) { Stop-AtEndBy $What; return }
        $t = Take-Busy "timing: sitting $sit (taken back after $What)" $log $RetakeFreeSec
        Close-Window
        if ($t -eq "TIMEOUT") { throw "could not take the lock back" }
        Say $log "settle $SettleSeconds s inside the lock after the take-back"
        Start-Sleep -Seconds $SettleSeconds
        if ((Pass-LaneGate $log) -eq "PASS") { break }
        if (Test-NoTimeForTail) { Stop-AtEndBy "a gate that did not open"; return }
        Release-Busy $log; Open-Window 600 $log; Wait-Window 600 $RetakeFreeSec; Close-Window
    }
    $script:held = Get-Date
    $script:runsSinceTake = 0
}
# A slot: release without window.txt, the other lane's lock comes (until $GiveUp) and goes (free $RetakeFreeSec s in a
# row), then take back, settle, gate
function Hold-Slot([string]$Who, [datetime]$GiveUp) {
    Close-Window
    Release-Busy $log
    Say $log "slot for $Who`: lock released, no window.txt; waiting for its lock until $($GiveUp.ToString('HH:mm:ss'))"
    $seen = ""; $freeSince = $null; $why = ""
    while ($true) {
        $b = Read-Busy
        if ($b -and -not $b.Text.StartsWith("$Label ")) {
            if (-not $seen) { $seen = $b.Text; Say $log "slot: lock taken by: $seen" }
            $freeSince = $null
        } elseif ($seen -and -not $b) {
            if (-not $freeSince) { $freeSince = Get-Date }
            elseif (((Get-Date) - $freeSince).TotalSeconds -ge $RetakeFreeSec) { $why = "given back (free $RetakeFreeSec s)"; break }
        } elseif (-not $seen -and (Get-Date) -gt $GiveUp) { $why = "no other lock came by $($GiveUp.ToString('HH:mm:ss'))"; break }
        Start-Sleep -Seconds 5
    }
    Say $log "slot for $Who`: $why; taking the lock back"
    Take-Back "the slot for $Who"
    if ($script:endByStop) { return }
    Say $log "slot for $Who`: lock taken back, gate PASS"
}
$script:slotsDone = @{}
function Invoke-Slot([string]$What, $u = $null) {
    $sn = Read-SlotNow
    if ($sn) {
        Say $log "slot request for $($sn.who) (runs\slot_now.txt, $($sn.at.ToString('HH:mm:ss'))) before $What"
        Remove-Item $SlotNow -Force -ErrorAction SilentlyContinue
        Hold-Slot $sn.who ((Get-Date).AddMinutes($sn.giveUpMin))
        return
    }
    $est = if ($u) { Get-RunEstSec $u } else { 0 }
    foreach ($s in @($p.slots)) {
        if (-not $s -or $script:slotsDone.ContainsKey($s.who + $s.release_at)) { continue }
        $rel = [datetime]::Parse($s.release_at); $gu = [datetime]::Parse($s.give_up_at)
        if ((Get-Date) -gt $gu) { $script:slotsDone[$s.who + $s.release_at] = $true; continue }
        $limit = if ($est -gt 0) { $est } elseif ($u -and $u.timeout_min) { 60 * [double]$u.timeout_min } else { 0 }
        if ((Get-Date).AddSeconds($limit + 60) -ge $rel) {
            $script:slotsDone[$s.who + $s.release_at] = $true
            if ((Get-Date) -lt $rel) {
                Say $log "slot for $($s.who) at $($rel.ToString('HH:mm:ss')): $What would not end before it; holding the lock idle"
                Touch-Busy "timing: sitting $sit holds the lock idle for the $($s.who) slot until $($rel.ToString('HH:mm'))"
                while ((Get-Date) -lt $rel) { Start-Sleep -Milliseconds 500 }
            }
            Hold-Slot $s.who $gu
        }
    }
}
function Invoke-Duty([string]$What, $u) {
    Invoke-Slot $What $u
    if ($script:endByStop) { return }
    $est = Get-RunEstSec $u
    $heldMin = ((Get-Date) - $script:held).TotalMinutes
    $others = @(Get-OthersWaiting)
    $why = ""
    if ($others.Count -gt 0) {
        if ($heldMin -ge 30) { $why = "held $([int]$heldMin) min" }
        elseif ($est -gt 0 -and $script:runsSinceTake -gt 0 -and ($heldMin + $est / 60) -gt 30) {
            $why = "held $([int]$heldMin) min and the next run would end past 30 min (about $([int][math]::Ceiling($est / 60)) min)"
        }
    }
    if ($why -and $u.role -eq "control-tail") {
        # The tail row closes the sitting (a few minutes, then the lock goes back for good): no window before it. It then
        # runs right after the last cell, as a control should, and no take-back after another lane's block can push it
        # past end_by into a fixed slot (11-09: two other lanes from 21:07).
        Say $log "duty: $why, others waiting ($($others -join ', ')): no window before the control tail, the lock goes back after it"
        $why = ""
    }
    if ($why -and $p.windows -eq $false) {
        # One block (the plan's windows false, tools/make_plan.py --no-windows): the PC's order gives this sitting the lock
        # to end_by in one piece (the night of 11-09/10), so another lane's note opens no window.
        Say $log "duty: $why, others waiting ($($others -join ', ')): no window (the plan: one block to end_by)"
        $why = ""
    }
    if ($why) {
        Say $log "duty: $why, others waiting ($($others -join ', ')): release for 16.5 min"
        Release-Busy $log
        Open-Window 990 $log
        Wait-Window 990 $RetakeFreeSec
        Close-Window
        Take-Back "a duty gap"
        if ($script:endByStop) { return }
    }
    $until = ""
    if ($est -gt 0) { $until = ", run until about $((Get-Date).AddSeconds($est).ToString('HH:mm'))" }
    elseif ($u.timeout_min) { $until = ", run until $((Get-Date).AddMinutes([double]$u.timeout_min + 1).ToString('HH:mm')) at the latest (its time limit)" }
    $block = if ($p.windows -eq $false -and $EndBy) { " (one block to $($EndBy.ToString('HH:mm')))" } else { "" }
    Touch-Busy "timing: sitting $sit$block $What$until"
}

# A cell left for a later sitting (end_by): its runs, the control tail, a duty gap when another lane waits (none in a
# one-block plan), and 4 min
# would not end before end_by. Once one cell is left, every cell after it is too (the plan's order is kept).
function Test-PastEndBy($u, [int]$Runs) {
    if (-not $EndBy -or $u.role) { return $false }
    if ($script:leftCells.Count) { return $true }
    $est = Get-RunEstSec $u
    if ($est -le 0) { $est = 60 * [double]$u.timeout_min }
    $need = $est * $Runs + $BetweenRunsSeconds * ($Runs - 1) + 240
    $tail = @($p.units | Where-Object { $_.role -eq "control-tail" } | Select-Object -First 1)
    if ($tail.Count) { $need += Get-RunEstSec $tail[0] }
    if ($p.windows -ne $false -and @(Get-OthersWaiting).Count) { $need += 990 + $SettleSeconds }   # no gap in one block
    return ((Get-Date).AddSeconds($need) -gt $EndBy)
}
function Leave-Cell($u) {
    $script:leftCells += $u.cell
    Say $log "end_by $($EndBy.ToString('HH:mm')): $($u.cell) not started, left for a later sitting"
}

# ---- control rows ------------------------------------------------------------------------------------------------
function Get-Rtf([string]$JsonPath) {
    try { $j = Get-Content $JsonPath -Raw -Encoding UTF8 | ConvertFrom-Json; if ($j.status -eq "OK") { return [double]$j.summary.rtf } } catch { }
    return $null
}
$script:control = [ordered]@{ known_rtf = $p.control_known_rtf; known_from = $(if ($p.control_known_rtf) { "plan" } else { $null }); head = @(); tail = @(); verdict = $null }
function Check-Control($u) {
    $dir = Join-Path $out ($u.cell -replace '/', '\')
    $vals = @(1..([int]$(if ($u.runs) { $u.runs } else { 1 })) | ForEach-Object { Get-Rtf (Join-Path $dir "run$_.json") })
    if ($u.role -eq "control-head") {
        $script:control.head = $vals
        if ($vals -contains $null) { $script:control.verdict = "head failed"; throw "control head: a run failed" }
        if (-not $script:control.known_rtf) {
            $s = @($vals | Sort-Object)
            $med = $s[[int][math]::Floor(($s.Count - 1) / 2)]
            $spread = ($s[-1] - $s[0]) / $med
            if ($s.Count -lt 3 -or $spread -gt 0.05) { $script:control.verdict = "head spread $([math]::Round(100 * $spread, 1)) % over $($s.Count) runs"; throw "control head: no known value ($($script:control.verdict))" }
            $script:control.known_rtf = $med; $script:control.known_from = "this sitting's head ($($s.Count) runs, spread $([math]::Round(100 * $spread, 1)) %)"
        }
        $off = [math]::Abs($vals[0] - $script:control.known_rtf) / $script:control.known_rtf
        Say $log "control head: rtf $($vals -join ', ') vs known $($script:control.known_rtf) ($($script:control.known_from)): off $([math]::Round(100 * $off, 1)) %"
        if ($off -gt 0.05) { $script:control.verdict = "head off $([math]::Round(100 * $off, 1)) %"; throw "control head off the known value by more than 5 %: no other rows" }
    } elseif ($u.role -eq "control-tail") {
        $script:control.tail = $vals
        if ($vals -contains $null) { $script:control.verdict = "tail failed" }
        else {
            $off = [math]::Abs($vals[0] - $script:control.known_rtf) / $script:control.known_rtf
            $script:control.verdict = if ($off -le 0.05) { "PASS (tail off $([math]::Round(100 * $off, 1)) %)" } else { "TAIL OFF $([math]::Round(100 * $off, 1)) %: the rows of this sitting are not used" }
        }
        Say $log "control tail: rtf $($vals -join ', '): $($script:control.verdict)"
    }
    Write-JsonFile (Join-Path $out "control.json") $script:control
}

$status = "OK"
try {
    if ($Mode -eq "timed" -and -not (Test-OursBusy)) { throw "mode timed needs the Z3 lock (tools/sitting.sh takes it and gates first)" }
    if ($Mode -eq "timed") {
        Touch-Busy "timing: sitting $sit settles $SettleSeconds s before its control rows"
        Say $log "settle $SettleSeconds s inside the lock before the control rows"
        Start-Sleep -Seconds $SettleSeconds
    }
    if ($Mode -eq "first") { Write-Utf8 $active "$Label first pass of $sit ($(@($p.units).Count) units, no timing, yields to every lock) since $(Get-Date -Format o)" }
    foreach ($u in @($p.units)) {
        if ($Mode -eq "first") { Write-Utf8 $active "$Label first pass of $sit ($(@($p.units).Count) units, no timing, yields to every lock), now $($u.cell), since $chainStart`r`nwritten $(Get-Date -Format HH:mm)" }
        $runs = if ($Mode -eq "first") { 1 } elseif ($u.runs) { [int]$u.runs } else { 1 }
        if ($script:endByStop) { if (-not $u.role) { Leave-Cell $u }; continue }   # no take-back near end_by: the rest waits
        if (Test-PastEndBy $u $runs) { Leave-Cell $u; continue }
        $left = $false
        for ($k = 1; $k -le $runs; $k++) {
            if ($Mode -eq "timed") { Invoke-Duty "unit $($u.cell) run $k" $u }
            if ($script:endByStop) { if (-not $u.role) { Leave-Cell $u }; $left = $true; break }
            if ($k -eq 1 -and (Test-PastEndBy $u $runs)) { Leave-Cell $u; $left = $true; break }   # a duty gap ate the time
            Run-Unit $u $k
            if ($Mode -eq "timed" -and $k -lt $runs) { Start-Sleep -Seconds $BetweenRunsSeconds }
        }
        if ($left) { continue }
        if ($Mode -eq "timed" -and $u.role -in "control-head", "control-tail") { Check-Control $u }
    }
} catch {
    $status = "FAIL $($_.Exception.Message)"
    Say $log "chain error: $($_ | Out-String)"
} finally {
    if ($Mode -eq "first" -and (Test-Path $active)) { Remove-Item $active -Force }
    if ($script:leftCells.Count) { $script:control["left_cells"] = @($script:leftCells); $script:control["end_by"] = $p.end_by }
    if ($script:endByStop -and -not $script:control.verdict) { $script:control.verdict = "no tail row: no take-back before end_by" }
    if ($Mode -eq "timed") { Close-Window; Write-JsonFile (Join-Path $out "control.json") $script:control }
    Write-Utf8 $done "$status $(Get-Date -Format o)"
    Say $log "chain done: $status"
}
if ($Mode -eq "timed") {
    Touch-Busy "timing: sitting $sit done, the records go to the Mac, until $((Get-Date).AddMinutes($HoldAtEndMin).ToString('HH:mm')) at the latest"
    $until = (Get-Date).AddMinutes($HoldAtEndMin)
    while ((Get-Date) -lt $until -and (Test-OursBusy)) { Start-Sleep -Seconds 5 }
    if (Test-OursBusy) { Say $log "the Mac did not release the lock in $HoldAtEndMin min: releasing"; Release-Busy $log }
}
