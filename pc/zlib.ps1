# Shared helpers of the Z3 scripts on the PC (dot-source: . "$PSScriptRoot\zlib.ps1"). Windows PowerShell 5.1.
# The shared-PC rules these follow (agreed by the projects that share this PC): busy.txt is the timing lock
# (CreateNew, one owner), a lane's presence note for untimed work is active\<lane>.txt, a lane that times gates on a
# quiet host first, releases the lock for 16.5 min after 30 min while other lanes wait (pcup\window.txt says until
# when), and reads no WMI in loops where it can.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot          # C:\src\audiozoo (pc\ sits in it)
$HomeDir = [Environment]::GetFolderPath("UserProfile")   # from the token, not the environment block
$Pcup = Join-Path $HomeDir "pcup"
if ($env:Z3_PCUP) { $Pcup = $env:Z3_PCUP }        # another folder for a dry run of the lock logic
$Busy = Join-Path $Pcup "busy.txt"
# a PC without the shared-PC folder (README walk elsewhere): the lock file and the presence notes start empty
foreach ($d in @($Pcup, (Join-Path $Pcup "active"))) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null } }
$Label = "Z3"
$BuildTools = @("cl", "link", "MSBuild", "cmake", "ninja", "nvcc", "cicc", "ptxas")
$Smi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
Add-Type -AssemblyName Microsoft.VisualBasic

function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding($false)))
}
function Append-Utf8([string]$Path, [string]$Text) {
    foreach ($i in 1..50) {
        try { [IO.File]::AppendAllText($Path, $Text, (New-Object Text.UTF8Encoding($false))); return }
        catch { Start-Sleep -Milliseconds (20 + (Get-Random -Maximum 80)) }
    }
}
function Write-JsonFile([string]$Path, $Value) { Write-Utf8 $Path ($Value | ConvertTo-Json -Depth 10) }
function Say([string]$Log, [string]$Text) { Append-Utf8 $Log "[$(Get-Date -Format o)] $Text`r`n" }

# ---- the timing lock -------------------------------------------------------------------------------------------
function Read-Busy {
    if (-not (Test-Path $Busy)) { return $null }
    try {
        $t = [IO.File]::ReadAllText($Busy)
        return [pscustomobject]@{ Text = $t.Trim(); AgeMin = ((Get-Date) - (Get-Item $Busy).LastWriteTime).TotalMinutes }
    } catch { return [pscustomobject]@{ Text = "?"; AgeMin = 0 } }
}
function Test-OursBusy { $b = Read-Busy; return [bool]($b -and $b.Text.StartsWith("$Label ")) }

# One try: TAKEN (created), OURS (rewritten), STALE (another lane's marker 120 min or older, taken over), or the other
# lane's text (busy).
# The marker's text (the shared PC's form, queue line 5): the lane first, then "timing: <what> until HH:MM" or
# "<what> (not a timing) until HH:MM", then the time it was written
function Try-TakeBusy([string]$What) {
    $line = "$Label $What $(Get-Date -Format o)"
    $b = Read-Busy
    if ($b) {
        if ($b.Text.StartsWith("$Label ")) { Write-Utf8 $Busy $line; return "OURS" }
        if ($b.AgeMin -ge 120) { Write-Utf8 $Busy $line; return "STALE" }
        return "BUSY $($b.Text)"
    }
    try {
        $fs = [IO.File]::Open($Busy, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        $bytes = [Text.Encoding]::UTF8.GetBytes($line)
        $fs.Write($bytes, 0, $bytes.Length); $fs.Close()
        return "TAKEN"
    } catch { return "BUSY (lost the race)" }
}
# Wait (5 s reads, the lock's gaps between other lanes' units can be a second) until the lock is ours. Another lane's
# marker must stay absent for $SettleSec before we take it: a lane that runs units back to back drops it for ~1 s, and
# a lane that gave it back reads its own files in that minute (60 s, the shared PC's form of 2026-11-09).
function Take-Busy([string]$What, [string]$Log, [int]$SettleSec = 60, [int]$MaxHours = 12) {
    $deadline = (Get-Date).AddHours($MaxHours)
    $last = ""
    while ((Get-Date) -lt $deadline) {
        $b = Read-Busy
        if ($b -and -not $b.Text.StartsWith("$Label ") -and $b.AgeMin -lt 120) {
            if ($b.Text -ne $last) { Say $Log "waiting for the lock: $($b.Text)"; $last = $b.Text }
            Start-Sleep -Seconds 5
            continue
        }
        if (-not $b -and $last) {
            # it was another lane's a moment ago: it must stay free for $SettleSec
            $free = $true
            for ($i = 0; $i -lt $SettleSec; $i += 5) { Start-Sleep -Seconds 5; $c = Read-Busy; if ($c -and -not $c.Text.StartsWith("$Label ")) { $free = $false; break } }
            if (-not $free) { continue }
        }
        $r = Try-TakeBusy $What
        if ($r -in "TAKEN", "OURS", "STALE") { Say $Log "lock: $r ($What)"; return $r }
        Start-Sleep -Seconds 5
    }
    return "TIMEOUT"
}
# The lock for a block that is not a timing (a step over 8 GB): taken only after busy.txt has been free $FreeSec in a
# row, whatever this call saw before (11-09 12:57: a step that yield_run had held suspended took it 9 s after another
# lane gave it back, ahead of the next lane's timing; the next lane takes within 45 s of a release)
function Take-BusyAfterFree([string]$What, [string]$Log, [int]$FreeSec = 60, [int]$MaxHours = 12) {
    $deadline = (Get-Date).AddHours($MaxHours)
    $freeSince = $null; $last = ""
    while ((Get-Date) -lt $deadline) {
        $b = Read-Busy
        if ($b -and -not $b.Text.StartsWith("$Label ") -and $b.AgeMin -lt 120) {
            if ($b.Text -ne $last) { Say $Log "waiting for the lock: $($b.Text)"; $last = $b.Text }
            $freeSince = $null
        } elseif ($b -and $b.Text.StartsWith("$Label ")) {
            return "OURS"
        } else {
            if (-not $freeSince) { $freeSince = Get-Date }
            elseif (((Get-Date) - $freeSince).TotalSeconds -ge $FreeSec) {
                $r = Try-TakeBusy $What
                if ($r -in "TAKEN", "OURS", "STALE") { Say $Log "lock: $r ($What) after $FreeSec s free"; return $r }
                $freeSince = $null
            }
        }
        Start-Sleep -Seconds 5
    }
    return "TIMEOUT"
}
# Untimed work waits while another lane holds the lock, and starts again only after busy.txt has been free $FreeSec in
# a row (the next lane in the order takes within 45 s of a release)
function Wait-OthersGone([string]$Log, [string]$What, [int]$FreeSec = 60) {
    if (-not (Test-OtherBusy)) { return }
    Say $Log "$What`: another lane holds the lock ($((Read-Busy).Text)): waiting"
    $freeSince = $null
    while ($true) {
        if (Test-OtherBusy) { $freeSince = $null }
        elseif (-not $freeSince) { $freeSince = Get-Date }
        elseif (((Get-Date) - $freeSince).TotalSeconds -ge $FreeSec) { break }
        Start-Sleep -Seconds 5
    }
    Say $Log "$What`: the lock has been free $FreeSec s: going on"
}
# Another lane holds the lock now (its marker is younger than 120 min)
function Test-OtherBusy { $b = Read-Busy; return [bool]($b -and -not $b.Text.StartsWith("$Label ") -and $b.AgeMin -lt 120) }
# A lock its lane marks "not a timing" (a short check on the RTX): first-pass work that stays off the GPU goes on under it
function Test-OtherTiming { $b = Read-Busy; return [bool]($b -and -not $b.Text.StartsWith("$Label ") -and $b.AgeMin -lt 120 -and $b.Text -notmatch "not a timing") }
function Touch-Busy([string]$What) { if (Test-OursBusy) { Write-Utf8 $Busy "$Label $What $(Get-Date -Format o)" } }
function Release-Busy([string]$Log) {
    if (Test-OursBusy) { Remove-Item $Busy -Force; if ($Log) { Say $Log "lock released" } }
}
# The window of a gap: while the lock is off for a duty gap (or a gate retry), pcup\window.txt says "Z3 open until
# <ISO time>"; a pausable lane starts a step over 8 GB only when it fits in the window's rest (2026-11-08:
# a 16.5 GB export started 3.5 min before a gap's end was suspended at the retake and its RAM shut that lane's gate).
$Window = Join-Path $Pcup "window.txt"
function Open-Window([int]$Seconds, [string]$Log) {
    $until = (Get-Date).AddSeconds($Seconds).ToString("o")
    Write-Utf8 $Window "$Label open until $until"
    if ($Log) { Say $Log "window.txt: $Label open until $until" }
}
function Close-Window {
    try { if ((Test-Path $Window) -and ([IO.File]::ReadAllText($Window)).StartsWith("$Label ")) { Remove-Item $Window -Force } } catch { }
}
# Sit out a window with the lock read every 5 s: a lane that gives the lock back shortly before the window ends still
# gets its $FreeSec (60 s) before Z3 takes it again (Take-Busy settles only after a marker it saw itself).
function Wait-Window([int]$Seconds, [int]$FreeSec = 60) {
    $end = (Get-Date).AddSeconds($Seconds)
    $seenAt = $null
    while ((Get-Date) -lt $end) { if (Test-OtherBusy) { $seenAt = Get-Date }; Start-Sleep -Seconds 5 }
    # $FreeSec counted from the first read that found it gone (5 s after the last read that saw it)
    if ($seenAt -and -not (Test-OtherBusy)) {
        $rest = $FreeSec + 5 - ((Get-Date) - $seenAt).TotalSeconds
        if ($rest -gt 0) { Start-Sleep -Seconds ([int][math]::Ceiling($rest)) }
    }
}
# Other lanes waiting on this PC: yield\*.txt (a job under the yield wrapper; counts for as long as it exists) or an
# active\*.txt note other than ours written in the last 120 min (the queue's rule: a waiting lane's PC-side watcher
# rewrites its note every 50-60 min, a note not rewritten is gone after 2 hours; a lane that forgot its note at its
# return would otherwise get a window every 30 min for nobody).
function Get-OthersWaiting {
    $noteSince = (Get-Date).AddMinutes(-120)
    @(@(Get-ChildItem (Join-Path $Pcup "yield\*.txt") -ErrorAction SilentlyContinue) +
      @(Get-ChildItem (Join-Path $Pcup "active\*.txt") -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -ge $noteSince }) |
        Where-Object { $_ -and -not $_.BaseName.StartsWith($Label) } | ForEach-Object { "$($_.Directory.Name)\$($_.Name)" })
}

# ---- load readings without WMI -----------------------------------------------------------------------------------
# CPU busy % over a window, from the processes' CPU seconds (all logical CPUs = 100 %), the build tools' CPU ms in it
# (a process that started inside the window counts from zero), and the five busiest processes.
function Measure-Load([double]$Seconds = 1.0) {
    $a = @{}
    foreach ($p in Get-Process) { try { $a[$p.Id] = $p.TotalProcessorTime.TotalMilliseconds } catch { } }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Start-Sleep -Milliseconds ([int]($Seconds * 1000))
    $el = $sw.Elapsed.TotalMilliseconds
    $rows = @()
    foreach ($p in Get-Process) {
        if ($p.Id -eq 0) { continue }
        try { $now = $p.TotalProcessorTime.TotalMilliseconds } catch { continue }
        $was = if ($a.ContainsKey($p.Id)) { $a[$p.Id] } else { 0 }
        $d = $now - $was
        if ($d -gt 0) { $rows += [pscustomobject]@{ n = $p.ProcessName; id = $p.Id; ms = $d } }
    }
    $total = ($rows | Measure-Object ms -Sum).Sum
    if (-not $total) { $total = 0 }
    $build = ($rows | Where-Object { $BuildTools -contains $_.n } | Measure-Object ms -Sum).Sum
    if (-not $build) { $build = 0 }
    $cpus = [Environment]::ProcessorCount
    $top = @($rows | Sort-Object ms -Descending | Select-Object -First 5 | ForEach-Object { "$($_.n) $($_.id) $([math]::Round($_.ms / $el, 2)) core" })
    return [pscustomobject]@{ cpu = [math]::Round(100.0 * $total / ($el * $cpus), 1); build_ms = [math]::Round($build); top = $top }
}
function Get-NvidiaMiB {
    $ErrorActionPreference = "Continue"   # a native command's stderr under Stop is a terminating error in 5.1
    if (-not (Test-Path $Smi)) { return 0 }
    try { return [int]((& $Smi "--query-gpu=memory.used" "--format=csv,noheader,nounits" 2>$null) | Select-Object -First 1) } catch { return -1 }
}
function Get-NvidiaLine {
    $ErrorActionPreference = "Continue"
    if (-not (Test-Path $Smi)) { return "" }
    try { return ((& $Smi "--query-gpu=pstate,clocks.gr,clocks.mem,power.draw,utilization.gpu,memory.used,temperature.gpu" "--format=csv,noheader" 2>$null) -join " ") } catch { return "" }
}
# OS build as CurrentBuild.UBR (a Windows update moved this PC's numbers: john-rocky/gemma-4-e4b-on-foundry-local, "After the Windows update"; a
# published number carries its build)
function Get-OsBuild {
    try { $cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'; return "$($cv.CurrentBuild).$($cv.UBR)" } catch { return "" }
}
# The power mode a run ran under (every timing records it): MSI Center's User Scenario Mode (on this PC 2 =
# Balanced, 1 = Extreme Performance, as the lanes that switched it on 11-09 read them) and the Windows power overlay on
# AC and on battery (00000000-0000-0000-0000-000000000000 = the default; MSI's Extreme set ded574b5-... = best
# performance), and whether the PC runs on AC (Win32_Battery status 2).
function Get-PowerMode {
    $r = [ordered]@{ msiUserScenarioMode = $null; overlayAc = $null; overlayDc = $null; battery = $null }
    try { $r.msiUserScenarioMode = (Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\MSI\MSI Center\Component\Base Module\User Scenario' -ErrorAction Stop).Mode } catch { }
    try {
        $o = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' -ErrorAction Stop
        $r.overlayAc = "$($o.ActiveOverlayAcPowerScheme)"; $r.overlayDc = "$($o.ActiveOverlayDcPowerScheme)"
    } catch { }
    try { $b = @(Get-CimInstance Win32_Battery -ErrorAction Stop)[0]; if ($b) { $r.battery = "status $($b.BatteryStatus), $($b.EstimatedChargeRemaining) %" } } catch { }
    return $r
}
# What the rows ran on, read once per chain start: OS build, GPU and NPU drivers, NVIDIA driver, power scheme, the
# power mode (Get-PowerMode), MSI's AI Engine switch, the desktop's lock screen
function Get-HostEnv {
    $ErrorActionPreference = "Continue"
    $e = [ordered]@{ read = (Get-Date -Format o); osBuild = (Get-OsBuild) }
    try { $e.machine = (Get-CimInstance Win32_ComputerSystem).Model; $e.cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name } catch { }
    try { $e.osDisplayVersion = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').DisplayVersion } catch { }
    try { $os = Get-CimInstance Win32_OperatingSystem; $e.osCaption = $os.Caption; $e.ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1) } catch { }
    try { $e.gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { [ordered]@{ name = $_.Name; driverVersion = $_.DriverVersion; driverDate = "$($_.DriverDate)" } }) } catch { }
    try { $e.npu = @(Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue | ForEach-Object { [ordered]@{ name = $_.FriendlyName; status = "$($_.Status)"; driverVersion = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_DriverVersion -ErrorAction SilentlyContinue).Data } }) } catch { }
    if (Test-Path $Smi) { try { $e.nvidiaDriver = ((& $Smi "--query-gpu=driver_version" "--format=csv,noheader" 2>$null) | Select-Object -First 1) } catch { } }
    try { $e.powerScheme = ((& powercfg /getactivescheme) -join " ").Trim() } catch { }
    $e.power = Get-PowerMode
    try { $e.msiAIMode = (Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\MSI\MSI Center\Component\AI Engine' -ErrorAction Stop).AIMode } catch { $e.msiAIMode = $null }
    # the desktop's lock screen (LogonUI runs while it is up)
    $e.desktopLocked = [bool](Get-Process -Name LogonUI -ErrorAction SilentlyContinue)
    return $e
}
# Suspend / resume a process tree (the shared PC's yield_run.ps1 does the same): a first-pass NPU engine is CPU-bound
# while it compiles and holds no GPU memory, so it waits out another lane's lock suspended instead of being killed
Add-Type -Namespace Z3 -Name Native -MemberDefinition @"
[DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr handle);
[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr handle);
[DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool CloseHandle(IntPtr handle);
"@
function Get-TreePids([int]$Root) {
    $all = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId)
    $out = New-Object System.Collections.Generic.List[int]; $q = New-Object System.Collections.Generic.Queue[int]; $q.Enqueue($Root)
    while ($q.Count) { $x = $q.Dequeue(); $out.Add($x); foreach ($p in $all) { if ([int]$p.ParentProcessId -eq $x -and -not $out.Contains([int]$p.ProcessId)) { $q.Enqueue([int]$p.ProcessId) } } }
    return $out
}
function Suspend-Tree([int]$Root) {
    $h = @{}
    for ($pass = 0; $pass -lt 4; $pass++) {
        $new = 0
        foreach ($p in (Get-TreePids $Root)) {
            if ($h.ContainsKey($p)) { continue }
            $x = [Z3.Native]::OpenProcess(0x0800, $false, $p)
            if ($x -eq [IntPtr]::Zero) { continue }
            if ([Z3.Native]::NtSuspendProcess($x) -eq 0) { $h[$p] = $x; $new++ } else { [void][Z3.Native]::CloseHandle($x) }
        }
        if (-not $new) { break }
    }
    return $h
}
function Resume-Tree($H) {
    $n = 0
    foreach ($p in @($H.Keys)) { if ([Z3.Native]::NtResumeProcess($H[$p]) -eq 0) { $n++ }; [void][Z3.Native]::CloseHandle($H[$p]) }
    return $n
}
function Get-AvailGB { [math]::Round((New-Object Microsoft.VisualBasic.Devices.ComputerInfo).AvailablePhysicalMemory / 1GB, 1) }
# The shared PC's RAM rule (queue, 11-07 23:10: when a lock falls, every lane's suspended job comes back at once): a step
# over 8 GB looks at free RAM before it starts and before it resumes from a suspend, and waits while it is short.
function Wait-FreeRam([double]$WantGB, [string]$Log, [string]$What, [int]$MaxHours = 6) {
    if ((Get-AvailGB) -ge $WantGB) { return $true }
    Say $Log "$What waits for $WantGB GB of free RAM (now $(Get-AvailGB) GB)"
    $until = (Get-Date).AddHours($MaxHours)
    while ((Get-AvailGB) -lt $WantGB -and (Get-Date) -lt $until) { Start-Sleep -Seconds 30 }
    if ((Get-AvailGB) -lt $WantGB) { Say $Log "$What`: under $WantGB GB of free RAM for $MaxHours h"; return $false }
    return $true
}
# The sum of every GPU engine's use (WMI; the bench job's reading). Only for runs on the Intel GPU.
function Get-GpuEnginePercent {
    try {
        $e = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine -ErrorAction Stop)
        if ($e.Count -eq 0) { return $null }
        return [math]::Round(($e | Measure-Object UtilizationPercentage -Sum).Sum, 1)
    } catch { return $null }
}

# The lane's gate after taking the lock: three 5 s windows in a row with CPU under 15 %, no RTX memory in use and the
# build tools under 200 ms; at most $MaxMin minutes. Returns PASS or TIMEOUT and the readings.
function Pass-LaneGate([string]$Log, [int]$MaxMin = 10) {
    $deadline = (Get-Date).AddMinutes($MaxMin)
    $ok = 0
    $seen = @()
    while ((Get-Date) -lt $deadline) {
        $l = Measure-Load 5
        $g = Get-NvidiaMiB
        $seen += "cpu $($l.cpu) % build $($l.build_ms) ms rtx $g MiB"
        if ($l.cpu -lt 15 -and $l.build_ms -lt 200 -and $g -ge 0 -and $g -lt 50) { $ok++ } else { $ok = 0; Say $Log "gate: not quiet: cpu $($l.cpu) %, build tools $($l.build_ms) ms, rtx $g MiB; top $($l.top -join '; ')" }
        if ($ok -ge 3) { Say $Log "gate: PASS ($($seen[-3..-1] -join ' | '))"; return "PASS" }
    }
    Say $Log "gate: TIMEOUT after $MaxMin min"
    return "TIMEOUT"
}

# The biggest private-memory holders on the PC, for the log when the per-run gate waits on RAM (who holds it, read
# inside our own lock, so no other connection is needed).
function Get-RamHolders([int]$Top = 3) {
    try {
        $cl = @{}
        Get-CimInstance Win32_Process -Property ProcessId, CommandLine | ForEach-Object { $cl[[int]$_.ProcessId] = "$($_.CommandLine)" }
        return @(Get-Process | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First $Top | ForEach-Object {
            $c = $cl[$_.Id]; if ($c -and $c.Length -gt 140) { $c = $c.Substring(0, 140) }
            "$($_.ProcessName) $($_.Id), private $([math]::Round($_.PrivateMemorySize64 / 1GB, 1)) GB: $c"
        })
    } catch { return @("(not read: $($_.Exception.Message))") }
}

# The per-run gate of the bench's job (windows/run_job.ps1 Wait-Quiet): CPU under 25 % in three 1 s reads, NVIDIA
# memory under 800 MiB, and for an Intel GPU run the GPU engines under 10 %; at most 30 min. Also available RAM of at
# least $RamNeedGB (the plan's per-cell need: the package's bytes x 1.2 + 2 GB, so the run does not page).
# RAM short for $RamMaxMin minutes in a row returns quietGate "ram-short" (the chain stops the sitting): waiting longer
# under our own lock frees nothing when the holder is another lane's process that waits for that lock (2026-11-08).
function Wait-RunQuiet([string]$Log, [switch]$Engines, [int]$MaxMin = 30, [double]$RamNeedGB = 8, [int]$RamMaxMin = 10) {
    $deadline = (Get-Date).AddMinutes($MaxMin)
    $last = ""
    $ramShortSince = $null; $ramSaid = $null
    while ($true) {
        $reads = @(1..3 | ForEach-Object { (Measure-Load 1).cpu })
        $cpu = ($reads | Measure-Object -Maximum).Maximum
        $gpu = Get-NvidiaMiB
        $ram = Get-AvailGB
        $eng = $null
        if ($Engines) { $eng = Get-GpuEnginePercent }
        $engOk = (-not $Engines) -or ($null -eq $eng) -or ($eng -lt 10)
        if ($cpu -lt 25 -and $gpu -lt 800 -and $ram -ge $RamNeedGB -and $engOk) {
            return [ordered]@{ quietGate = "quiet"; cpuReads = $reads; nvidiaMiB = $gpu; availGB = $ram; ramNeedGB = $RamNeedGB; gpuEngines = $eng }
        }
        $text = "host not quiet: cpu $cpu %, nvidia $gpu MiB, available RAM $ram GB (need $RamNeedGB)" + $(if ($Engines) { ", gpu engines $eng %" } else { "" })
        if ($ram -lt $RamNeedGB) {
            if (-not $ramShortSince) { $ramShortSince = Get-Date }
            if (-not $ramSaid -or ((Get-Date) - $ramSaid).TotalMinutes -ge 5) {
                foreach ($h in (Get-RamHolders)) { Say $Log "  RAM holder: $h" }
                $ramSaid = Get-Date
            }
            if (((Get-Date) - $ramShortSince).TotalMinutes -ge $RamMaxMin) {
                Say $Log "$text, RAM short for $RamMaxMin min in a row: stopping"
                return [ordered]@{ quietGate = "ram-short"; cpuReads = $reads; nvidiaMiB = $gpu; availGB = $ram; ramNeedGB = $RamNeedGB; gpuEngines = $eng }
            }
        } else { $ramShortSince = $null }
        if ((Get-Date) -ge $deadline) {
            Say $Log "$text, past the $MaxMin min wait: running"
            return [ordered]@{ quietGate = "timeout"; cpuReads = $reads; nvidiaMiB = $gpu; availGB = $ram; ramNeedGB = $RamNeedGB; gpuEngines = $eng }
        }
        if ($text -ne $last) { Say $Log $text; $last = $text }
        Start-Sleep -Seconds 30
    }
}

# GPU memory of one process from the Windows counters (\GPU Process Memory(pid_<pid>_luid_*)\Dedicated / Shared
# Usage), MiB per adapter LUID; $null when the category does not answer.
$script:GpmCat = $null
function Get-ProcGpuMiB([int]$ProcId) {
    try {
        if (-not $script:GpmCat) { $script:GpmCat = New-Object System.Diagnostics.PerformanceCounterCategory("GPU Process Memory") }
        $out = @{}
        foreach ($inst in @($script:GpmCat.GetInstanceNames() | Where-Object { $_ -like "pid_$($ProcId)_*" })) {
            $luid = ($inst -split "_luid_")[1] -replace "_phys_.*$", ""
            $ded = New-Object System.Diagnostics.PerformanceCounter("GPU Process Memory", "Dedicated Usage", $inst, $true)
            $sha = New-Object System.Diagnostics.PerformanceCounter("GPU Process Memory", "Shared Usage", $inst, $true)
            $d = [math]::Round($ded.RawValue / 1MB); $s = [math]::Round($sha.RawValue / 1MB)
            if (-not $out.ContainsKey($luid)) { $out[$luid] = @{ dedicated = 0; shared = 0 } }
            $out[$luid].dedicated += $d; $out[$luid].shared += $s
        }
        return $out
    } catch { return $null }
}
function Merge-GpuPeak($Peak, $Now) {
    if ($null -eq $Now) { return $Peak }
    foreach ($k in $Now.Keys) {
        if (-not $Peak.ContainsKey($k)) { $Peak[$k] = @{ dedicated = 0; shared = 0 } }
        if ($Now[$k].dedicated -gt $Peak[$k].dedicated) { $Peak[$k].dedicated = $Now[$k].dedicated }
        if ($Now[$k].shared -gt $Peak[$k].shared) { $Peak[$k].shared = $Now[$k].shared }
    }
    return $Peak
}
