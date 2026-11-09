<#
The Z3 timing lock from the Mac, one SSH call per action (the wait loops run here, not as polls from the Mac).

  lock.ps1 -Action take -What "<text>"      waits (5 s reads) until busy.txt is ours: prints TAKEN / OURS / STALE / TIMEOUT
                                            (<text> in the shared PC's form: "timing: <what> until HH:MM")
  lock.ps1 -Action gate                     the lane's quiet gate (zlib Pass-LaneGate): prints PASS / TIMEOUT
  lock.ps1 -Action release [-OpenSec <s>]   removes busy.txt if it is ours: prints RELEASED / NOT-OURS; with -OpenSec,
                                            pcup\window.txt says "Z3 open until <now + s>" (a gap Z3 comes back from)
  lock.ps1 -Action status                   prints the marker, the active\ and yield\ notes
  lock.ps1 -Action slot -What "<who> <give up min>"   runs\slot_now.txt for the running timed chain (pc\chain.ps1):
                                            a slot for another lane at the user's word, no window.txt
  lock.ps1 -Action note -What "<text>"      pcup\active\Z3.txt = "<text>" + "written HH:MM" (the lane waits for a window);
                                            -What "" removes it
take removes Z3's window.txt once the lock is ours; release without -OpenSec removes it too (the PC is free, no window).
#>
param([Parameter(Mandatory = $true)][ValidateSet("take", "gate", "release", "status", "slot", "note")][string]$Action,
      [string]$What = "timing", [int]$MaxHours = 12, [int]$OpenSec = 0)
. "$PSScriptRoot\zlib.ps1"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$log = Join-Path $Root "logs\lock.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
# Z3's own untimed work (a first pass, a setup or export step) must not run beside a timed sitting: take refuses
# while one runs.
function Get-OwnUntimed {
    @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
            $c = "$($_.CommandLine)"
            $c -match '\\src\\audiozoo' -and ($c -match '\\pc\\(steps|setup)\.ps1' -or $c -match '\\pc\\chain\.ps1 .*-Mode first') } |
        ForEach-Object { "$($_.ProcessId) $("$($_.CommandLine)" -replace '^.*\\pc\\', 'pc\')" })
}
switch ($Action) {
    "take" {
        # a timing needs Z3's own untimed work stopped; a block "(not a timing)" (the first passes under Z3's lock) does not
        if ($What -notmatch "not a timing") {
            $own = Get-OwnUntimed
            if ($own.Count) { "OWN-UNTIMED $($own -join '; ') $(Get-Date -Format o)"; break }
        }
        $r = Take-BusyAfterFree $What $log 60 $MaxHours
        if ($r -eq "OURS") { Touch-Busy $What }   # Z3 held it already (a block of the same slot): the text says what runs now
        if (Test-OursBusy) { Close-Window }
        "$r $(Get-Date -Format o)"
    }
    "gate" { $r = Pass-LaneGate $log; "$r $(Get-Date -Format o)" }
    "release" {
        if (Test-OursBusy) {
            Release-Busy $log
            if ($OpenSec -gt 0) { Open-Window $OpenSec $log } else { Close-Window }
            "RELEASED $(Get-Date -Format o)"
        } else { "NOT-OURS $(Get-Date -Format o)" }
    }
    "status" {
        $b = Read-Busy
        if ($b) { "busy: $($b.Text) ($([int]$b.AgeMin) min)" } else { "busy: none" }
        Get-ChildItem (Join-Path $Pcup "active\*.txt"), (Join-Path $Pcup "yield\*.txt") -ErrorAction SilentlyContinue |
            ForEach-Object { "$($_.Directory.Name)\$($_.Name): $(([IO.File]::ReadAllText($_.FullName)).Trim() -replace '\s+', ' ')" }
    }
    "slot" {
        $f = Join-Path $Root "runs\slot_now.txt"
        New-Item -ItemType Directory -Force (Split-Path $f) | Out-Null
        Write-Utf8 $f $What
        Say $log "slot request: $What"
        "SLOT $What $(Get-Date -Format o)"
    }
    "note" {
        $f = Join-Path $Pcup "active\Z3.txt"
        if ($What) { Write-Utf8 $f "Z3 $What`r`nwritten $(Get-Date -Format HH:mm)"; "NOTE written $(Get-Date -Format o)" }
        else { Remove-Item $f -Force -ErrorAction SilentlyContinue; "NOTE removed $(Get-Date -Format o)" }
    }
}
