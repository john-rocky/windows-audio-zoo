<#
Starts a script detached through WMI (it outlives the SSH session and the Mac), after a parse check on this PC, and
checks 20 s later that the process it started still runs, or ended with exit 0 (a script that does not parse dies
without a done file, seen on this PC). The check reads that process by its pid: other lanes run scripts of the same
name. Prints STARTED <pid> or FAILED <why>.

  launch.ps1 -Script <path.ps1> -ArgLine "<arguments as one string>" [-Yield]

-Yield wraps it in %USERPROFILE%\pcup\yield_run.ps1 -Lane Z3 (untimed work: suspended while another lane holds
busy.txt).
#>
param([Parameter(Mandatory = $true)][string]$Script, [string]$ArgLine = "", [switch]$Yield)
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$errs = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$errs)
if ($errs -and $errs.Count) { "FAILED parse: $($errs | ForEach-Object { "$($_.Extent.StartLineNumber): $($_.Message)" })"; exit 1 }
# Paths here have no spaces (C:\src\audiozoo): the command line carries no inner quotes through yield_run's -Cmd string.
if ("$Script $ArgLine" -match '"') { "FAILED the script path and -ArgLine must not need quotes"; exit 1 }
$inner = "powershell -NoProfile -ExecutionPolicy Bypass -File $Script $ArgLine"
if ($Yield) {
    $yr = Join-Path ([Environment]::GetFolderPath("UserProfile")) "pcup\yield_run.ps1"
    if (Test-Path $yr) { $inner = "powershell -NoProfile -ExecutionPolicy Bypass -File $yr -Lane Z3 -Cmd `"$inner`"" }
    else { "NOTE no $yr (the shared PC's suspend-on-lock helper): started without it" }
}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $inner }
if ($r.ReturnValue -ne 0) { "FAILED Win32_Process.Create ReturnValue $($r.ReturnValue)"; exit 1 }
# hold a handle now: a script that ends within the 20 s (a list that fails at once) keeps its exit code readable
$proc = Get-Process -Id $r.ProcessId -ErrorAction SilentlyContinue
if ($proc) { [void]$proc.Handle }
Start-Sleep -Seconds 20
$leaf = Split-Path $Script -Leaf
if ($proc -and -not $proc.HasExited) { "STARTED $($r.ProcessId) ($leaf running 20 s after the start)" }
elseif ($proc -and $proc.ExitCode -eq 0) { "STARTED $($r.ProcessId) ($leaf ended within 20 s, exit 0: read its done file)" }
elseif ($proc) { "FAILED $leaf ended within 20 s with exit $($proc.ExitCode) (pid $($r.ProcessId))"; exit 1 }
else { "FAILED pid $($r.ProcessId) ($leaf) was gone before the first read"; exit 1 }
