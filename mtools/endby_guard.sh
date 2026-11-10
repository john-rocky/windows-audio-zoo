#!/bin/bash
# A sitting's end_by, kept from the Mac when the chain on the PC cannot keep it itself (a chain started before
# pc/chain.ps1 learned to skip a take-back near end_by: 11-09's evening s1). At <end_by> (HH:MM:SS today), if the
# sitting's chain has not written its done file, stop it: never by an ssh during another lane's timing lock (then wait
# for that lock to go, and act inside the free minute before the chain could take the lock back). On the PC: kill the
# chain and its engine processes, take Z3's window.txt and busy.txt away if they are Z3's, and write the chain's done
# file ("FAIL stopped from the Mac at end_by"), so tools/sitting.sh ends as after any chain: pull, release, exit.
#
#   mtools/endby_guard.sh <sitting> <HH:MM:SS> [--dry]
# Log: raw/<sitting>.endby_guard.log.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
cd "$K"
sit="${1:?sitting}"; at="${2:?HH:MM:SS}"; dry="${3:-}"
R="${Z3_PC_ROOT:-C:\\src\\audiozoo}"
log="$K/raw/$sit.endby_guard.log"
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$log"; }
done_on_mac() { grep -q "sitting $sit end:" "raw/$sit.sitting.log" 2>/dev/null; }
last_busy() { bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null | awk '$3=="BUSY"' | tail -n 1; }

say "guard for sitting $sit armed: acts at $at if its chain has not ended"
while [[ "$(date +%H:%M:%S)" < "$at" ]]; do
  done_on_mac && { say "sitting $sit ended before $at: nothing to do"; exit 0; }
  sleep 10
done
done_on_mac && { say "sitting $sit ended: nothing to do"; exit 0; }
say "$at: sitting $sit has not ended"
while true; do
  done_on_mac && { say "sitting $sit ended while waiting: nothing to do"; exit 0; }
  b=$(last_busy)
  if [[ "$b" == *"timing:"* && "$b" != *" BUSY ${Z3_LABEL:-Z3} "* ]]; then
    sleep 3; continue   # another lane's timing: no ssh; its return frees a minute in which the chain cannot take back yet
  fi
  break
done
say "acting (lock: ${b:0:110})"
ps1=$(cat <<EOF
\$root = '$R'; \$sit = '$sit'
\$chains = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" | Where-Object { \$_.CommandLine -match 'chain\.ps1' -and \$_.CommandLine -match ('plans\\\\' + \$sit + '\.json') })
"chains: " + ((\$chains | ForEach-Object { \$_.ProcessId }) -join ', ')
if ('$dry' -ne '--dry') {
  foreach (\$c in \$chains) { & "\$env:SystemRoot\System32\taskkill.exe" /PID \$c.ProcessId /T /F 2>&1 | Out-Null }
  \$pc = Join-Path \$env:USERPROFILE 'pcup'
  foreach (\$f in @((Join-Path \$pc 'window.txt'), (Join-Path \$pc 'busy.txt'))) {
    if ((Test-Path \$f) -and ((Get-Content \$f -Raw).TrimStart([char]0xFEFF) -match '^Z3 ')) { Remove-Item \$f -Force; "removed \$f" }
  }
  \$done = Join-Path \$root "runs\\\$sit\\chain.done"
  if (-not (Test-Path \$done)) { [IO.File]::WriteAllText(\$done, "FAIL stopped from the Mac at end_by " + (Get-Date -Format o)); "wrote \$done" }
}
EOF
)
if [[ "$dry" == "--dry" ]]; then printf '%s\n' "$ps1" | Z3_FORCE=1 tools/pc.sh ps 2>&1 | tee -a "$log"; exit 0; fi
printf '%s\n' "$ps1" | Z3_FORCE=1 tools/pc.sh ps 2>&1 | tee -a "$log"
say "guard done"
