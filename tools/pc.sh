#!/bin/bash
# Mac-side calls to the PC (ssh winpc; its default shell is PowerShell), one SSH connection per call.
# Before every connection it reads the shared lock feed on the Mac (~/code/standup/tools/winpc/busy_now.sh, no ssh):
# another lane's "timing:" lock, or a feed older than 3 min, refuses the call (exit 3). Z3_FORCE=1 skips the read
# (tools/sitting.sh inside Z3's own lock; `wait` never reads it: it is the one long connection of a running job).
#
#   pc.sh status                      busy.txt, the active\ and yield\ notes, free disk
#   pc.sh ps                          run the PowerShell read from stdin (-EncodedCommand)
#   pc.sh push                        pc/ and shelf/ -> the PC folder ($Z3_PC_ROOT, default C:\src\audiozoo)
#   pc.sh pushfix <S150 dir> <S220 dir>   the clip sets -> <PC folder>\fixtures (S220 under fixtures\s220)
#   pc.sh pull <rel path> <K dest>    <PC folder>\<rel path> -> K/<K dest>/<its last part> (one tar, one connection;
#                                     without the cache\ and wav\ folders)
#   pc.sh py <pc\script.py> [args]   that script with the folder's .venv-ovg python (-I), in the foreground;
#                                     Z3_VENV=<name> picks .venv-<name>
#   pc.sh lock take|gate|release|status|slot|note ["what"] [open s]   pc\lock.ps1 (the waits run on the PC)
#   pc.sh launch <script> "<args>" [yield]          pc\launch.ps1 (detached through WMI, parse check first)
#   pc.sh wait <done file> <log file> [max hours]  pc\wait_done.ps1 (one long connection, the log's new lines)
set -euo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
R="${Z3_PC_ROOT:-C:\\src\\audiozoo}"   # the PC folder (README walk: another empty folder)
RF="${R//\\//}"                        # the same with forward slashes (ssh tar)

if [[ -z "${Z3_FORCE:-}" && "${1:-}" != "wait" ]]; then
  feed=$(bash ~/code/standup/tools/winpc/busy_now.sh 2>&1) || { echo "REFUSED (lock feed stale or missing): $feed" >&2; exit 3; }
  busy=$(printf '%s\n' "$feed" | awk '$3=="BUSY"' | tail -n 1)
  if [[ "$busy" == *"timing:"* && "$busy" != *" BUSY Z3 "* ]]; then
    echo "REFUSED (another lane's timing lock): ${busy}" >&2
    exit 3
  fi
fi

ps_run() {  # stdin: PowerShell text
  local b64
  b64=$( (printf '%s\n' "\$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8"; cat) \
    | iconv -f UTF-8 -t UTF-16LE | base64)
  ssh winpc "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $b64"
}

case "${1:-}" in
  status)
    ps_run <<'EOF'
"now $(Get-Date -Format o)"
$p = Join-Path ([Environment]::GetFolderPath('UserProfile')) 'pcup'
if (Test-Path "$p\busy.txt") { "busy: $((Get-Content "$p\busy.txt" -Raw).Trim()) (written $([int]((Get-Date) - (Get-Item "$p\busy.txt").LastWriteTime).TotalMinutes) min ago)" } else { "busy: none" }
Get-ChildItem "$p\active\*.txt", "$p\yield\*.txt" -ErrorAction SilentlyContinue | ForEach-Object { "$($_.Directory.Name)\$($_.Name): $(((Get-Content $_.FullName -Raw).Trim()) -replace '\s+', ' ')" }
"free C: $([math]::Round((Get-PSDrive C).Free / 1GB, 1)) GB"
EOF
    ;;
  ps) ps_run ;;
  py)
    script="$2"; shift 2
    ps_run <<EOF
\$env:ORT_DISABLE_TELEMETRY = '1'
& '$R\\.venv-${Z3_VENV:-ovg}\\Scripts\\python.exe' -I '$R\\$script' $* 2>&1 | ForEach-Object { "\$_" }
"exit \$LASTEXITCODE"
EOF
    ;;
  push)
    COPYFILE_DISABLE=1 tar -cf - -C "$K" --exclude __pycache__ pc shelf \
      | ssh winpc "New-Item -ItemType Directory -Force $RF | Out-Null; tar -xf - -C $RF"
    echo "pushed pc/, shelf/ -> $R"
    ;;
  pushfix)
    s150="$2"; s220="$3"
    COPYFILE_DISABLE=1 tar -cf - -C "$s150" meta.json fleurs examples \
      | ssh winpc "New-Item -ItemType Directory -Force $RF/fixtures | Out-Null; tar -xf - -C $RF/fixtures"
    COPYFILE_DISABLE=1 tar -cf - -C "$s220" clips.json clips \
      | ssh winpc "New-Item -ItemType Directory -Force $RF/fixtures/s220 | Out-Null; tar -xf - -C $RF/fixtures/s220"
    echo "pushed the clip sets -> $R\\fixtures"
    ;;
  pull)
    mkdir -p "$K/$3"
    # not the runtimes' compile caches ({RUN}\cache: GPU / NPU blobs) nor the TTS waves (the write-back reads them on the PC)
    ssh winpc "tar -cf - --exclude cache --exclude wav -C $RF/$(dirname "$2") $(basename "$2")" | tar -xf - -C "$K/$3"
    ;;
  lock)
    what="${3:-timing}"
    ps_run <<EOF
& '$R\\pc\\lock.ps1' -Action $2 -What '$what' -OpenSec ${4:-0}
EOF
    ;;
  launch)
    y=""; [[ "${4:-}" == "yield" ]] && y="-Yield"
    ps_run <<EOF
& '$R\\pc\\launch.ps1' -Script '$R\\$2' -ArgLine '$3' $y
EOF
    ;;
  wait)
    ps_run <<EOF
& '$R\\pc\\wait_done.ps1' -Done '$2' -Log '$3' -MaxHours ${4:-12}
EOF
    ;;
  *) sed -n 2,17p "$0"; exit 2 ;;
esac
