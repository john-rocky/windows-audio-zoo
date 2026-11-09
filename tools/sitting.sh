#!/bin/bash
# One timed Z3 sitting on the PC, from the Mac:
#   1. wait on the Mac (the shared feed, ~/code/standup/tools/winpc/busy_now.sh, every 10 s, no ssh) until busy.txt
#      has been free for 60 s in a row
#   2. the Z3 lock ("Z3 timing: sitting <plan> ... until HH:MM") and the lane's quiet gate (3 x 5 s under 15 % CPU,
#      RTX empty, build tools quiet; at most 10 min, else the lock goes back and the sitting waits 10 min)
#   3. pc/chain.ps1 -Mode timed on the plan, detached: control-warm, control-head (stops the sitting when off the known
#      value by more than 5 %), the cells, control-tail; duty gaps and slots inside. This side waits in one long
#      connection (pc/wait_done.ps1), pulls runs/<plan> into raw/ while the lock is still ours (the chain holds it
#      20 min for that), then releases it.
#
#   tools/sitting.sh <plan name> [--resume] [--wait-hours N]
# --resume: the chain still runs and this side's wait ended or was stopped: wait for it again, then pull.
# Log: raw/<plan>.sitting.log. Exit 1 when the lock, the gate or the chain failed (the lock is released on every exit).
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
plan="$1"; shift || true
resume=0; waith=12
while (( $# )); do
  case "$1" in --resume) resume=1 ;; --wait-hours) waith="$2"; shift ;; esac
  shift
done
mkdir -p "$K/raw"
log="$K/raw/$plan.sitting.log"
pcsh="$K/tools/pc.sh"
R="${Z3_PC_ROOT:-C:\\src\\audiozoo}"   # the PC folder (the same default as tools/pc.sh)
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$log"; }
release() { say "release: $(Z3_FORCE=1 "$pcsh" lock release timing "${1:-0}" 2>&1 | tail -1)"; }
until_hm=$(python3 -c "
import json, datetime as d
p = json.load(open('$K/pc/plans/$plan.json'))
s = sum((u.get('est_sec') or 60 * u.get('timeout_min', 10)) + 90 for u in p['units'] for _ in range(u.get('runs', 1)))
t = d.datetime.now().astimezone() + d.timedelta(seconds=s + 300 + 180)
if p.get('end_by'):   # the chain starts no cell that would end past it (its tail row included): the lock goes back by then
    t = min(t, d.datetime.fromisoformat(p['end_by']))
print(t.strftime('%H:%M'))")

free_for() {  # seconds busy.txt has been free in the feed, 0 when held (or the feed is stale)
  local f; f=$(bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null) || { echo 0; return; }
  printf '%s\n' "$f" | awk '$3=="BUSY"' | tail -n 1 | python3 -c "
import sys, datetime as d
l = sys.stdin.read().split()
if len(l) < 4 or l[3] != 'none': print(0)
else: print(int((d.datetime.now(d.timezone.utc) - d.datetime.strptime(l[0], '%Y-%m-%dT%H:%M:%S%z')).total_seconds()))"
}

ours_now() {  # the feed's lock is Z3's own (the first-pass block of Z3's slot runs right before the sitting)
  bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null | awk '$3=="BUSY"' | tail -n 1 | grep -q ' BUSY Z3 '
}

take() {
  local r n
  while true; do
    n=0
    while ! ours_now && (( $(free_for) < 60 )); do (( n++ % 30 == 0 )) && say "waiting for busy.txt to be free 60 s (feed)"; sleep 10; done
    r=$("$pcsh" lock take "timing: sitting $plan until $until_hm" 2>&1 | tail -1); say "lock: $r"
    [[ "$r" == TAKEN* || "$r" == OURS* || "$r" == STALE* ]] || return 1
    r=$(Z3_FORCE=1 "$pcsh" lock gate 2>&1 | tail -1); say "gate: $r"
    [[ "$r" == PASS* ]] && return 0
    release 600; say "gate did not open: waiting 10 min before taking the lock again (window.txt)"; sleep 600
  done
}

if (( resume )); then
  say "sitting $plan resume: waiting for its chain again"
else
  say "sitting $plan start (expected end about $until_hm)"
  take || { say "no lock"; exit 1; }
  trap 'release' EXIT
  r=$(Z3_FORCE=1 "$pcsh" launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\$plan.json -Mode timed" 2>&1 | tail -1); say "chain: $r"
  [[ "$r" == STARTED* ]] || exit 1
fi
Z3_FORCE=1 "$pcsh" wait "$R\\runs\\$plan\\chain.done" "$R\\runs\\$plan\\chain.log" "$waith" >> "$log" 2>&1
cw=$(tail -1 "$log"); say "chain wait: $cw"
if [[ "$cw" != DONE* ]]; then
  trap - EXIT
  say "the wait ended before the chain's done file (the chain holds the lock): nothing released; tools/sitting.sh $plan --resume"
  exit 1
fi
st=$(Z3_FORCE=1 "$pcsh" lock status 2>&1 | head -1); say "lock after the chain: $st"
if [[ "$st" == "busy: Z3 "* ]]; then
  Z3_FORCE=1 "$pcsh" pull "runs/$plan" raw && say "pulled runs\\$plan -> raw/$plan"
else say "the lock is not ours after the chain: no pull now (pull runs/$plan in the next own lock or free time)"; fi
release
trap - EXIT
say "sitting $plan end: $cw"
[[ "$cw" == "DONE OK"* ]]
