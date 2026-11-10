#!/bin/bash
# Take Z3's untimed lock back after another lane's short block, from the Mac (the shared feed only, no ssh until the take):
#   1. wait until the feed shows another lane's lock (give up after <wait min>: then the PC counts as free)
#   2. wait until busy.txt has been free 60 s in a row
#   3. tools/pc.sh lock take "<what>" (pc\lock.ps1 waits its own 60 s free on the PC)
#
#   mtools/retake_after.sh "<what>" [wait min, default 20]
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
what="$1"; waitmin="${2:-20}"
say() { echo "[$(date '+%F %T %Z')] $*"; }
last_busy() { bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null | awk '$3=="BUSY"' | tail -n 1; }
free_for() {  # seconds busy.txt has been free in the feed, 0 when held
  last_busy | python3 -c "
import sys, datetime as d
l = sys.stdin.read().split()
if len(l) < 4 or l[3] != 'none': print(0)
else: print(int((d.datetime.now(d.timezone.utc) - d.datetime.strptime(l[0], '%Y-%m-%dT%H:%M:%S%z')).total_seconds()))"
}
t0=$(date +%s); seen=""
while [[ -z "$seen" ]]; do
  b=$(last_busy)
  if [[ "$b" == *" BUSY "* && "$b" != *" BUSY none"* && "$b" != *" BUSY ${Z3_LABEL:-Z3} "* ]]; then seen="$b"; say "other lock: $b"; break; fi
  (( $(date +%s) - t0 > waitmin * 60 )) && { say "no other lock in $waitmin min"; break; }
  sleep 10
done
while (( $(free_for) < 60 )); do sleep 10; done
say "free 60 s: taking"
r=$("$K/tools/pc.sh" lock take "$what" 2>&1 | tr -d '\r' | tail -1); say "lock: $r"
[[ "$r" == TAKEN* || "$r" == OURS* || "$r" == STALE* ]]
