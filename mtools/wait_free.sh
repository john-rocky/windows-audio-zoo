#!/bin/bash
# Wait on the Mac (the shared lock feed, no ssh) until busy.txt is free or held as "(not a timing)" for N seconds in a
# row; prints the line it saw and exits 0 (or 1 after max minutes). Usage: wait_free.sh [N=20] [max min=180]
# The feed = the command in PC_FEED_CMD (tools/pc.sh header).
: "${PC_FEED_CMD:?set PC_FEED_CMD to the command that prints the PC lock feed (tools/pc.sh header)}"
n=${1:-20}; max=${2:-180}; ok=0; end=$(( $(date +%s) + max * 60 ))
while (( $(date +%s) < end )); do
  f=$(bash -c "$PC_FEED_CMD" 2>/dev/null) || { ok=0; sleep 10; continue; }
  b=$(printf '%s\n' "$f" | awk '$3=="BUSY"' | tail -n 1)
  if [[ "$b" == *" BUSY none"* || ( "$b" == *"not a timing"* && "$b" != *"timing:"* ) ]]; then
    ok=$(( ok + 5 )); (( ok >= n )) && { echo "FREE $(date '+%H:%M:%S') $b"; exit 0; }
  else ok=0; fi
  sleep 5
done
echo "TIMEOUT $(date '+%H:%M:%S')"; exit 1
