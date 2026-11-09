#!/bin/bash
# The third visit (after the 13:00 kill inside another lane's lock), at the end of another lane's cleanup in the PC's order: push, then
# the references and the Whisper GenAI export side by side (about 13 GB together: the PC order's 16 GB cap for untimed
# work without a lock), detached without yield_run (killed around other lanes' locks, run again after 60 s free).
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
pcsh="$K/tools/pc.sh"
cd "$K" || exit 1
"$pcsh" push | tail -1 || exit 1
for l in refs export-whisper; do
  "$pcsh" launch 'pc\steps.ps1' "-List $l" 2>&1 | tr -d '\r' | grep -v 'CLIXML\|<Objs' | tail -1
done
