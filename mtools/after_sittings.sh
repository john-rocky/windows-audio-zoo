#!/bin/bash
# After the timed sittings, from the Mac (README step 7): the TTS write-back of every timed Kokoro run with status OK
# (pc/writeback.py: one fixed ASR, OpenVINO GenAI Whisper large-v3-turbo fp16 on the PC's CPU; untimed, so pc/steps.ps1
# kills it for another lane's lock and runs it again from its start once the lock has been free 60 s), the pull of the
# sittings' runs (now with their write-backs) and of the references, then results/ and the table.
#
#   mtools/after_sittings.sh <sitting>[,<sitting>...]      e.g. s1,s2,s3,s4
#
# Z3_SCRUB_NAMES=<name>[,<name>...]: also replace those account names with USER in what was pulled (tools/scrub.py;
# the names stay out of this file). Log: raw/after_sittings.log. Exit 1 when a step list failed or a pull was refused.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
cd "$K"
[[ $# -ge 1 ]] || { sed -n 2,10p "$0"; exit 2; }
sits="$1"
FIRST="first-a,first-an,first-b,first-bn,first-a2"
R="${Z3_PC_ROOT:-C:\\src\\audiozoo}"
PY="${Z3_SCORE_PY:-$HOME/venvs/d702exp/bin/python}"   # tools/results.py: the scorers' normalizers
log="$K/raw/after_sittings.log"
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$log"; }
rc=0

lists=()
for s in ${sits//,/ }; do
  cells=$(python3 - "$s" <<'PY'
import json, pathlib, sys
out = []
for rj in sorted(pathlib.Path("raw", sys.argv[1], "kokoro").glob("*/run1.json")):
    try:
        if json.loads(rj.read_text(encoding="utf-8-sig")).get("status") == "OK":
            out.append(f"kokoro/{rj.parent.name}")
    except ValueError:
        pass
print(",".join(out))
PY
)
  if [[ -z "$cells" ]]; then say "$s: no timed Kokoro run with status OK"; continue; fi
  python3 tools/make_plan.py writeback --sitting "$s" --cells "$cells" 2>&1 | tee -a "$log"
  lists+=("writeback-$s")
done

if (( ${#lists[@]} )); then
  tools/pc.sh push 2>&1 | tail -n 2 | tee -a "$log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || { say "push refused or failed: nothing launched"; exit 1; }
  dones=""; logs=""
  for l in "${lists[@]}"; do
    say "launch $l: $(tools/pc.sh launch 'pc\steps.ps1' "-List $l" 2>&1 | tail -n 1)"
    dones+="${dones:+,}$R\\logs\\$l.done"; logs+="${logs:+,}$R\\logs\\$l.log"
  done
  tools/pc.sh wait "$dones" "$logs" 4 2>&1 | grep -E 'done|FAIL|killed|exit [^0]' | tee -a "$log"
  for l in "${lists[@]}"; do
    st=$(printf '%s\n' "Get-Content '$R\\logs\\$l.done'" | tools/pc.sh ps 2>&1 | tail -n 1)
    say "$l: $st"
    [[ "$st" == OK* ]] || rc=1
  done
fi

for s in ${sits//,/ }; do
  if tools/pc.sh pull "runs/$s" raw 2>>"$log"; then say "pulled runs\\$s -> raw/$s"; else say "pull of runs\\$s refused or failed"; rc=1; fi
done
if tools/pc.sh pull refs raw 2>>"$log"; then say "pulled refs -> raw/refs"; else say "pull of refs refused or failed"; rc=1; fi
if [[ -n "${Z3_SCRUB_NAMES:-}" ]]; then
  python3 tools/scrub.py --names "$Z3_SCRUB_NAMES" raw 2>&1 | tail -n 1 | tee -a "$log"
fi

"$PY" tools/results.py build --runs "$FIRST,$sits" 2>&1 | tail -n 3 | tee -a "$log"
"$PY" tools/results.py table 2>&1 | tail -n 2 | tee -a "$log"
say "after the sittings: done (rc $rc)"
exit "$rc"
