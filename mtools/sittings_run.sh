#!/bin/bash
# The three timed sittings in a row, from the Mac: s1 (CPU) sets the control row's known value from its head (3 runs),
# then s2 (Intel GPU + NPU) and s3 (RTX) are planned with that value (tools/sittings.py --known) and run.
#
#   mtools/sittings_run.sh [--after <lane>] [--from s1|s2|s3] [--end-by <iso>] [--slots '<json>']
#   mtools/sittings_run.sh --left <name> [--own-head] [--no-windows] [--end-by <iso>] [--not-before <iso>]
#                          [--after <lane> [--after-quiet <s>]]
#                                                              # a later sitting of the cells an end_by left
#
# --after <label>: first wait (the shared feed only, no ssh) until that lane's lock has come and gone, so the sittings
#   start in the PC's order and not in a gap before it (tools/sitting.sh then waits its own 60 s free). A lane that
#   takes the lock several times needs the start by hand after its last return instead.
# --end-by: no cell runs past it (pc/chain.ps1 leaves the rest, control.json left_cells); a sitting whose settle,
#   control rows and first cell would not end before it is not started. The cells left go to raw/left_cells.txt.
# --after-quiet <s>: after that lane's lock has gone, also wait until the feed has shown no lock for <s> seconds in a row
#   (a lane that takes the lock again after a short gap, e.g. two timing sets and a smoke, goes first).
# --not-before: first wait (no ssh) until that time; --after then looks for the lane's lock only from then on (a lane
#   that took the lock earlier in the day does not start the sittings in a gap before its later slot).
# --left <name>: one sitting named <name> of the cells in raw/left_cells.txt (cells may be added to it by hand first,
#   e.g. those of a sitting without a passing tail row), with s1's known control value. raw/s1 is pulled first when
#   its control.json is missing (a sitting stopped from the Mac at end_by); pc/ and shelf/ go to the PC with a parse
#   check of the chain (a later sitting may run a newer chain than the one the plans were last pushed with).
# --own-head: the --left sitting's control head runs 3 times and sets its own known value instead of s1's (the PC's
#   conditions changed since s1: 11-09 20:12 MSI Center fixed at Extreme Performance). A head that stops the
#   sitting (spread over 5 %) is taken once more from the start, its records kept as <name>-headfail-<time>.
# --no-windows: the sittings are one block to --end-by (the PC's order of the night 11-09/10): the chain opens no duty
#   window for other lanes' notes (tools/make_plan.py --no-windows; a slot_now.txt request still applies).
# Log: raw/sittings_run.log; each sitting's own log is raw/<s>.sitting.log.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
cd "$K"
after=""; from="s1"; slots=""; endby=""; leftname=""; notbefore=""; ownhead=0; afterquiet=0; nowindows=0
while (( $# )); do
  case "$1" in
    --after) after="$2"; shift ;; --from) from="$2"; shift ;; --slots) slots="$2"; shift ;;
    --end-by) endby="$2"; shift ;; --left) leftname="$2"; shift ;; --not-before) notbefore="$2"; shift ;;
    --own-head) ownhead=1 ;; --after-quiet) afterquiet="$2"; shift ;; --no-windows) nowindows=1 ;;
  esac
  shift
done
FIRST="first-a,first-an,first-b,first-bn,first-a2"   # first-a2: the Parakeet Intel GPU redo (runner fix, 11-09)
log="$K/raw/sittings_run.log"
leftf="$K/raw/left_cells.txt"
R="${Z3_PC_ROOT:-C:\\src\\audiozoo}"
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$log"; }
last_busy() { bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null | awk '$3=="BUSY"' | tail -n 1; }
plan() {  # plan <only> [known] [cells]
  local a=(python3 tools/sittings.py --first "$FIRST" --only "$1")
  [[ -n "${2:-}" ]] && a+=(--known "$2")
  [[ -n "${3:-}" ]] && a+=(--cells "$3")
  [[ -n "$slots" ]] && a+=(--slots "$slots")
  [[ -n "$endby" ]] && a+=(--end-by "$endby")
  (( nowindows )) && a+=(--no-windows)
  "${a[@]}" 2>&1 | tee -a "$log"
}
feed_free() {  # feed_free <what>: wait (feed only, at most 60 min) until no other lane holds a timing lock
  local i feed b
  for i in $(seq 1 120); do
    if ! feed=$(bash ~/code/standup/tools/winpc/busy_now.sh 2>/dev/null); then
      (( i == 1 )) && say "feed stale: $1 waits"; sleep 30; continue
    fi
    b=$(printf '%s\n' "$feed" | awk '$3=="BUSY"' | tail -n 1)
    if [[ "$b" == *"timing:"* && "$b" != *" BUSY Z3 "* ]]; then
      (( i == 1 )) && say "another lane's timing lock: $1 waits (${b:0:110})"; sleep 30; continue
    fi
    return 0
  done
  say "$1: no free feed in 60 min"; return 1
}
push_plans() {  # only the plan files (the runners on the PC stay as they are); no ssh during another lane's timing lock
  feed_free "the push" || return 1
  COPYFILE_DISABLE=1 tar -cf - pc/plans | ssh winpc "tar -xf - -C ${Z3_PC_ROOT_FWD:-C:/src/audiozoo}" && { say "plans pushed"; return 0; }
  say "plans not pushed (ssh / tar failed)"; return 1
}
push_code() {  # pc/ and shelf/, then every pc/*.ps1 of the Mac parsed on the PC with the same sha
  local out want got miss ow
  feed_free "the push" || return 1
  tools/pc.sh push >> "$log" 2>&1 || { say "code not pushed (tools/pc.sh push failed)"; return 1; }
  out=$(tools/pc.sh ps 2>&1 <<EOF
Get-ChildItem '$R\\pc\\*.ps1' | ForEach-Object {
    \$e = \$null
    [void][System.Management.Automation.Language.Parser]::ParseFile(\$_.FullName, [ref]\$null, [ref]\$e)
    \$n = if (\$e) { \$e.Count } else { 0 }
    'ps1 ' + \$_.Name + ' ' + \$n + ' ' + (Get-FileHash \$_.FullName -Algorithm SHA256).Hash.Substring(0, 16)
}
try { . '$R\\pc\\zlib.ps1'; \$o = @(Get-OthersWaiting); 'others waiting ' + \$o.Count + ': ' + (\$o -join ', ') }
catch { 'others waiting FAILED ' + \$_.Exception.Message }
EOF
)
  got=$(printf '%s\n' "$out" | tr -d '\r' | grep '^ps1 ' | sort)
  want=$(for f in pc/*.ps1; do printf 'ps1 %s 0 %s\n' "${f##*/}" "$(shasum -a 256 "$f" | cut -c1-16 | tr 'a-f' 'A-F')"; done | sort)
  miss=$(comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$got"))
  ow=$(printf '%s\n' "$out" | tr -d '\r' | grep '^others waiting ' | tail -n 1)
  if [[ -n "$want" && -z "$miss" && -n "$ow" && "$ow" != *FAILED* ]]; then
    say "pushed pc/, shelf/: $(grep -c . <<< "$want") .ps1 files parse on the PC with the Mac's sha; $ow"; return 0
  fi
  say "pc/*.ps1 on the PC: not parsed, not the Mac's, or zlib's Get-OthersWaiting failed (missing / PC / others):"
  printf '%s\n' "$miss" "--- PC:" "$got" "${ow:-no others line}" | tee -a "$log"
  return 1
}
fits() {  # fits <plan name>: its settle, gate, control rows and first cell end before end_by
  [[ -z "$endby" ]] && return 0
  python3 - "$K/pc/plans/$1.json" "$endby" <<'PY' | tee -a "$log"
import json, sys, datetime as d
p = json.load(open(sys.argv[1]))
end = d.datetime.fromisoformat(sys.argv[2])
units = p["units"]
need = 180 + 120
for u in units:
    if u.get("role"):
        need += ((u.get("est_sec") or 600) + 90) * u.get("runs", 1)
cells = [u for u in units if not u.get("role")]
if cells:
    need += (cells[0].get("est_sec") or 60 * cells[0]["timeout_min"]) + 90
ok = d.datetime.now().astimezone() + d.timedelta(seconds=need) <= end
print(f"{p['sitting']}: {'fits' if ok else 'does not fit'} (its control rows and first cell need {need / 60:.0f} min, end by {end:%H:%M})")
sys.exit(0 if ok else 1)
PY
}
retake_head() {  # retake_head <name>: the records of a sitting its control head stopped are kept under another name
  local n="$1" tag out
  tag="$1-headfail-$(date +%H%M%S)"
  sleep 30   # the chain's own end (it releases nothing more; its last log lines) before its folder is renamed
  feed_free "the rename of runs\\$n" || return 1
  out=$(tools/pc.sh ps 2>&1 <<EOF
Rename-Item -LiteralPath '$R\\runs\\$n' -NewName '$tag'
'renamed ' + (Test-Path '$R\\runs\\$tag') + ' ' + (Test-Path '$R\\runs\\$n')
EOF
)
  [[ "$out" == *"renamed True False"* ]] || { say "runs\\$n not renamed on the PC (${out##*$'\n'}): no retake"; return 1; }
  [[ -d "raw/$n" ]] && mv "raw/$n" "raw/$tag"
  mv "raw/$n.sitting.log" "raw/$tag.sitting.log"
  say "control head stopped $n: its records are $tag (PC runs\\ and raw/); $n once more from its start"
}
cells_of() { python3 -c "import json; print(','.join(u['cell'] for u in json.load(open('pc/plans/$1.json'))['units'] if not u.get('role')))"; }
left_of() {  # the cells a finished sitting left (control.json left_cells)
  python3 -c "
import json
try: print(','.join(json.load(open('raw/$1/control.json', encoding='utf-8-sig')).get('left_cells') or []))
except Exception: print('')"
}
add_left() { [[ -n "$1" ]] && { tr ',' '\n' <<< "$1" >> "$leftf"; say "left for a later sitting: $1"; }; }
known_s1() {
  python3 -c "
import json
c = json.load(open('raw/s1/control.json', encoding='utf-8-sig'))
print(c.get('known_rtf') or '')" 2>/dev/null
}

if [[ -n "$notbefore" ]]; then
  say "waiting until $notbefore (no ssh)"
  until python3 -c "import datetime as d, sys; sys.exit(d.datetime.now().astimezone() < d.datetime.fromisoformat('$notbefore'))"; do sleep 30; done
fi
if [[ -n "$after" ]]; then
  say "waiting for $after's lock to come and go (feed)"
  until last_busy | grep -q " BUSY $after "; do sleep 15; done
  say "$after's lock: $(last_busy)"
  while last_busy | grep -q " BUSY $after "; do sleep 15; done
  say "$after's lock gone: $(last_busy)"
  if (( afterquiet > 0 )); then
    say "waiting for the lock to stay free $afterquiet s (feed)"
    q=0
    while (( q < afterquiet )); do
      if last_busy | grep -q " BUSY none"; then q=$((q + 15)); else q=0; fi
      sleep 15
    done
    say "the lock stayed free $afterquiet s: $(last_busy)"
  fi
fi

if [[ -n "$leftname" ]]; then
  if [[ ! -s raw/s1/control.json ]]; then
    feed_free "the pull of runs\\s1" && tools/pc.sh pull runs/s1 raw && say "pulled runs\\s1 -> raw/s1 (its control.json was missing)"
  fi
  known=""
  if (( ! ownhead )); then
    known=$(known_s1); [[ -n "$known" ]] || { say "no known control value in raw/s1/control.json"; exit 1; }
  fi
  cells=$(grep -v '^$' "$leftf" | sort -u | paste -sd, -)
  [[ -n "$cells" ]] || { say "no left cells in $leftf"; exit 0; }
  mv "$leftf" "$leftf.$(date +%H%M%S)"
  plan "$leftname" "$known" "$cells" || { add_left "$cells"; exit 1; }
  fits "$leftname" || { add_left "$cells"; exit 1; }
  push_code || { add_left "$cells"; exit 1; }
  for try in 1 2; do
    tools/sitting.sh "$leftname" && break
    if (( try == 1 && ownhead )) && grep -q "sitting $leftname end: DONE FAIL control head" "raw/$leftname.sitting.log"; then
      retake_head "$leftname" && continue
    fi
    say "$leftname did not finish (see raw/$leftname.sitting.log)"; add_left "$cells"; exit 1
  done
  add_left "$(left_of "$leftname")"
  say "sitting $leftname done"
  exit 0
fi

if [[ "$from" == "s1" ]]; then
  plan s1 || exit 1
  fits s1 || { say "s1 does not fit before $endby: nothing started"; exit 1; }
  push_plans || exit 1
  tools/sitting.sh s1 || { say "s1 did not finish (see raw/s1.sitting.log)"; exit 1; }
  add_left "$(left_of s1)"
fi
known=$(known_s1)
[[ -n "$known" ]] || { say "no known control value in raw/s1/control.json"; exit 1; }
say "control known rtf from s1: $known"
for s in s2 s3; do
  [[ "$from" == "s3" && "$s" == "s2" ]] && continue
  plan "$s" "$known" || exit 1
  if ! fits "$s"; then add_left "$(cells_of "$s")"; continue; fi
  push_plans || exit 1
  tools/sitting.sh "$s" || { say "$s did not finish (see raw/$s.sitting.log)"; exit 1; }
  add_left "$(left_of "$s")"
done
say "sittings done$( [[ -s "$leftf" ]] && echo "; left for later: $(sort -u "$leftf" | paste -sd, -)")"
