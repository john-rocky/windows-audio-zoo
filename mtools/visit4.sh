#!/bin/bash
# The fourth visit, at Z3's export slot in the PC's order (after another lane's job): the Qwen3-ASR OpenVINO exports under Z3's lock
# ("Z3 export-ov ... (not a timing)", taken after 60 s free, kept between the two and then kept as "first pass (not a
# timing) until <the sitting's start>"), and beside them the two first-pass chains (each unit waits for its RAM).
#   mtools/visit4.sh <HH:mm the sittings start>
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
pcsh="$K/tools/pc.sh"
R='C:\src\audiozoo'
until="${1:?the sittings start HH:mm}"
cd "$K" || exit 1
"$pcsh" launch 'pc\steps.ps1' "-List export-ov -KeepUntil $until" 2>&1 | tr -d '\r' | grep -v 'CLIXML\|<Objs' | tail -1
for s in first-a first-an; do
  "$pcsh" launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\$s.json -Mode first" 2>&1 | tr -d '\r' | grep -v 'CLIXML\|<Objs' | tail -1
done
