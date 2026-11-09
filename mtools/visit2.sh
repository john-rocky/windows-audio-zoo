#!/bin/bash
# The second visit (README walk step 3): push the plans, start the exports and the references under the yield wrapper,
# and the first pass of the cells on downloaded files: one chain for the CPU / Intel GPU / RTX cells, one beside it for
# the NPU cells (first mode). Each pc.sh call reads the lock feed first.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
pcsh="$K/tools/pc.sh"
R='C:\src\audiozoo'
cd "$K" || exit 1
python3 tools/make_plan.py first --sitting first-a --group a --devices cpu,igpu,rtx | tail -1
python3 tools/make_plan.py first --sitting first-an --group a --devices npu | tail -1
"$pcsh" push | tail -1 || exit 1
# optional arguments: step lists to start first (e.g. redo-setup = the setup steps that failed, tools/make_plan.py redo)
for l in "$@" export refs; do
  "$pcsh" launch 'pc\steps.ps1' "-List $l" yield 2>&1 | tr -d '\r' | grep -v 'CLIXML\|<Objs' | tail -1
done
for s in first-a first-an; do
  "$pcsh" launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\$s.json -Mode first" 2>&1 | tr -d '\r' | grep -v 'CLIXML\|<Objs' | tail -1
done
