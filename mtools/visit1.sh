#!/bin/bash
# The first visit to the PC (README walk steps 1 and 2): the folder must not exist yet; push the scripts, the shelf and
# the two clip sets; start the setup and the Hub fetch under the shared yield wrapper. Each pc.sh call reads the lock
# feed first and refuses during another lane's timing.
set -uo pipefail
K="$(cd "$(dirname "$0")/.." && pwd)"
pcsh="$K/tools/pc.sh"
R='C:\src\audiozoo'
"$pcsh" ps <<'PS' || exit 1
"now $(Get-Date -Format o)"
"folder exists: $(Test-Path C:\src\audiozoo)"
"free C: $([math]::Round((Get-PSDrive C).Free / 1GB, 1)) GB"
"uv: $((Get-Command uv -ErrorAction SilentlyContinue).Source) $(& uv --version 2>&1)"
"yield_run: $(Test-Path (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'pcup\yield_run.ps1'))"
"RAM free GB: $([math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1))"
PS
