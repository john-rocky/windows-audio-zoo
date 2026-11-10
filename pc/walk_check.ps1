<#
The README walk's check, on the PC after the walk (Windows PowerShell 5.1): every file each README step should leave,
as "OK <step> <path>" or "MISSING <step> <path>", each step list's and chain's done line, and per first-pass or timed
sitting the run records against its plan. Exit 1 when a file is missing or a done line is absent.

  powershell -NoProfile -ExecutionPolicy Bypass -File pc\walk_check.ps1 [-Sittings first-a2,s1,s2,s3,s4]

first-a, first-an, first-b and first-bn are always checked; -Sittings adds the others. A done line that says FAIL is
printed as it is: the walk's record says which redo list filled it. A cell a timed sitting left at its end_by
(control.json left_cells) counts as "left (end_by)", not as missing: the later sitting that ran it belongs in -Sittings.

  powershell -NoProfile -ExecutionPolicy Bypass -File pc\walk_check.ps1 -Z3b -Sittings z3b-probe,z3b-first-g,...

-Z3b: the README's steps of 2026-11-10 ("again (2026-11-10)"): the pushed files, two venvs (ovg, litertlm), the three
Hub repos of setup-z3b / fetch-z3b, the NPU plugin folders of npujit-z3b, and only the sittings given.
#>
param([string[]]$Sittings = @(), [switch]$Z3b)   # "s1,s2" from -File, or s1,s2 (an array) from & inside PowerShell
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$script:miss = 0
function Want([string]$Step, [string]$Rel, [int64]$Size = -1) {
    $p = Join-Path $Root $Rel
    if (-not (Test-Path -LiteralPath $p)) { "MISSING $Step $Rel"; $script:miss++; return }
    if ($Size -ge 0 -and (Get-Item -LiteralPath $p).Length -ne $Size) { "MISSING $Step $Rel (size differs from the manifest)"; $script:miss++; return }
    "OK      $Step $Rel"
}
function Done([string]$Step, [string]$Rel) {
    $p = Join-Path $Root $Rel
    if (-not (Test-Path -LiteralPath $p)) { "MISSING $Step $Rel"; $script:miss++; return }
    "DONE    $Step $Rel = $(([IO.File]::ReadAllText($p)).Trim())"
}

# 1. push, pushfix
foreach ($f in "pc\chain.ps1", "pc\steps.ps1", "pc\zlib.ps1", "pc\lock.ps1", "pc\launch.ps1", "pc\wait_done.ps1",
               "pc\audiolib.py", "shelf\hub_manifest.json", "fixtures\meta.json", "fixtures\s220\clips.json") { Want "1" $f }
# 2. setup, fetch: the interpreter, eight venvs and their version lists, every Hub file at its manifest size
# (-Z3b: two venvs, the three repos of fetch-z3b)
Want "2" "python\cpython-3.12-windows-x86_64-none\python.exe"
$venvs = if ($Z3b) { @("ovg", "litertlm") } else { @("ovg", "genai", "genai-cuda", "ort", "ort-cuda", "litertlm", "ref", "ovexport") }
foreach ($v in $venvs) {
    Want "2" ".venv-$v\Scripts\python.exe"
    Want "2" "logs\versions-$v.txt"
}
$suffix = if ($Z3b) { "-z3b" } else { "" }
Done "2" "logs\setup$suffix.done"
Done "2" "logs\fetch$suffix.done"
Want "2" "models\fetch_report.json"
$man = @(Get-Content (Join-Path $Root "shelf\hub_manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json)
$man = @($man | ForEach-Object { $_ })   # PowerShell 5.1 reads a JSON list as one object
if ($Z3b) {
    $man = @($man | Where-Object { $_.repo -in "OpenVINO/whisper-large-v3-turbo-int8-ov", "litert-community/Qwen3-ASR-1.7B",
                                               "litert-community/granite-speech-5.0-2b" })
}
$nf = 0; $mf = $script:miss
foreach ($r in $man) {
    foreach ($f in @($r.files)) {
        $nf++
        $rel = "models\" + $r.repo.Replace("/", "__") + "\" + $f.path.Replace("/", "\")
        $line = Want "2" $rel ([int64]$f.size)
        if ($line -notlike "OK*") { $line }
    }
}
"OK      2 models: $nf files of $($man.Count) repos checked by size ($($script:miss - $mf) missing or different)"
if ($Z3b) {
    # the NPU plugin folders (pc\npu_plugin_dir.py): the compiler plugin beside a dispatch library, and their manifest
    foreach ($d in "p1", "p2") {
        foreach ($f in "LiteRtCompilerPlugin.dll", "LiteRtDispatch.dll", "manifest.json") { Want "npu" "npujit\$d\$f" }
    }
    Done "npu" "logs\npujit-z3b.done"
} else {
# 3. references, the Whisper export for ONNX Runtime GenAI
foreach ($m in "whisper", "parakeet", "kokoro") { Want "3" "refs\$m\ref.json" }
Want "3" "refs\kokoro\ref.writeback.json"
Done "3" "logs\refs.done"
foreach ($d in "whisper-genai-fp32-cpu", "whisper-genai-fp16-cuda") {
    foreach ($f in "genai_config.json", "encoder.onnx", "decoder.onnx") { Want "3" "exports\$d\$f" }
}
Done "3" "logs\export-whisper.done"
# 4. the Qwen3-ASR IR; 4 and 5: the first passes
foreach ($d in "qwen3asr-ov-fp16", "qwen3asr-ov-int8") {
    foreach ($f in "openvino_encoder_model.xml", "openvino_decoder_model.xml", "openvino_tokenizer.xml", "openvino_detokenizer.xml") { Want "4" "exports\$d\$f" }
}
Done "4" "logs\export-ov.done"
}
$always = @(if (-not $Z3b) { "first-a", "first-an", "first-b", "first-bn" })
$sits = $always + @($Sittings | ForEach-Object { $_ -split "," } | Where-Object { $_ })
foreach ($s in $sits) {
    $plan = Join-Path $Root "pc\plans\$s.json"
    if (-not (Test-Path -LiteralPath $plan)) { "MISSING sit $s pc\plans\$s.json"; $script:miss++; continue }
    Done "sit" "runs\$s\chain.done"
    $p = Get-Content $plan -Raw -Encoding UTF8 | ConvertFrom-Json
    $left = @()
    $cj = Join-Path $Root "runs\$s\control.json"
    if (Test-Path -LiteralPath $cj) {
        try { $left = @((Get-Content $cj -Raw -Encoding UTF8 | ConvertFrom-Json).left_cells | Where-Object { $_ }) } catch { }
    }
    $st = @{}
    foreach ($u in @($p.units)) {
        $n = if ($u.runs) { [int]$u.runs } else { 1 }
        for ($k = 1; $k -le $n; $k++) {
            $cdir = "runs\$s\" + $u.cell.Replace("/", "\")
            $rj = Join-Path $Root "$cdir\run$k.json"
            $mj = Join-Path $Root "$cdir\run$k.meta.json"
            $status = if (Test-Path -LiteralPath $rj) {
                try { (Get-Content $rj -Raw -Encoding UTF8 | ConvertFrom-Json).status } catch { "unreadable" }
            } elseif (Test-Path -LiteralPath $mj) { "no runner JSON" } elseif ($left -contains $u.cell) { "left (end_by)" } else { "MISSING" }
            if ($status -eq "MISSING") { "MISSING sit $cdir\run$k.json"; $script:miss++ }
            $st[$status] = 1 + [int]$st[$status]
        }
    }
    "RUNS    sit $s`: " + (($st.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name) $($_.Value)" }) -join ", ")
}
"walk check: $($script:miss) missing"
if ($script:miss) { exit 1 }
exit 0
