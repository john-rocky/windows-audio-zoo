# The LiteRT-LM cells on the Intel GPU and the NPU, again (2026-11-10)

A cell is one combination of model, runtime, form and device. A sitting is one timed session on the PC. A walk is one run of every step from an empty folder.

Three LiteRT-LM cells had no number after the walk: Qwen3-ASR and Granite Speech on LiteRT-LM's GPU backend on the
Intel GPU (their runs took the RTX) and Granite Speech on LiteRT-LM's NPU backend (its first pass was stopped by
hand). They ran again on 11-10 under the label `Z3b` (the PC folder's `label.txt`; `Z3_LABEL=Z3b` for the Mac-side
scripts), in a new `C:\src\audiozoo` with the two venvs and the three Hub repos these cells and the control row read
(`pc/steps/setup-z3b.json`, `pc/steps/fetch-z3b.json`). The NPU cell then turned out to run on the CPU, and so did
the walk's NPU row of Qwen3-ASR; with a compiler plugin added, Qwen3-ASR's language model ran on the NPU, a new cell
(below).

Windows keeps a per-app GPU preference under `HKCU\Software\Microsoft\DirectX\UserGpuPreferences`: the value's name is
an executable's full path, its data `GpuPreference=1;` asks for the power-saving GPU. Which name it needs
(`raw/z3b-probe/`: Qwen3-ASR on the GPU backend, three clips per form; the adapter as the run's log names it,
`Selected adapter: ...`, and the GPU memory per adapter in its `run1.meta.json`):

| form | GpuPreference=1 written | value names | adapter |
|---|---|---|---|
| p1 | by `pc/chain.ps1`, before the engine process starts | the venv's launcher and the interpreter through uv's junction (`...\python\cpython-3.12-windows-x86_64-none\python.exe`) | Intel GPU |
| p2 | inside the process, before the runtime's import (`--gpu-pref 1 --gpu-pref-names asis`) | the same two paths | Intel GPU |
| p3 | by `pc/chain.ps1`, before the engine process starts | the launcher and the junction's target (`...\cpython-3.12.16-windows-x86_64-none\python.exe`, the name of the walk's runs) | RTX |
| p0 | not written | | RTX |

On this PC, in these runs, the value's name decided which GPU the backend took: the path the interpreter process runs
by (the path `Get-Process` shows for it) reached the Intel GPU, written before the start or inside the process; the
junction's target (what `os.path.realpath` and `QueryFullProcessImageNameW` return) did not. The Intel GPU cells now
take form p1: the plan's unit carries `gpu_pref` (`tools/make_plan.py`), the chain writes the value before the engine
starts and removes it after it ends, and the run's meta records the values written, the adapter line and the engine's
path.

The Intel GPU rows come from `raw/z3b-s5/`: one timed block on 11-10 from 12:15 to 12:59, its control head 0.26185
and tail 0.25959 against s4's known value 0.26228 (0.2 % and 1.0 % off). Each run's log names the Intel GPU
(`Selected adapter: Intel(R) Graphics, arch=xe-3lpg, vendor=intel, backend=Direct3D 12, adapterType=Integrated GPU`)
and each run held Intel GPU memory (shared: 2235 MiB for Qwen3-ASR, 4012 MiB for Granite Speech). Granite Speech
passes (en, ja and fr+de+es+pt each within 0.5 points of the model's reference), and its 220 answers are the same
text as its RTX row's. Qwen3-ASR does not pass (en WER 14.17, zh CER 52.76, ja CER 70.51): of its 150 answers, 15 are
under 70 % of the CPU row's answer in length and 4 over 150 % (its RTX row: 17 and 6).

### LiteRT-LM 0.19.1's NPU backend ran these two bundles on the CPU

LiteRT-LM 0.19.1's `NPU()` points LiteRT at one folder, `<litert_lm>\vendors\intel_openvino\dispatch\`
(`python/litert_lm/interfaces.py` of LiteRT-LM v0.19.1), which LiteRT-LM passes as both the dispatch library and the
compiler plugin directory (`runtime/util/litert_util.cc`); the installed wheel has one file under `vendors\`, that
folder's `LiteRtDispatch.dll`. A `.litertlm` that packages no `TF_LITE_AUX` model (the auxiliary model LiteRT-LM's
specialized NPU executor reads), as the two litert-community bundles here, goes through LiteRT's generic
compiler-plugin path; with no plugin found, the whole model runs on the CPU through XNNPACK. The NPU-backend runs of
these bundles show it three ways:

- its log: `NPU backend requested, but TF_LITE_AUX is not packaged in the model. Using the generic LiteRT
  compiler-plugin path.`, then `Failed to apply compiler plugins: No compiler plugin found` and
  `Created TensorFlow Lite XNNPACK delegate for CPU.`;
- the engine process's `GPU Process Memory` counter for the NPU's adapter stays at 0 MiB in its `run1.meta.json`
  (the counter does read the NPU: 2558 MiB in s4's OpenVINO GenAI Whisper int8 run);
- every answer is the same text as the CPU backend's: Qwen3-ASR 150 of 150 (s4), Granite Speech 220 of 220
  (`raw/z3b-first-n/` against s4).

`tools/results.py` reads the log and the NPU's memory (`npu_check` in the JSON of each LiteRT-LM NPU cell) and marks
such a run `ran_on: cpu`, with the device "CPU (Intel NPU requested; ...)". Granite Speech's NPU cell keeps its first
pass over all 220 clips (`raw/z3b-first-n/`, untimed); its unit in z3b-s5 was stopped by hand 14 s after its start
(the run would have timed the CPU again; its record is the exit 1 in `raw/z3b-s5/granite/litertlm-npu/`).

LiteRT's Intel OpenVINO backend has a compiler plugin for on-device compilation
([Intel NPU (OpenVino) with LiteRT](https://ai.google.dev/edge/litert/next/intel)), shipped in the
`ai-edge-litert-nightly` wheel (`ai_edge_litert\vendors\intel_openvino\compiler\LiteRtCompilerPlugin.dll`).
`pc/npu_plugin_dir.py` puts the one from `ai-edge-litert-nightly` 2.4.0.dev20261024 (its Intel SDK of the same date
pins `openvino==2026.4.1`, the version in the LiteRT-LM venv) beside LiteRT-LM's own dispatch library, and
`pc/run_asr_litertlm.py --npu-lib-dir` hands that folder to `NPU(litert_dispatch_lib_dir=...)`. One clip per language
(`raw/z3b-npujit/`):

- Qwen3-ASR's language model reached the NPU: `1 compiler plugins were applied successfully: IntelOpenVINO compiler
  plugin (ver 0.1.0)`, `Dispatch: using device 'NPU' from bytecode header`, 2378 MiB on the NPU. The plugin took 1242
  of 1298 ops and 1188 of 1244 ops of the two subgraphs the log lists (29 and 30 partitions); the ops it left, 56 per
  subgraph, are DYNAMIC_UPDATE_SLICE (`op type 151 is not supported`). The load, compilation included, took 59 s. The
  audio encoder runs on the CPU in every LiteRT-LM cell here (`--audio-backend cpu`, or the CPU backend itself).
- Granite Speech did not: 164 of its 165 partitions compiled, the last one failed in the NPU compiler
  (`vclAllocatedExecutableCreate4 result: 0x78000004 - [NPU_VCL] Compiler returned msg: Compilation failed`), so no
  plugin was applied (`0 compiler plugins were applied successfully`) and the model ran on the CPU after a 402 s load.

Qwen3-ASR then ran every clip this way, first untimed under a lock that was not a timing (`raw/z3b-first-q/`) and
then timed (`raw/z3b-s6/`: one block on 11-10 from 13:50 to 14:08, its control head 0.26307 and tail 0.26987 against
s4's known value 0.26228, 0.3 % and 2.9 % off): en WER 4.43 / zh CER 6.50 / ja CER 5.39, which passes,
RTF 0.2672, 2378 MiB on the NPU. 119 of its 150 answers are the same text as the CPU backend's (s4), and none is
under 70 % or over 150 % of that answer's length.

In each of the three plugin runs of Qwen3-ASR the dispatch library wrote a 3,084,095,820-byte weights file to the
Windows temp folder (`litert_ov_bank_<16 hex digits>.bin`, a new random name each run); the files were still there
afterwards (`raw/logs/z3b_pc_cleanup.txt`, the cleanup that removed them). The runner never closes the engine and ends
its process with `os._exit` (`pc/run_asr_litertlm.py`), which skips the destructor that removes the file. That
destructor is `NpuSharedBank::~NpuSharedBank` in `litert/vendors/intel_openvino/dispatch/weight_bank_runtime.cc` at
LiteRT dcb657e8, the commit that LiteRT-LM v0.19.1's `WORKSPACE` pins as `LITERT_REF`. A run that ends without
`os._exit` was not tried.

The commands of 11-10, in order. `pc/plans/z3b-s5.json` and `z3b-s6.json` are the files as they ran: set their
`end_by` (11-10 14:30 and 14:33) to a time ahead first, or the chain starts none of their cells. After the pulls,
`tools/results.py build --runs first-a,first-an,first-b,first-bn,first-a2,s4,z3b-first-g,z3b-first-n,z3b-s5,z3b-first-q,z3b-s6`
rebuilds `results/` from the records.

```bash
export Z3_PC_ROOT='C:\src\audiozoo' Z3_LABEL=Z3b
R='C:\src\audiozoo'
# the clips (step 1), the scripts, and the folder's label
python3 tools/fetch_clips.py --out clips
tools/pc.sh push
tools/pc.sh pushfix clips/s150 clips/s220
printf '%s\n' "[IO.File]::WriteAllText('$R\label.txt', 'Z3b')" | tools/pc.sh ps
# two venvs and three Hub repos
tools/pc.sh launch 'pc\steps.ps1' '-List setup-z3b'
tools/pc.sh launch 'pc\steps.ps1' '-List fetch-z3b'
tools/pc.sh wait "$R\logs\setup-z3b.done,$R\logs\fetch-z3b.done" "$R\logs\setup-z3b.log,$R\logs\fetch-z3b.log" 1
# the probe of the four forms, and the first passes (Intel GPU: 10 clips per language; NPU: every clip of S220)
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\z3b-probe.json -Mode first"
python3 tools/make_plan.py first --sitting z3b-first-g --cells qwen3asr/litertlm-igpu,granite/litertlm-igpu --first-n 10 --timeout-min 30
tools/pc.sh push
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\z3b-first-n.json -Mode first"
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\z3b-first-g.json -Mode first"
for s in z3b-probe z3b-first-g z3b-first-n; do tools/pc.sh pull "runs/$s" raw; done
# the timed sitting (pc/plans/z3b-s5.json: control head and tail against s4's known value 0.26228, one block)
Z3_LOCK_WHAT="timing: sitting z3b-s5 (control head/tail + Intel GPU 2 cells + Intel NPU Granite, one block) until HH:MM" \
  tools/sitting.sh z3b-s5
# the compiler-plugin folders (pc/npu_plugin_dir.py fetches the nightly wheel and checks its sha256), then the probe
# under one lock that is not a timing (its runs hold over 8 GB)
tools/pc.sh push
tools/pc.sh launch 'pc\steps.ps1' '-List npujit-z3b'
tools/pc.sh wait "$R\logs\npujit-z3b.done" "$R\logs\npujit-z3b.log" 1
tools/pc.sh lock take "npu plugin probe (not a timing) until HH:MM"
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\z3b-npujit.json -Mode first"
tools/pc.sh wait "$R\runs\z3b-npujit\chain.done" "$R\runs\z3b-npujit\chain.log" 1
tools/pc.sh pull runs/z3b-npujit raw
tools/pc.sh lock release
# Qwen3-ASR on the NPU through the plugin: every clip under one lock that is not a timing, then its timed sitting
# (pc/plans/z3b-s6.json: the first pass's time as the unit's estimate, control head and tail, one block)
tools/pc.sh lock take "qwen3asr npu first pass (not a timing) until HH:MM"
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\z3b-first-q.json -Mode first"
tools/pc.sh wait "$R\runs\z3b-first-q\chain.done" "$R\runs\z3b-first-q\chain.log" 1
tools/pc.sh pull runs/z3b-first-q raw
tools/pc.sh lock release
tools/pc.sh push
Z3_LOCK_WHAT="timing: sitting z3b-s6 (control head/tail + Intel NPU Qwen3-ASR, one block) until HH:MM" \
  tools/sitting.sh z3b-s6
# what these steps left on the PC (exit 1 when a file or a done line is missing; raw/walk_check_z3b.txt)
printf '%s\n' "& '$R\pc\walk_check.ps1' -Z3b -Sittings z3b-probe,z3b-first-g,z3b-first-n,z3b-s5,z3b-npujit,z3b-first-q,z3b-s6; exit \$LASTEXITCODE" | tools/pc.sh ps
```
