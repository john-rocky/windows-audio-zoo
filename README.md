# Audio models on one Windows PC: ASR and TTS cells across runtimes and devices

Speech models on one Windows laptop, one cell per model x runtime x form x device: the harness that runs them, the
record of every run, and one JSON per cell in [results/](results/) with accuracy against the model's own reference
(WER or CER), real-time factor (RTF), first token, peak memory and the record each number comes from. A table that
puts the runtimes side by side is not committed here: `python3 tools/table.py` writes it from `results/` on your
disk (`RESULTS_AUDIO_WINDOWS.md`). TensorRT-RTX and TensorRT-Edge-LLM are not run here.

| | |
|---|---|
| PC | MSI Stealth 16 AI+ (Core Ultra 9 386H, Intel NPU, Intel GPU, RTX 5080 Laptop at its 95 W cap, 31.5 GB RAM), Windows 11 |
| ASR | Whisper large-v3-turbo, Qwen3-ASR 1.7B, Granite Speech 5.0 2B, Parakeet TDT 0.6B v3 |
| TTS | Kokoro-82M |
| runtimes | OpenVINO GenAI 2026.4.1, ONNX Runtime GenAI 0.19.0, ONNX Runtime 1.31.1 (with the OpenVINO and WebGPU EP plugins), LiteRT-LM 0.19.1 |

## How a cell is measured

- One engine process per cell: load, one untimed warm-up clip, then one pass over the clip set, one clip at a time,
  greedy decoding. Audio is read into memory before the timed loop (a runtime that only takes a file path reads it
  inside the timed span; the JSON says which).
- Clip sets: S150 = FLEURS test, 50 clips each of en_us, cmn_hans_cn and ja_jp (1806 s); S220 = the Granite Speech set
  (FLEURS en / ja / fr / de / es / pt and 20 LibriSpeech clips, 2687 s); T50 = the 50 en_us sentences as TTS input.
- Accuracy: S150 en WER and zh / ja CER with one fixed normalization (tools/fleurs_score.py), rates from summed error
  counts. A cell passes when every language is within 0.5 points of the model's reference (transformers fp32 on CPU)
  and no answer is empty. TTS: every wave is transcribed by one fixed ASR (OpenVINO GenAI, Whisper large-v3-turbo fp16
  on CPU) and scored against the input sentence; a cell passes within 1.0 point of the PyTorch reference's waves.
- RTF = synthesis or transcription wall time / audio seconds, summed over the clips (TTS: over the generated audio).
- Every engine process first takes itself out of Windows' execution-speed power throttling (High QoS:
  `SetProcessInformation(ProcessPowerThrottling)`, the run JSON's `power_throttling`): a process started from ssh or
  WMI without a window is otherwise throttled from about 3 s after its creation, its threads on the efficiency cores.
- The laptop's vendor tool, MSI Center, sets a power mode: its User Scenario (registry
  `HKLM\SOFTWARE\WOW6432Node\MSI\MSI Center\Component\Base Module\User Scenario` `Mode`; on this PC 2 = Balanced,
  1 = Extreme Performance) and with it the Windows power overlay
  (`HKLM\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes` `ActiveOverlayAcPowerScheme`). It changed during
  this walk (11-09 20:12 JST, Balanced to Extreme Performance, a change made for the whole PC). Every run's
  `run<k>.meta.json` (`power`, `desktopLocked`) and every sitting's `env.*.json` record the mode, the overlay and
  whether the desktop's lock screen was up. The table's timed rows come from one sitting, s4, except the LiteRT-LM
  Intel GPU rows (z3b-s5) and the compiler-plugin NPU row (z3b-s6) of the section on 2026-11-10; the evening sitting
  s1 is a record.
- Timed rows run under the PC's shared timing lock, after a quiet gate (CPU, GPU memory and GPU engines idle), with a
  control row at the head and the tail of every sitting (Whisper int8 on CPU, 30 clips): a head over 5 % off the
  known value stops the sitting; a tail over 5 % off marks the sitting's rows unusable.

## Walk from an empty folder

From the Mac (bash), with `ssh winpc` reaching the PC's PowerShell. The PC is shared: `tools/pc.sh` reads the shared
lock state before every connection and refuses during another job's timing. Every untimed step and first-pass run on
the PC is killed when another job's lock appears and runs again from its start once the lock has been free 60 s (a
suspended process would keep its RAM). A step over 8 GB takes the PC's lock once as "(not a timing)", at the slot the
PC's order gives it.

`Z3` is this repo's label on the shared PC (its lock text and notes) and in the scripts' environment variables.
What the walk needs from outside this repo: the shared PC's lock feed on the Mac,
`~/code/standup/tools/winpc/busy_now.sh` (not part of this repo). `tools/pc.sh` reads it before every connection
and refuses without it (`Z3_FORCE=1` skips the read); `tools/sitting.sh` and four scripts in `mtools/`
(`sittings_run.sh`, `retake_after.sh`, `endby_guard.sh`, `wait_free.sh`) wait on it, and without it a timed
sitting waits forever. Step 7 scores with a Python that has openai-whisper 20250625 (`Z3_SCORE_PY`).

```bash
export Z3_PC_ROOT='C:\src\audiozoo'
R='C:\src\audiozoo'
# 1. the two clip sets, rebuilt from their sources and checked clip by clip (tools/fetch_clips.py: numpy, pyarrow,
#    soundfile, huggingface_hub and ffmpeg on the Mac, about 4.3 GB of downloads), then the scripts, the pinned Hub file
#    list and the clips into the empty folder
python3 tools/fetch_clips.py --out clips
tools/pc.sh push
tools/pc.sh pushfix clips/s150 clips/s220
# 2. this repo's own Python 3.12 (uv) and eight venvs (pc/req/*.txt), beside the Hub files (20.25 GiB, every file
#    checked against the Hub's sha256 or git blob hash)
tools/pc.sh launch 'pc\steps.ps1' '-List setup'
tools/pc.sh launch 'pc\steps.ps1' '-List fetch'
tools/pc.sh wait "$R\logs\setup.done,$R\logs\fetch.done" "$R\logs\setup.log,$R\logs\fetch.log" 3
```

```bash
# 3. the references (transformers fp32 CPU: Whisper, Parakeet; PyTorch Kokoro and the write-back of its waves) and the
#    Whisper export for ONNX Runtime GenAI (pc/patch_genai_whisper.py fixes the 0.19.0 builder for a decoder whose
#    layer count differs from the encoder's, then the builder: fp32 CPU, fp16 CUDA)
tools/pc.sh launch 'pc\steps.ps1' '-List refs'
tools/pc.sh launch 'pc\steps.ps1' '-List export-whisper'
tools/pc.sh wait "$R\logs\refs.done,$R\logs\export-whisper.done" "$R\logs\refs.log,$R\logs\export-whisper.log" 2
# 4. optimum-intel's Qwen3-ASR IR (fp16, int8: over 8 GB each, under the PC's lock), and beside it the first pass of
#    the cells on downloaded files (10 clips per language, one run per cell; the NPU cells in a second chain)
python3 tools/make_plan.py first --sitting first-a --group a --devices cpu,igpu,rtx
python3 tools/make_plan.py first --sitting first-an --group a --devices npu && tools/pc.sh push
tools/pc.sh launch 'pc\steps.ps1' '-List export-ov'
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\first-a.json -Mode first"
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\first-an.json -Mode first"
tools/pc.sh wait "$R\logs\export-ov.done,$R\runs\first-a\chain.done,$R\runs\first-an\chain.done" "$R\logs\export-ov.log,$R\runs\first-a\chain.log,$R\runs\first-an\chain.log" 3
# 5. the first pass of the cells on exports
python3 tools/make_plan.py first --sitting first-b --group b --devices cpu,igpu,rtx
python3 tools/make_plan.py first --sitting first-bn --group b --devices npu && tools/pc.sh push
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\first-b.json -Mode first"
tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\pc\plans\first-bn.json -Mode first"
tools/pc.sh wait "$R\runs\first-b\chain.done,$R\runs\first-bn\chain.done" "$R\runs\first-b\chain.log,$R\runs\first-bn\chain.log" 2
# the first passes' records to the Mac (step 6 plans the sittings from them)
for s in first-a first-an first-b first-bn; do tools/pc.sh pull "runs/$s" raw; done
```

```bash
# 6. the timed sittings, every cell whose first pass ended OK: s1 = CPU (its control head sets the known value),
#    s2 = Intel GPU and NPU, s3 = RTX (both with s1's value). A sitting waits until the PC's lock has been free 60 s,
#    takes it and passes the quiet gate; while another job's note waits it gives the lock back for 16.5 min after
#    30 min (then free 60 s, 180 s settle, the gate). A cell that would end past --end-by is not started, nor any cell
#    after it (raw/left_cells.txt); --left runs those in one later sitting with s1's known value, or with its own
#    head's (--own-head, e.g. after the PC's power mode changed); --no-windows makes a sitting one block to --end-by
#    (no window for other jobs' notes) when the PC's order gives it the lock in one piece.
mtools/sittings_run.sh --end-by <ISO time the PC's next booked job starts>
mtools/sittings_run.sh --left s4 [--own-head] [--no-windows] --end-by <ISO time>
# 7. the write-back of every timed Kokoro run (untimed), the pull of the sittings' records and of the references,
#    results/ and the table (Z3_SCORE_PY=<a Python with openai-whisper 20250625>; Z3_SCRUB_NAMES=<account names>:
#    replaced with USER in what was pulled)
mtools/after_sittings.sh s1,s2,s3,s4      # the sittings that ran
# 8. the walk's check on the PC: every file a step leaves, each done line, each sitting's runs against its plan
#    (-Sittings: the sittings that ran; a cell left at an end_by counts as left, not missing)
printf '%s\n' "& '$R\pc\walk_check.ps1' -Sittings s1,s2,s3,s4; exit \$LASTEXITCODE" | tools/pc.sh ps
```

### How the walk of 2026-11-09/10 differed from the steps above

The walk ran in `C:\src\audiozoo` (absent before 10:39 on 11-09) on the shared PC, between other projects' timed
sittings. The scripts were fixed while it ran; the steps above are the fixed form.

- Step 1 pushed two clip folders this project already had on the Mac (made earlier from the same sources);
  `tools/fetch_clips.py`, which rebuilds them and checks every clip, came after the walk (below). Step 1 as written
  ran on 11-10 into an empty folder again (06:49 to 06:54, the same `C:\src\audiozoo`, absent before and deleted
  after): `tools/fetch_clips.py --out` an empty folder PASS (370 clips, 0 bad, 0 warnings), `tools/pc.sh push` and
  `pushfix` exit 0, and on the PC every clip a cell reads (150 FLEURS, the warm-up `en.wav`, 220 S220) had the
  sha256 its pushed index names (371 of 371) and `pc\walk_check.ps1` printed OK on all ten of step 1's lines. The
  rebuilt S150 index differs from the walked one only in `fixtures_root` (the output folder; no script reads it), and
  the S220 index in its `source` line and the 20 LibriSpeech files' sha256 (a new WAV container around the same
  samples).
- Steps 1 and 2 started at 10:40 under the PC's suspend-on-lock helper; other jobs' locks held them suspended for
  most of the morning. `pc/req/genai-cuda.txt` gained the `[cuda,cudnn]` extras after setup had started, so the
  export list re-installs it first. One download (`nvidia_cudnn_cu13`) died on uv's 30 s read timeout after a
  17-minute suspension; `pc/steps.ps1` now sets `UV_HTTP_TIMEOUT=600` and `UV_HTTP_RETRIES=8`, and the step ran again
  in a `redo-setup` list. Setup and fetch ended at 12:15 and 12:16 (every Hub file matched its hash).
- Untimed work stopped for other jobs' locks by suspension until 13:00, when three suspended processes held 8.6 GB
  under another job's timing and were killed by hand; since then `pc/steps.ps1` and the first-pass chains kill their
  process for another job's lock and run it again from its start once the lock has been free 60 s.
- Step 3: the Whisper export for ONNX Runtime GenAI first stopped with `IndexError: index 4 is out of range` (12:18);
  `pc/patch_genai_whisper.py` was written then, and the export passed at 13:22. The Parakeet reference failed once on
  a missing librosa (`pc/req/ref.txt` gained `librosa==1.0.0`) and ran again in `refs-parakeet`.
- Steps 4 and 5: the first-pass chains `first-a` / `first-an` started at 12:17 next to step 3 and were among the
  processes killed at 13:00; they started again at 13:34. One runner fix came mid-walk: the ONNX Runtime cells on
  the OpenVINO EP looked for the device `GPU`, which the plugin names `GPU.0`; `parakeet/ort-fp32-igpu` ran again in
  `first-a2` (the Kokoro Intel GPU cell, run after the fix, failed in the EP's session creation).
- `granite/litertlm-npu` was stopped by hand at 15:12, 22 minutes in, on a misread: the stop counted its 8 progress
  lines as 8 clips, but the runner prints clips 1, 2 and every 10th, and the last line was clip 60, so 60 to 69 of
  its 70 first-pass clips were done (11.6 to 28.5 s per printed clip: the last 10 clips at that pace take under 5
  minutes, and its 30-minute limit was 8 minutes away; corrected 11-10). Its printed lines are kept in `raw/first-an/granite/litertlm-npu/run1.stdout.kept.txt`;
  the cell did not run again in this walk.
- Not in the steps: two diagnostic lists, `diag-qwen3` and `diag-qwen3b` (the qwen-asr package's PyTorch fp32, and the
  exported IR through optimum-intel's own Python class, 10 clips per language each), which placed the Qwen3-ASR
  OpenVINO cells' gap in the IR.
- Step 6 ran from 17:55 on 11-09, after the other projects' timings of the afternoon, with `--end-by 21:05` (another
  project's fixed slot at 21:07). The first s1 head stopped the sitting (three runs 9.6 % apart, the first one slow
  over the whole run; cause unverified); its records are kept as `raw/s1-headfail/` and s1 started again at 18:11
  (head 1.6 % apart, known value 0.26535). Five cells ran; at the first duty window (18:28) another project held the
  lock until 20:13 and a third timed until 20:33. s1 took the lock back at 20:34; after the settle and the gate
  (20:37) its eight other cells would not have ended before 21:05, so they were left and the tail row ran (20:39,
  0.25861 against 0.26535: 2.5 % off, PASS). An older copy of the sittings script, still running, then planned s2 as
  fitting before 21:05 and took the lock at 20:40:52; it was stopped from the Mac and the lock given back by hand at
  20:41:56, before any run.
- MSI Center's power mode went from Balanced to Extreme Performance at 20:12 (set by another project), between s1's
  head and its tail. So s2 and s3 did not run: their cells (15 on the Intel GPU and NPU, 7 on the RTX) and all of
  s1's (the five that ran, the eight left) went to one sitting, s4, which set its own known value under the new mode
  and ran as one block on 11-10, holding the lock from 00:25 to 04:50:
  `mtools/sittings_run.sh --left s4 --own-head --no-windows --after <the project before it> --after-quiet 180
  --end-by 2026-11-10T09:00:00+09:00` (started three times while it only read the shared lock feed, as the night's
  order moved). All 35 cells ran, none was left at the end_by; head 0.26228 (three runs 0.7 % apart), tail 0.26548
  (1.2 % off: PASS); every run's record names the same power mode. The table's timed rows are s4's; s1's records
  stay in `raw/s1/`.
- Tools changed while step 6 ran (the steps above are the changed form): the plan push waits out other projects'
  timings; the lock's text says until end_by at most; cells run shortest first; no duty window before the tail row;
  no take-back that would carry the tail row past end_by (in the chain from s4 on; for s1 a Mac-side guard,
  `mtools/endby_guard.sh`, stood in); `--left` pushes pc/ with a parse check of every .ps1; a presence note not
  rewritten for 120 min no longer opens a duty window; `--own-head` and `--no-windows` (for s4).
- Step 7 ran on 11-10 from 05:02, after the next project's slot in the PC's order, one step per Kokoro cell (seven
  cells in one step could have passed a step's 60-minute limit). Another project's lock killed the first step twice
  (05:05–05:11, 05:12–05:14); it ran again from its start each time, and the list ended OK at 05:49. The Mac side of
  `mtools/after_sittings.sh` was stopped while it waited, so that its scoring would not run inside the Mac's own
  measurement window for another project; its later half ran from a copy: the pulls at 05:50, the scoring and the
  table at 05:57, when that window had closed.
- Step 8 ran at 04:56 on 11-10 with `-Sittings first-a2,s1,s4` (the sittings that ran): 0 missing
  (`raw/walk_check.txt`). It counts missing files, done lines and runs; a done line that says FAIL is printed, not
  counted: the setup and refs lists end FAIL (pip-ort-cuda, ref-parakeet: the two steps above), and the lists that ran
  those steps again, redo-setup and refs-parakeet, ended OK (`raw/logs/`).
- After the walk (11-10): `tools/fetch_clips.py` rebuilt both clip folders on the Mac into an empty folder (05:57 to
  06:03): PASS, every S150 and S220 clip and the warm-up file equal to the measured ones (FLEURS clips and the
  warm-up file byte for byte, the LibriSpeech clips sample for sample), 0 warnings. Run again with one FLEURS clip's
  sha256 in the index and one LibriSpeech clip's sample hash changed, it named those two clips BAD and exited 1.
  Two failure paths of the PC scripts were fixed and run on the PC (05:55–05:56): `pc/steps.ps1` given a list name
  with no list file ran 0 steps and wrote OK (now FAIL, exit 1), and `pc/launch.ps1` judged a start by any process
  with the script's file name (other projects run scripts of that name; a list that ended within 20 s read as
  FAILED); it now reads the process it started by its pid. A missing list, a list ending within 20 s and a list
  running past 20 s gave FAILED, STARTED and STARTED.
- Found after the walk: every Intel GPU run of LiteRT-LM (Qwen3-ASR, Granite Speech; first pass and s4) took the RTX.
  `pc/run_asr_litertlm.py --gpu-pref 1` wrote the per-app GPU preference under the interpreter's path as
  `os.path.realpath` gives it (uv's minor-version junction `cpython-3.12-windows-x86_64-none` resolved to
  `cpython-3.12.16-windows-x86_64-none`), while the process runs by the junction's path; each run's log says
  `Selected adapter: NVIDIA GeForce RTX 5080 Laptop GPU`. `tools/results.py` marked those cells WRONG DEVICE until the
  runs of 11-10 replaced them (the next section); the RTX runs' records stay in `raw/first-a/` and `raw/s4/`.

## The LiteRT-LM cells on the Intel GPU and the NPU, again (2026-11-10)

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
Windows temp folder (`litert_ov_bank_<hash>.bin`, a new name each run); the files were still there afterwards
(`raw/logs/z3b_pc_cleanup.txt`, the cleanup that removed them).

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

## Licences

The models keep their own licences (Hub cards, read on 2026-11-09): see [LICENSES.md](LICENSES.md).
