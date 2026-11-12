# Run everything from an empty folder

A cell is one combination of model, runtime, form and device. A sitting is one timed session on the PC. A walk is one run of every step from an empty folder.

The comments in `tools/`, `mtools/` and `pc/` call these the README's steps, where they were first written.

From the Mac (bash), with `ssh winpc` reaching the PC's PowerShell. The PC is shared: `tools/pc.sh` reads the shared
lock state before every connection and refuses during another job's timing. Every untimed step and first-pass run on
the PC is killed when another job's lock appears and runs again from its start once the lock has been free 60 s (a
suspended process would keep its RAM). A step over 8 GB takes the PC's lock once as "(not a timing)", at the slot the
PC's order gives it.

`Z3` is this repo's label on the shared PC (its lock text and notes) and in the scripts' environment variables.
What the walk needs from outside this repo: a command on the Mac that prints the shared PC's lock feed, named in
`PC_FEED_CMD` (its output is described in `tools/pc.sh`'s header; this repo has no such command). `tools/pc.sh` runs
it before every connection and refuses without it (`Z3_FORCE=1` skips the read); `tools/sitting.sh` and four
scripts in `mtools/` (`sittings_run.sh`, `retake_after.sh`, `endby_guard.sh`, `wait_free.sh`) wait on it and stop
at their start without it. Step 7 scores with a Python that has openai-whisper 20250625 (`Z3_SCORE_PY`).

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

Step 3 patches the installed copy of ONNX Runtime GenAI 0.19.0's model builder with `pc/patch_genai_whisper.py`. The
lines the script edits are unchanged on the builder's main branch at 75fa2250c9 (2026-11-12). A fix for the builder
itself is proposed in [microsoft/onnxruntime-genai#2940](https://github.com/microsoft/onnxruntime-genai/pull/2940), a
pull request, open and not merged on 2026-11-12.

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

### Distil-Whisper distil-large-v3 (2026-11-12)

Distil-Whisper distil-large-v3 was added on 11-12 in a folder of its own on the shared PC (`C:\src\distil` below), with
a label of its own in its `label.txt` (the lock text and notes of these steps start with it). It is an English model:
its reference and its cells run on the 50 en_us clips of S150 (`--langs en_us`).
From the Mac, with `tools/pc.sh`'s PC folder and label variables (named in its header) set to that folder and that label:

```bash
R='C:\src\distil'
bash tools/pc.sh push                                              # pc/ and shelf/
bash tools/pc.sh pushfix clips/s150 clips/s220                     # the clips as in the walk above (these steps read S150's 50 en_us clips)
bash tools/pc.sh launch 'pc\steps.ps1' '-List setup-distil'        # Python 3.12, four venvs (ovg, genai, genai-cuda, ref), the S150 clips checked
bash tools/pc.sh launch 'pc\steps.ps1' '-List fetch-distil'        # distil-whisper/distil-large-v3 at 8031d2e6 and the control row's IR
bash tools/pc.sh launch 'pc\steps.ps1' '-List export-distil'       # pc/patch_genai_whisper.py, then the builder: fp32 CPU, fp16 CUDA
bash tools/pc.sh launch 'pc\steps.ps1' '-List refs-distil'         # transformers fp32 on the CPU, the 50 en_us clips
bash tools/pc.sh launch 'pc\steps.ps1' '-List export-distil-int4'  # int4 CPU, int4 CUDA
bash tools/pc.sh launch 'pc\steps.ps1' '-List export-distil-levers'   # int8 CPU, int8 CUDA, fp16 WebGPU
bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-first.json -Mode first"   # the first pass: 7 cells, 50 clips each
python3 tools/make_plan.py first --sitting distil-lev --first-n 0 --group dl                     # the speed levers: 13 cells then (group dl has 33 now: run the committed pc/plans/distil-lev.json)
bash tools/pc.sh push && bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev.json -Mode first"
python3 tools/make_plan.py first --sitting distil-lev2 --cells distil-large-v3/genai-int4_sdpa1-rtx,distil-large-v3/genai-int8_sdpa1-rtx --first-n 0 --timeout-min 15
bash tools/pc.sh push && bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev2.json -Mode first"
bash tools/pc.sh launch 'pc\steps.ps1' '-List export-distil-fp32cuda'    # fp32 CUDA
bash tools/pc.sh launch 'pc\steps.ps1' '-List folders-distil-openvino'   # five folders whose genai_config.json names the OpenVINO EP
bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev3.json -Mode first"   # fp32 CUDA and the five folders
python3 tools/make_plan.py timed --sitting distil-s1d --known 0.26692 --control-est 104 --no-windows --end-by 2026-11-12T21:14:00+09:00 --est-from distil-s1c,distil-lev3,distil-lev,distil-lev2,distil-first --cells distil-large-v3/genai-fp16_sdpa1-rtx,distil-large-v3/genai-fp16-rtx,distil-large-v3/genai-fp32_openvino-igpu,distil-large-v3/genai-int4_openvino-igpu,distil-large-v3/genai-int8_openvino-cpu,distil-large-v3/genai-fp32-cpu
bash tools/pc.sh push && bash tools/sitting.sh distil-s1d
bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev4.json -Mode first"   # the encoder on the NPU
bash tools/pc.sh launch 'pc\steps.ps1' '-List folders-distil-npu'        # the NPU folders, the input shapes given
bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev5.json -Mode first"
bash tools/pc.sh launch 'pc\steps.ps1' '-List folders-distil-npu2'       # int8 on the NPU, the decoder on the Intel GPU, the compile cache
bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev6.json -Mode first"
python3 tools/make_plan.py timed --sitting distil-s1e --known 0.26692 --control-est 104 --no-windows --end-by 2026-11-12T22:26:00+09:00 --est-from distil-lev6,distil-lev5 --cells distil-large-v3/genai-int8_encnpustatic_ovcpu-npu
bash tools/pc.sh push && bash tools/sitting.sh distil-s1e
bash tools/pc.sh launch 'pc\steps.ps1' '-List folders-distil-sharebuf'   # the fp16 CUDA export with the KV buffer shared
bash tools/pc.sh launch 'pc\steps.ps1' '-List folders-distil-npu3'       # the int8 encoder on the NPU, the decoder on the Intel GPU
python3 tools/make_plan.py first --sitting distil-lev7 --first-n 0 --cells distil-large-v3/genai-fp16_sharebuf-rtx,distil-large-v3/genai-fp16_sharebufcg-rtx,distil-large-v3/genai-fp16_sharebufsdpa1-rtx,distil-large-v3/genai-int8_encnpustatic_ovgpu-npu
bash tools/pc.sh push && bash tools/pc.sh launch 'pc\chain.ps1' "-Plan $R\\pc\\plans\\distil-lev7.json -Mode first"
python3 tools/results.py build --runs distil-first,distil-lev,distil-lev2,distil-lev3,distil-lev4,distil-lev5,distil-lev6,distil-s1d,distil-s1e,distil-lev7
```

The plans `distil-lev3` to `distil-lev6` were written by hand; `distil-first` (group d) and `distil-lev` (group dl, 13 cells
at the time) by `tools/make_plan.py first`; the others as above. All are in `pc/plans/`.

- `setup-distil` and `fetch-distil` ran side by side; each later list started after the one before it had ended
  (`logs\<list>.done`).
- Without `pc/patch_genai_whisper.py`, the 0.19.0 builder stops on this model with `IndexError: index 2 is out of
  range` (32 encoder layers, 2 decoder layers).
- First pass: the NPU cell stopped in the NPU compiler (`Reshape node 'Reshape_130': expected exactly 1 dynamic output
  bound dimension, got 2`), as Whisper large-v3-turbo's NPU cell did. The encoder compiles for the NPU only with its
  input shape fixed, the OpenVINO EP's `reshape_input` `audio_features[1,128,3000]` in the folder's
  `genai_config.json`; without it the process ended with an access violation (exit 0xC0000005, 12 s in) after the
  compiler's message `Upper bounds are not specified for node '/model/preprocessing/Conv_1/WithoutBiases'`
  (`distil-lev4`). The fp32 export's decoder does not compile for the NPU with its lengths bounded (`Can't compute total
  shape size on dynamic shape`, `distil-lev5`; the int8 decoder was not tried there), so the NPU rows run the decoder on
  the OpenVINO EP's CPU or Intel GPU device (GenAI 0.19.0 refuses a model with two providers, `genai-fp32_encnpu-npu`
  in `distil-lev`).
- `genai-fp32_encnpucachea_ovcpu-npu` and `genai-fp32_encnpucacheb_ovcpu-npu` run one folder twice, in that order
  (`distil-lev6`): its encoder's entry sets the OpenVINO EP's `cache_dir`, so the first compiles the encoder and writes
  it there and the second loads it from there (`docs/measuring.md`, `load_s`).
- The `genai-*_openvino-*` cells run folders made by `pc/genai_providers_folder.py` (hard links to an export, and a
  `genai_config.json` that names the OpenVINO EP for the encoder and the decoder) with `pc/run_asr_genai.py --ep
  config`; the folder of `genai-int8_openvino-cpu` holds the files of the Hub repo
  mlboydaisuke/distil-large-v3-onnxruntime-genai-int8-openvino.
- Three timed sittings did not give rows: `distil-s1` and `distil-s1b` stopped at their control head (head spread 7.4 %
  and 9.9 %), and the control tail of `distil-s1c` was 7.4 % slower than its head (`docs/measuring.md`); their records
  stay in `raw/`. `tools/results.py` takes no timed run of a sitting whose control tail did not pass. The timed rows
  are from `distil-s1d`, whose control head was 1.1 % and its tail 1.8 % off the known value 0.26692 (the median of
  `distil-s1c`'s three head runs), and the Intel NPU row from `distil-s1e` (head 1.5 %, tail 2.5 %).
- The fp32 CPU export with `--ep cuda` (`genai-fp32-rtx`) ended with an access violation; the export made for CUDA
  (`-p fp32 -e cuda`, `export-distil-fp32cuda`) gave all 50 answers there (`genai-fp32_cuda-rtx`), and its process
  then ended with the same exit code, 0xC0000005, after writing its results.
