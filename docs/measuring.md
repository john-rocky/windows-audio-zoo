# How the results are measured

A cell is one combination of model, runtime, form and device. A sitting is one timed session on the PC. A walk is one run of every step from an empty folder.

Speech models on one Windows laptop, one cell per model x runtime x form x device: the harness that runs them, the
record of every run, and one JSON per cell in [results/](../results/) with accuracy against the model's own reference
(WER or CER), real-time factor (RTF), first token, peak memory and the record each number comes from. A table that
puts the runtimes side by side is not committed here: `python3 tools/table.py` writes it from `results/` on your
disk (`RESULTS_AUDIO_WINDOWS.md`). TensorRT-RTX and TensorRT-Edge-LLM are not run here.

| | |
|---|---|
| PC | MSI Stealth 16 AI+ (Core Ultra 9 386H, Intel NPU, Intel GPU, RTX 5080 Laptop at its 95 W cap, 31.5 GB RAM), Windows 11 |
| ASR | Whisper large-v3-turbo, Distil-Whisper distil-large-v3 (English only: the 50 en_us clips of S150), Qwen3-ASR 1.7B, Granite Speech 5.0 2B, Parakeet TDT 0.6B v3 |
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
- From 2026-11-12, `pc/chain.ps1` rests 20 s (`-BetweenRunsSeconds`) before every timed run but the sitting's first,
  the first run of a cell too, and 150 s (`-TailIdleSeconds`) before the control tail. Before, a cell's first run
  started about 6 s after the cell before it ended, every other run about 27 s after the run before; two sittings of
  11-12 (`distil-s1`, `distil-s1b`) stopped at their control head that way: the first head run, the only one started
  that soon, was the slowest of the three (head spread 7.4 % and 9.9 %). In `distil-s1c` the control tail, started 27 s after an OpenVINO EP
  CPU int8 cell ended, was 7.4 % slower than the head, and that sitting's rows are not used.
- ONNX Runtime GenAI cells set the execution provider in code (`pc/run_asr_genai.py --ep`). In the Whisper cells on
  the Intel GPU (`genai-fp32-igpu`) the Intel GPU held 1,042 MiB (Whisper large-v3-turbo) and 695 MiB
  (distil-large-v3), while distil-large-v3's fp32 encoder weights alone are 2,574,264,320 bytes: those rows ran the
  decoder on the Intel GPU and the encoder on the CPU (the export's `genai_config.json` names no provider for the
  encoder), and the table says "(decoder)". `--encoder-ep same` (from 2026-11-12) writes the decoder's provider for
  the encoder too, into a copy of `genai_config.json`; distil-large-v3's `genai-fp32_whole-igpu` row ran that way
  (3,712 MiB on the Intel GPU).
- `--ep config` (from 2026-11-12) sets no provider in code: the folder's `genai_config.json` names it for the encoder
  and the decoder, the form of a Hub repo that names its provider. `pc/genai_providers_folder.py` writes such a folder
  (hard links to an export's files, a copy of its `genai_config.json` with the two `provider_options` lines changed).
  distil-large-v3's `*_openvino-*` rows ran that way. The same form set in code and in the file gave the same answer
  on all 50 clips twice (CPU int8: `genai-int8_ovcpu-cpu` and `genai-int8_openvino-cpu`; Intel GPU fp32:
  `genai-fp32_whole-igpu` and `genai-fp32_openvino-igpu`, first passes of 11-12).
- distil-large-v3's timed rows are from the sitting `distil-s1d` (11-12, 20:45 to 21:07): the control head 1.1 % and the
  tail 1.8 % off the known value 0.26692; its Intel NPU row is from `distil-s1e` (11-12, 21:58 to 22:15): the head 1.5 %
  and the tail 2.5 % off the same value. The sittings `distil-s1`, `distil-s1b` and `distil-s1c` gave no rows (above).
- A GenAI cell's `load_s` is the time of `og.Model` and its processor in the engine process. On the Intel NPU it
  includes the encoder's compile: 216 to 219 s for distil-large-v3's fp32 encoder (`distil-lev5`, `distil-lev6`), 277 and
  279 s for its int8 encoder (`distil-lev6`, `distil-s1e`). With the OpenVINO EP's `cache_dir` in the encoder's entry, the
  first process wrote the compiled encoder there and a second process loaded the fp32 model in 3.3 s
  (`genai-fp32_encnpucachea_ovcpu-npu`, `genai-fp32_encnpucacheb_ovcpu-npu`).
