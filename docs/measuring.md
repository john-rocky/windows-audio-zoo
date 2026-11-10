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
