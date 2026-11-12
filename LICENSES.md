# Licences of what the cells use

Read on 2026-11-09: the model licences from the Hub API (`cardData.license` of each pinned repo, `shelf/hub_manifest.json`);
the package licences from PyPI's metadata of the pinned version (`license` / `license_expression` / classifiers) or,
where PyPI states none, from the wheel's own files. This repo ships none of these files: it downloads them at the
pinned revisions.

## Models (Hub)

| repo | revision | licence | used for |
|---|---|---|---|
| openai/whisper-large-v3-turbo | 41f01f3f | mit | ORT GenAI builder export, transformers reference |
| OpenVINO/whisper-large-v3-turbo-fp16-ov | 0250c28d | mit | OpenVINO GenAI cells, TTS write-back ASR |
| OpenVINO/whisper-large-v3-turbo-int8-ov | b568445d | mit | OpenVINO GenAI cells, control row |
| OpenVINO/whisper-large-v3-turbo-int4-ov | ae50b4d9 | mit | OpenVINO GenAI cells |
| Qwen/Qwen3-ASR-1.7B | 7278e1e7 | apache-2.0 | optimum-intel export to OpenVINO IR |
| litert-community/Qwen3-ASR-1.7B | 2ad21a0b | apache-2.0 | LiteRT-LM cells |
| litert-community/granite-speech-5.0-2b | c344fa2a | apache-2.0 | LiteRT-LM cells |
| istupakov/parakeet-tdt-0.6b-v3-onnx | 8f23f0c0 | cc-by-4.0 | ONNX Runtime cells |
| nvidia/parakeet-tdt-0.6b-v3 | 541d1f99 | cc-by-4.0 | transformers reference |
| OpenVINO/Kokoro-82M-fp16-ov | 787375b6 | apache-2.0 | OpenVINO GenAI cells |
| OpenVINO/Kokoro-82M-int8-ov | 9c035d0b | apache-2.0 | OpenVINO GenAI cells |
| onnx-community/Kokoro-82M-v1.0-ONNX | 1939ad2a | apache-2.0 | ONNX Runtime cells |
| hexgrad/Kokoro-82M | f3ff3571 | apache-2.0 | PyTorch reference |
| distil-whisper/distil-large-v3 | 8031d2e6 | mit (read 2026-11-12) | ORT GenAI builder export, transformers reference |

## Packages (PyPI, pinned in pc/req/)

| package | version | licence (as stated) |
|---|---|---|
| openvino-genai | 2026.4.1.0 | Apache License 2.0 |
| openvino | 2026.4.1 | Apache-2.0 |
| onnxruntime / onnxruntime-gpu | 1.31.1 | MIT License |
| onnxruntime-genai / onnxruntime-genai-cuda | 0.19.0 | MIT License |
| onnxruntime-ep-openvino | 1.8.1 | MIT |
| onnxruntime-ep-webgpu | 0.5.0 | MIT |
| litert-lm-api | 0.19.1 | Apache-2.0 |
| onnx-asr | 0.12.0 | MIT |
| kokoro-onnx | 0.6.1 | MIT (the wheel's LICENSE; PyPI states none) |
| kokoro / misaki | 0.9.4 | Apache License 2.0 |
| espeakng-loader | 0.2.4 | not stated on PyPI or in the wheel's metadata (unverified); the wheel carries `espeak-ng.dll` |
| phonemizer | 3.6.0 | GPL-3.0-or-later |
| optimum-intel | 2.2.0 | Apache |
| qwen-asr | 0.0.6 | Apache-2.0 |
| transformers | 5.22.2 / 4.57.6 | Apache 2.0 |
| torch | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| nvidia-cuda-runtime and the CUDA / cuDNN wheels of onnxruntime-gpu's extras | 13.5.37 / — | NVIDIA's own licence (unverified) |

## Data (clips and their transcriptions)

The repo ships no audio: `tools/fetch_clips.py` downloads it at the pinned revisions. The clip lists in `fixtures/`
carry the datasets' transcriptions, and the run records carry the models' answers on those clips. Read on 2026-11-10
(the Hub API's `cardData.license`; LibriSpeech's own page for the dataset the dummy copy is cut from).

| source | revision | licence | used for |
|---|---|---|---|
| google/fleurs (test: en_us, cmn_hans_cn, ja_jp, fr_fr, de_de, es_419, pt_br) | 70bb2e84 | cc-by-4.0 | S150 and S220 clips, references, T50 sentences |
| hf-internal-testing/librispeech_asr_dummy (LibriSpeech dev-clean) | 5be91486 | none stated on the Hub; LibriSpeech is CC BY 4.0 (openslr.org/12) | S220's 20 LibriSpeech clips and references |
| FunAudioLLM/Fun-ASR-Nano-2512, `example/en.mp3` | 272c57b8 | apache-2.0 | the untimed warm-up clip (not scored) |
