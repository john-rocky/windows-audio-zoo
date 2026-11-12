#!/usr/bin/env python3
"""The PC's step lists (pc/steps/<name>.json for pc/steps.ps1) and the chain plans (pc/plans/<sitting>.json for
pc/chain.ps1), from the cell table below (PLAN.md sections 2 and 3).

    python3 tools/make_plan.py steps                                  # setup, fetch, export-whisper, export-ov, refs
    python3 tools/make_plan.py first --sitting first-a --group a --devices cpu,igpu,rtx   # first pass, 10 clips per language
    python3 tools/make_plan.py first --sitting first-an --group a --devices npu          # the NPU cells beside it
    python3 tools/make_plan.py timed --sitting s1 --cells <id,id,...> --est-from first-a,first-b [--known <rtf>] [--end-by <iso>] [--control-est <s>] [--warm-runs 2]
    python3 tools/make_plan.py list                                   # the cell ids
    python3 tools/make_plan.py redo --from setup --names pip-genai,versions-genai   # pc/steps/redo-setup.json

A cell id is <model>/<runtime>-<form>-<device>. A probe is a cell outside its runtime's documented devices: first pass
only, until it runs.
"""
import argparse
import json
import pathlib
import sys

K = pathlib.Path(__file__).resolve().parents[1]
M = r"{ROOT}\models"
X = r"{ROOT}\exports"


def cell(cid, venv, script, args, cset, device, ram, probe=False, group="a", timeout=20):
    return {"cell": cid, "venv": venv, "script": script, "args": args, "set": cset, "device": device,
            "ram_need_gb": ram, "probe": probe, "group": group, "timeout_min": timeout,
            "engines_gate": device == "igpu"}


OVDEV = {"cpu": "CPU", "igpu": "GPU.0", "rtx": "GPU.1", "npu": "NPU"}


def cells():
    c = []
    # Whisper large-v3-turbo
    for form, repo in (("fp16", "fp16"), ("int8", "int8"), ("int4", "int4")):
        for dev in ("cpu", "igpu", "npu", "rtx"):
            if dev == "rtx" and form != "int8":
                continue
            c.append(cell(f"whisper/ovg-{form}-{dev}", "ovg", "run_asr_ovg.py",
                          ["--model-dir", rf"{M}\OpenVINO__whisper-large-v3-turbo-{repo}-ov", "--family", "whisper",
                           "--device", OVDEV[dev], "--cache-dir", r"{RUN}\cache"], "S150", dev, 4,
                          probe=(dev == "rtx"), timeout=30 if dev == "npu" else 20))
    # group a: the cells on downloads and on the Whisper GenAI export (README walk step 3, before the first pass)
    c.append(cell("whisper/genai-fp32-cpu", "genai", "run_asr_genai.py",
                  ["--model-dir", rf"{X}\whisper-genai-fp32-cpu", "--ep", "cpu"], "S150", "cpu", 6))
    c.append(cell("whisper/genai-fp32-igpu", "genai", "run_asr_genai.py",
                  ["--model-dir", rf"{X}\whisper-genai-fp32-cpu", "--ep", "openvino", "--ov-device", "GPU"], "S150", "igpu", 6))
    c.append(cell("whisper/genai-fp32-npu", "genai", "run_asr_genai.py",
                  ["--model-dir", rf"{X}\whisper-genai-fp32-cpu", "--ep", "openvino", "--ov-device", "NPU"], "S150", "npu", 6,
                  probe=True, timeout=30))
    c.append(cell("whisper/genai-fp16-rtx", "genai-cuda", "run_asr_genai.py",
                  ["--model-dir", rf"{X}\whisper-genai-fp16-cuda", "--ep", "cuda"], "S150", "rtx", 6))
    # Distil-Whisper distil-large-v3 (an English model: the 50 en_us clips of S150), group d: the ORT GenAI exports of
    # the same builder after pc/patch_genai_whisper.py (32 encoder layers, 2 decoder layers). The fp32 export on CUDA
    # and the int4 exports are the speed levers' rows (precision, quant)
    dx = rf"{X}\distil-large-v3-genai"
    for cid, venv, a, dev, probe in (
            ("distil-large-v3/genai-fp32-cpu", "genai", [rf"{dx}-fp32-cpu", "--ep", "cpu"], "cpu", False),
            ("distil-large-v3/genai-fp32-igpu", "genai", [rf"{dx}-fp32-cpu", "--ep", "openvino", "--ov-device", "GPU"], "igpu", False),
            ("distil-large-v3/genai-fp32-npu", "genai", [rf"{dx}-fp32-cpu", "--ep", "openvino", "--ov-device", "NPU"], "npu", True),
            ("distil-large-v3/genai-fp16-rtx", "genai-cuda", [rf"{dx}-fp16-cuda", "--ep", "cuda"], "rtx", False),
            ("distil-large-v3/genai-fp32-rtx", "genai-cuda", [rf"{dx}-fp32-cpu", "--ep", "cuda"], "rtx", False),
            ("distil-large-v3/genai-int4-cpu", "genai", [rf"{dx}-int4-cpu", "--ep", "cpu"], "cpu", False),
            ("distil-large-v3/genai-int4-rtx", "genai-cuda", [rf"{dx}-int4-cuda", "--ep", "cuda"], "rtx", False)):
        c.append(cell(cid, venv, "run_asr_genai.py", ["--model-dir"] + a + ["--langs", "en_us"], "S150", dev, 6,
                      probe=probe, group="d", timeout=30 if dev == "npu" else 20))
    # distil-large-v3's speed levers (11-12), group dl: int8 weights (quant), CUDA graph and the attention kernel on the
    # decoder (state, ep option), a 132-token KV buffer (shape: 4 + 128 new tokens against 4 + 440), the encoder on the
    # same device as the decoder (--encoder-ep: "whole" = the Intel GPU's OpenVINO EP for both, "webgpu" = the WebGPU
    # plugin EP for both on adapter 0 = the RTX or 1 = the Intel GPU, "ovcpu" = the OpenVINO EP on the CPU for both)
    # and the encoder alone on the NPU (decoder on the CPU). A variant is named in the form part
    # (<runtime>-<form>-<device>: tools/results.py reads three parts)
    cg = ["--ep-option", "enable_cuda_graph=1"]
    for cid, venv, a, dev, probe in (
            ("distil-large-v3/genai-int8-cpu", "genai", [rf"{dx}-int8-cpu", "--ep", "cpu"], "cpu", False),
            ("distil-large-v3/genai-int8-rtx", "genai-cuda", [rf"{dx}-int8-cuda", "--ep", "cuda"], "rtx", False),
            ("distil-large-v3/genai-fp16_cudagraph-rtx", "genai-cuda", [rf"{dx}-fp16-cuda", "--ep", "cuda"] + cg, "rtx", False),
            ("distil-large-v3/genai-int8_cudagraph-rtx", "genai-cuda", [rf"{dx}-int8-cuda", "--ep", "cuda"] + cg, "rtx", False),
            ("distil-large-v3/genai-int4_cudagraph-rtx", "genai-cuda", [rf"{dx}-int4-cuda", "--ep", "cuda"] + cg, "rtx", False),
            ("distil-large-v3/genai-fp16_sdpa1-rtx", "genai-cuda",
             [rf"{dx}-fp16-cuda", "--ep", "cuda", "--ep-option", "sdpa_kernel=1"], "rtx", False),
            ("distil-large-v3/genai-int4_sdpa1-rtx", "genai-cuda",
             [rf"{dx}-int4-cuda", "--ep", "cuda", "--ep-option", "sdpa_kernel=1"], "rtx", False),
            ("distil-large-v3/genai-int8_sdpa1-rtx", "genai-cuda",
             [rf"{dx}-int8-cuda", "--ep", "cuda", "--ep-option", "sdpa_kernel=1"], "rtx", False),
            ("distil-large-v3/genai-fp16_kv132-rtx", "genai-cuda", [rf"{dx}-fp16-cuda", "--ep", "cuda", "--max-new", "128"], "rtx", False),
            ("distil-large-v3/genai-fp32_cuda-rtx", "genai-cuda", [rf"{dx}-fp32-cuda", "--ep", "cuda"], "rtx", False),
            ("distil-large-v3/genai-fp32_whole-igpu", "genai",
             [rf"{dx}-fp32-cpu", "--ep", "openvino", "--ov-device", "GPU", "--encoder-ep", "same"], "igpu", False),
            ("distil-large-v3/genai-fp32_ovcpu-cpu", "genai",
             [rf"{dx}-fp32-cpu", "--ep", "openvino", "--ov-device", "CPU", "--encoder-ep", "same"], "cpu", False),
            ("distil-large-v3/genai-int8_ovcpu-cpu", "genai",
             [rf"{dx}-int8-cpu", "--ep", "openvino", "--ov-device", "CPU", "--encoder-ep", "same"], "cpu", False),
            # the OpenVINO EP named for the encoder and the decoder in the folder's own genai_config.json (11-12:
            # pc/genai_providers_folder.py, run with --ep config, no provider set in code): int8 and int4 on the CPU,
            # int8, int4 and fp32 on the Intel GPU (quant)
            ("distil-large-v3/genai-int8_openvino-cpu", "genai", [rf"{dx}-int8-openvino-cpu", "--ep", "config"], "cpu", False),
            ("distil-large-v3/genai-int4_openvino-cpu", "genai", [rf"{dx}-int4-openvino-cpu", "--ep", "config"], "cpu", False),
            ("distil-large-v3/genai-int8_openvino-igpu", "genai", [rf"{dx}-int8-openvino-gpu", "--ep", "config"], "igpu", False),
            ("distil-large-v3/genai-int4_openvino-igpu", "genai", [rf"{dx}-int4-openvino-gpu", "--ep", "config"], "igpu", False),
            ("distil-large-v3/genai-fp32_openvino-igpu", "genai", [rf"{dx}-fp32-openvino-gpu", "--ep", "config"], "igpu", False),
            ("distil-large-v3/genai-fp16_webgpu-rtx", "genai",
             [rf"{dx}-fp16-webgpu", "--ep", "webgpu", "--ep-option", "adapterIndex=0", "--encoder-ep", "same"], "rtx", True),
            ("distil-large-v3/genai-fp16_webgpu-igpu", "genai",
             [rf"{dx}-fp16-webgpu", "--ep", "webgpu", "--ep-option", "adapterIndex=1", "--encoder-ep", "same"], "igpu", True),
            ("distil-large-v3/genai-fp32_encnpu-npu", "genai",
             [rf"{dx}-fp32-cpu", "--ep", "cpu", "--encoder-ep", "openvino-npu"], "npu", True),
            # GenAI 0.19.0 refused the one above ("Running a model with multiple providers is not supported.
            # Encountered OpenVINO and CPU"): the same OpenVINO EP for both, the encoder on the NPU, the decoder on
            # the CPU
            ("distil-large-v3/genai-fp32_encnpu_ovcpu-npu", "genai",
             [rf"{dx}-fp32-cpu", "--ep", "openvino", "--ov-device", "CPU", "--encoder-ep", "openvino-npu"], "npu", True),
            # that one ended with an access violation (0xC0000005, 12 s in) after the NPU compiler's message on its
            # unbounded batch_size (distil-lev4: "Upper bounds are not specified for node
            # '/model/preprocessing/Conv_1/WithoutBiases' ... bounds are '[9223372036854775807, 128, 3000]'"): the OpenVINO EP's reshape_input fixes the encoder's audio_features to [1,128,3000] in the
            # folder's genai_config.json (pc/genai_providers_folder.py --encoder-option; steps folders-distil-npu and
            # folders-distil-npu2). The decoder on the NPU with batch 1 and its lengths bounded, and both parts on the
            # NPU, stop in the NPU compiler (distil-lev5). shape: the static encoder input; quant: the int8 export's
            # encoder on the NPU; ep: the decoder on the Intel GPU; state: the compiled encoder cached (cache_dir), the
            # same folder run twice (the first compiles, the second loads)
            ("distil-large-v3/genai-fp32_encnpustatic_ovcpu-npu", "genai", [rf"{dx}-fp32-openvino-encnpu", "--ep", "config"], "npu", False),
            ("distil-large-v3/genai-fp32_decnpubound_ovcpu-npu", "genai", [rf"{dx}-fp32-openvino-decnpu", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-fp32_openvino-npu", "genai", [rf"{dx}-fp32-openvino-npu", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-int8_encnpustatic_ovcpu-npu", "genai", [rf"{dx}-int8-openvino-encnpu", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-fp32_encnpustatic_ovgpu-npu", "genai", [rf"{dx}-fp32-openvino-encnpu-gpu", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-fp32_encnpucachea_ovcpu-npu", "genai", [rf"{dx}-fp32-openvino-encnpu-cache", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-fp32_encnpucacheb_ovcpu-npu", "genai", [rf"{dx}-fp32-openvino-encnpu-cache", "--ep", "config"], "npu", True),
            # distil-lev7: the int8 encoder on the NPU with the decoder on the Intel GPU (steps folders-distil-npu3);
            # state: the KV buffer shared, past_present_share_buffer true in a copy of the fp16 CUDA export's
            # genai_config.json (pc/genai_search_folder.py, steps folders-distil-sharebuf), alone, with the CUDA graph
            # (which the builder's false refuses: "Graph capture is not supported with past_present_share_buffer set
            # to false.") and with sdpa_kernel=1
            ("distil-large-v3/genai-int8_encnpustatic_ovgpu-npu", "genai", [rf"{dx}-int8-openvino-encnpu-gpu", "--ep", "config"], "npu", True),
            ("distil-large-v3/genai-fp16_sharebuf-rtx", "genai-cuda", [rf"{dx}-fp16-cuda-sharebuf", "--ep", "cuda"], "rtx", False),
            ("distil-large-v3/genai-fp16_sharebufcg-rtx", "genai-cuda", [rf"{dx}-fp16-cuda-sharebuf", "--ep", "cuda"] + cg, "rtx", False),
            ("distil-large-v3/genai-fp16_sharebufsdpa1-rtx", "genai-cuda",
             [rf"{dx}-fp16-cuda-sharebuf", "--ep", "cuda", "--ep-option", "sdpa_kernel=1"], "rtx", False)):
        c.append(cell(cid, venv, "run_asr_genai.py", ["--model-dir"] + a + ["--langs", "en_us"], "S150", dev, 6,
                      probe=probe, group="dl", timeout=30 if dev == "npu" else 20))
    # Qwen3-ASR 1.7B
    for form in ("fp16", "int8"):
        for dev in ("cpu", "igpu", "npu", "rtx"):
            if dev in ("npu", "rtx") and form != "int8":
                continue
            c.append(cell(f"qwen3asr/ovg-{form}-{dev}", "ovg", "run_asr_ovg.py",
                          ["--model-dir", rf"{X}\qwen3asr-ov-{form}", "--family", "qwen3asr", "--device", OVDEV[dev],
                           "--cache-dir", r"{RUN}\cache"], "S150", dev, 9 if form == "fp16" else 7,
                          probe=dev in ("npu", "rtx"), group="b", timeout=40 if dev == "cpu" else 30))
    qlt = rf"{M}\litert-community__Qwen3-ASR-1.7B\Qwen3-ASR-1.7B.litertlm"
    glt = rf"{M}\litert-community__granite-speech-5.0-2b\granite-speech-5.0-2b.litertlm"
    for fam, bundle, cset in (("qwen3asr", qlt, "S150"), ("granite", glt, "S220")):
        for dev, be, pref in (("cpu", "cpu", 0), ("igpu", "gpu", 1), ("rtx", "gpu", 2), ("npu", "npu", 0)):
            a = ["--bundle", bundle, "--family", fam, "--backend", be, "--cache-dir", r"{RUN}\cache"]
            if be != "cpu":
                # both bundles' cards: the audio encoder runs on the CPU (Granite's 36 s file: "CPU only"; Qwen3-ASR's
                # GPU rows: "LM on the GPU, audio encoder CPU"): the language model goes to the device
                a += ["--audio-backend", "cpu"]
            if pref == 2:
                a += ["--gpu-pref", str(pref)]
            c.append(cell(f"{fam}/litertlm-{dev}", "litertlm", "run_asr_litertlm.py", a, cset, dev, 7,
                          probe=(dev == "npu"), timeout=45 if dev == "cpu" else 30))
            if pref == 1:
                # Z3b (11-10): the Intel GPU through the chain, which writes GpuPreference=1 for the venv's exes before
                # the engine process starts (pc/chain.ps1 "gpu_pref"); Z3's runner (--gpu-pref) wrote it under the
                # junction's target (os.path.realpath) and every such run of Z3 took the RTX
                c[-1]["gpu_pref"] = 1
    # Parakeet TDT 0.6B v3 (en_us)
    px = rf"{M}\istupakov__parakeet-tdt-0.6b-v3-onnx"
    for cid, venv, a, dev, probe in (
            ("parakeet/ort-fp32-cpu", "ort", ["--ep", "cpu"], "cpu", False),
            ("parakeet/ort-int8-cpu", "ort", ["--ep", "cpu", "--quant", "int8"], "cpu", False),
            ("parakeet/ort-fp32-rtx", "ort-cuda", ["--ep", "cuda"], "rtx", False),
            ("parakeet/ort-fp32-igpu", "ort", ["--ep", "openvino", "--ov-device", "GPU"], "igpu", False),
            ("parakeet/ort-fp32-npu", "ort", ["--ep", "openvino", "--ov-device", "NPU"], "npu", True)):
        c.append(cell(cid, venv, "run_asr_ort_parakeet.py", ["--model-dir", px] + a, "S150", dev, 4, probe=probe))
    # Kokoro-82M (T50)
    for form in ("fp16", "int8"):
        for dev in ("cpu", "igpu", "npu", "rtx"):
            if dev in ("npu", "rtx") and form != "fp16":
                continue
            c.append(cell(f"kokoro/ovg-{form}-{dev}", "ovg", "run_tts_ovg.py",
                          ["--model-dir", rf"{M}\OpenVINO__Kokoro-82M-{form}-ov", "--device", OVDEV[dev],
                           "--cache-dir", r"{RUN}\cache"], "T50", dev, 3, probe=dev in ("npu", "rtx")))
    kx = rf"{M}\onnx-community__Kokoro-82M-v1.0-ONNX"
    for cid, venv, a, dev, probe in (
            ("kokoro/ort-fp32-cpu", "ort", ["--ep", "cpu"], "cpu", False),
            ("kokoro/ort-fp32-rtx", "ort-cuda", ["--ep", "cuda"], "rtx", False),
            ("kokoro/ort-fp32-igpu", "ort", ["--ep", "openvino", "--ov-device", "GPU"], "igpu", False),
            ("kokoro/ort-fp32-npu", "ort", ["--ep", "openvino", "--ov-device", "NPU"], "npu", True)):
        c.append(cell(cid, venv, "run_tts_ort.py",
                      ["--model", rf"{kx}\onnx\model.onnx", "--voice-bin", rf"{kx}\voices\af_heart.bin"] + a, "T50", dev, 3,
                      probe=probe))
    return c


CONTROL = {"cell": "control", "venv": "ovg", "script": "run_asr_ovg.py",
           "args": ["--model-dir", rf"{M}\OpenVINO__whisper-large-v3-turbo-int8-ov", "--family", "whisper", "--device",
                    "CPU", "--cache-dir", r"{RUN}\cache", "--set", "S150", "--fixtures", "{FIX}", "--first", "10"],
           "device": "cpu", "ram_need_gb": 4, "timeout_min": 10, "engines_gate": False}


def unit(c, first=0):
    a = list(c["args"]) + ["--set", c["set"], "--fixtures", "{FIX}"]
    if first:
        a += ["--first", str(first)]
    u = {k: c[k] for k in ("cell", "venv", "script", "device", "ram_need_gb", "timeout_min", "engines_gate")}
    if c.get("gpu_pref"):
        u["gpu_pref"] = c["gpu_pref"]
    u["args"] = a
    return u


def step(name, exe, args, venv=None, timeout=60, ram=None, lock=False, stop=False):
    s = {"name": name, "exe": exe, "args": args, "timeout_min": timeout}
    if venv:
        s["venv"] = venv
    if ram:
        s["ram_gb"] = ram
    if lock:
        s["lock"] = True
    if stop:
        s["stop_on_fail"] = True
    return s


VENVS = ["ovg", "genai", "genai-cuda", "ort", "ort-cuda", "litertlm", "ref", "ovexport"]


def steps():
    out = {}
    s = [step("python", "uv", ["python", "install", "3.12"], timeout=15, stop=True)]
    for v in VENVS:
        s.append(step(f"venv-{v}", "uv", ["venv", rf"{{ROOT}}\.venv-{v}", "--python", "3.12", "--python-preference",
                                          "only-managed", "--allow-existing"], timeout=10))
        s.append(step(f"pip-{v}", "uv", ["pip", "install", "--python", rf"{{ROOT}}\.venv-{v}\Scripts\python.exe", "-r",
                                         rf"{{ROOT}}\pc\req\{v}.txt"], timeout=45))
        if v == "ort-cuda":
            s.append(step("pip-ort-cuda-nodeps", "uv", ["pip", "install", "--python", r"{ROOT}\.venv-ort-cuda\Scripts\python.exe",
                                                        "--no-deps", "-r", r"{ROOT}\pc\req\ort-cuda.nodeps.txt"], timeout=10))
        s.append(step(f"versions-{v}", "py", [r"{ROOT}\pc\venv_versions.py", "--out", rf"{{ROOT}}\logs\versions-{v}.txt"], venv=v,
                      timeout=5))
    s.append(step("fixtures", "py", [r"{ROOT}\pc\check_fixtures.py", "--fixtures", "{FIX}"], venv="ovg", timeout=5))
    out["setup"] = s
    out["fetch"] = [step("python", "uv", ["python", "install", "3.12"], timeout=15, stop=True),
                    step("fetch", "uv", ["run", "--no-project", "--python", "3.12", "--python-preference", "only-managed",
                                         "python", r"{ROOT}\pc\fetch.py", "--manifest", r"{ROOT}\shelf\hub_manifest.json",
                                         "--dest", r"{ROOT}\models"], timeout=240)]
    w = rf"{M}\openai__whisper-large-v3-turbo"   # the builder's -i (a local folder: no -m, its help says)
    # ONNX Runtime GenAI's Whisper export: the builder's Whisper encoder makes cross K/V per encoder layer from
    # decoder.layers[i] = large-v3-turbo (32 / 4 layers) stops with IndexError (11-09); pc/patch_genai_whisper.py makes it
    # count decoder layers, as the runtime does (src/models/io/cross_kv_cache.cpp)
    out["export-whisper"] = [
        step("patch-genai-whisper", "py", [r"{ROOT}\pc\patch_genai_whisper.py"], venv="ref", timeout=5, stop=True),
        step("genai-whisper-fp32-cpu", "py", ["-m", "onnxruntime_genai.models.builder",
                                              "-i", w, "-o", rf"{X}\whisper-genai-fp32-cpu", "-p", "fp32", "-e", "cpu",
                                              "-c", r"{ROOT}\exports\cache-genai"], venv="ref", timeout=40, ram=8),
        step("genai-whisper-fp16-cuda", "py", ["-m", "onnxruntime_genai.models.builder",
                                               "-i", w, "-o", rf"{X}\whisper-genai-fp16-cuda", "-p", "fp16", "-e", "cuda",
                                               "-c", r"{ROOT}\exports\cache-genai"], venv="ref", timeout=40, ram=8),
    ]
    # Distil-Whisper distil-large-v3 (11-12): the same patch, then the builder from the checkpoint's model.safetensors
    # (fp16 values; fp32 for the CPU export, as the transformers reference reads them); the int4 forms are separate
    # (speed levers: a form the builder cannot make is a no-go row, not a failed walk)
    dw = rf"{M}\distil-whisper__distil-large-v3"

    def genai_export(name, form, ep):
        return step(name, "py", ["-m", "onnxruntime_genai.models.builder", "-i", dw, "-o",
                                 rf"{X}\distil-large-v3-genai-{form}-{ep}", "-p", form, "-e", ep,
                                 "-c", r"{ROOT}\exports\cache-genai"], venv="ref", timeout=40, ram=8)
    out["export-distil"] = [
        step("patch-genai-whisper", "py", [r"{ROOT}\pc\patch_genai_whisper.py"], venv="ref", timeout=5, stop=True),
        genai_export("genai-distil-fp32-cpu", "fp32", "cpu"),
        genai_export("genai-distil-fp16-cuda", "fp16", "cuda"),
    ]
    out["export-distil-int4"] = [
        step("patch-genai-whisper", "py", [r"{ROOT}\pc\patch_genai_whisper.py"], venv="ref", timeout=5, stop=True),
        genai_export("genai-distil-int4-cpu", "int4", "cpu"),
        genai_export("genai-distil-int4-cuda", "int4", "cuda"),
    ]
    out["export-distil-levers"] = [
        step("patch-genai-whisper", "py", [r"{ROOT}\pc\patch_genai_whisper.py"], venv="ref", timeout=5, stop=True),
        genai_export("genai-distil-int8-cpu", "int8", "cpu"),
        genai_export("genai-distil-int8-cuda", "int8", "cuda"),
        genai_export("genai-distil-fp16-webgpu", "fp16", "webgpu"),
    ]
    # the venvs and Hub files the distil-large-v3 cells and the control row need (a folder of its own on the PC)
    keep = {"python"} | {f"{k}-{v}" for k in ("venv", "pip", "versions") for v in ("ovg", "genai", "genai-cuda", "ref")}
    out["setup-distil"] = [x for x in s if x["name"] in keep] + [
        step("fixtures", "py", [r"{ROOT}\pc\check_fixtures.py", "--fixtures", "{FIX}", "--sets", "S150"], venv="ovg", timeout=5)]
    out["fetch-distil"] = [out["fetch"][0], step("fetch", "uv", out["fetch"][1]["args"] + [
        "--only", "distil-whisper/distil-large-v3", "whisper-large-v3-turbo-int8-ov"], timeout=60)]
    # optimum-intel's Qwen3-ASR IR: over 8 GB each = the PC's lock once as "(not a timing)", at the slot of the PC's order
    out["export-ov"] = []
    for form in ("fp16", "int8"):
        out["export-ov"].append(step(f"ov-qwen3asr-{form}", r"{ROOT}\.venv-ovexport\Scripts\optimum-cli.exe",
                                     ["export", "openvino", "--model", rf"{M}\Qwen__Qwen3-ASR-1.7B", "--task",
                                      "automatic-speech-recognition-with-past", "--trust-remote-code", "--weight-format", form,
                                      rf"{X}\qwen3asr-ov-{form}"], timeout=30, ram=14, lock=True))
    out["refs"] = [
        step("ref-whisper", "py", [r"{ROOT}\pc\ref_hf.py", "--family", "whisper", "--model-dir", w, "--set", "S150",
                                   "--fixtures", "{FIX}", "--out", r"{ROOT}\refs\whisper\ref.json"], venv="ref", timeout=120, ram=6),
        step("ref-parakeet", "py", [r"{ROOT}\pc\ref_hf.py", "--family", "parakeet", "--model-dir",
                                    rf"{M}\nvidia__parakeet-tdt-0.6b-v3", "--set", "S150", "--langs", "en_us", "--fixtures",
                                    "{FIX}", "--out", r"{ROOT}\refs\parakeet\ref.json"], venv="ref", timeout=40, ram=6),
        step("ref-kokoro", "py", [r"{ROOT}\pc\ref_kokoro_pt.py", "--model-dir", rf"{M}\hexgrad__Kokoro-82M", "--set", "T50",
                                  "--fixtures", "{FIX}", "--out", r"{ROOT}\refs\kokoro\ref.json"], venv="ref", timeout=40, ram=4),
        step("writeback-ref-kokoro", "py", [r"{ROOT}\pc\writeback.py", "--asr-dir",
                                            rf"{M}\OpenVINO__whisper-large-v3-turbo-fp16-ov", "--run-json",
                                            r"{ROOT}\refs\kokoro\ref.json"], venv="ovg", timeout=30, ram=4),
    ]
    out["refs-distil"] = [
        step("ref-distil", "py", [r"{ROOT}\pc\ref_hf.py", "--family", "whisper", "--model-dir", dw, "--set", "S150",
                                  "--langs", "en_us", "--fixtures", "{FIX}", "--out", r"{ROOT}\refs\distil-large-v3\ref.json"],
             venv="ref", timeout=60, ram=6),
    ]
    # the Qwen3-ASR OpenVINO cells' first-pass gap from the reference of earlier Qwen3-ASR work (11-09): the exported checkpoint in its own
    # package, PyTorch fp32, on the first pass's 30 clips (pc/ref_qwen_asr_pkg.py)
    out["diag-qwen3"] = [
        step("ref-qwen3asr-pkg", "py", [r"{ROOT}\pc\ref_qwen_asr_pkg.py", "--model-dir", rf"{M}\Qwen__Qwen3-ASR-1.7B",
                                        "--set", "S150", "--first", "10", "--fixtures", "{FIX}", "--out",
                                        r"{ROOT}\refs\qwen3asr-pkg\first10.json"], venv="ovexport", timeout=40, ram=10),
    ]
    # then the exported fp16 IR through optimum-intel's Python class, fed the package's processor inputs
    # (pc/ref_qwen_asr_ovpy.py, optimum-intel's test form: f32 inference precision on CPU)
    out["diag-qwen3b"] = [
        step("ref-qwen3asr-ovpy-fp16", "py", [r"{ROOT}\pc\ref_qwen_asr_ovpy.py", "--ir-dir", rf"{X}\qwen3asr-ov-fp16",
                                              "--model-dir", rf"{M}\Qwen__Qwen3-ASR-1.7B", "--set", "S150", "--first", "10",
                                              "--fixtures", "{FIX}", "--out", r"{ROOT}\refs\qwen3asr-ovpy\fp16-f32.json"],
             venv="ovexport", timeout=40, ram=8),
    ]
    d = K / "pc" / "steps"
    d.mkdir(parents=True, exist_ok=True)
    for name, lst in out.items():
        (d / f"{name}.json").write_text(json.dumps(lst, indent=1) + "\n", encoding="utf-8")
        print(f"pc/steps/{name}.json: {len(lst)} steps")


def writeback_steps(sitting, cell_ids):
    # one step per TTS cell (11-10): Whisper encodes a 30 s window per sentence (the first pass's write-back: 60
    # sentences in 9.9 min with the load, 11-09 15:11), so 7 timed cells x 50 take about an hour; a kill for another
    # lane's lock then reruns one cell, not all, and each step's limit holds while another job shares the CPU
    lst = [step(f"writeback-{sitting}-{c.split('/')[1]}", "py",
                [r"{ROOT}\pc\writeback.py", "--asr-dir", rf"{M}\OpenVINO__whisper-large-v3-turbo-fp16-ov", "--run-json",
                 rf"{{ROOT}}\runs\{sitting}\{c.replace('/', chr(92))}\run1.json"], venv="ovg", timeout=40, ram=4)
           for c in cell_ids]
    p = K / "pc" / "steps" / f"writeback-{sitting}.json"
    p.write_text(json.dumps(lst, indent=1) + "\n", encoding="utf-8")
    print(f"{p.relative_to(K)}: {len(cell_ids)} TTS runs")


def est_from(sittings, c):
    """Seconds one full run of cell c is expected to take: a first-pass run's load and warm-up plus its loop time
    scaled by clip count (full set / the clips it ran)."""
    n_full = {"S150": 150, "S220": 220, "T50": 50}[c["set"]]
    if "--langs" in c["args"] or c["cell"].startswith("parakeet/"):
        n_full = 50   # Parakeet: en_us only
    for s in sittings:
        meta = K / "raw" / s / c["cell"] / "run1.meta.json"
        run = K / "raw" / s / c["cell"] / "run1.json"
        if not (meta.exists() and run.exists()):
            continue
        m = json.loads(meta.read_text(encoding="utf-8-sig"))
        r = json.loads(run.read_text(encoding="utf-8-sig"))
        if r.get("status") != "OK" or not r.get("rows"):
            continue   # a failed run here; a redo in a later first pass may have the OK one
        el = float(m["engine"]["elapsedSeconds"])
        fixed = float(r.get("load_s") or 0) + float(r.get("warmup_s") or 0)
        return round(fixed + max(el - fixed, 1) * n_full / len(r["rows"]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["steps", "first", "timed", "list", "writeback", "redo"])
    ap.add_argument("--from", dest="from_list", default="", help="redo: the step list whose named steps run again")
    ap.add_argument("--names", default="", help="redo: the step names (comma list)")
    ap.add_argument("--out", default="", help="redo: the list's name (default redo-<from>)")
    ap.add_argument("--sitting")
    ap.add_argument("--group", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--est-from", default="")
    ap.add_argument("--devices", default="", help="only cells on these devices (cpu,igpu,rtx,npu)")
    ap.add_argument("--known", type=float, default=0)
    ap.add_argument("--control-est", type=int, default=0,
                    help="timed: the control row's expected seconds per run (default: its first-pass run in --est-from)")
    ap.add_argument("--warm-runs", type=int, default=1,
                    help="timed: uncounted control-warm runs before the head (2 when a first head run came out slow)")
    ap.add_argument("--slots", default="", help="JSON list of {who, release_at, give_up_at}")
    ap.add_argument("--end-by", default="", help="timed: ISO time no cell may run past (pc/chain.ps1 leaves the rest)")
    ap.add_argument("--first-n", type=int, default=10,
                    help="first: clips per language (0 = every clip of the set; Z3b's full first pass of one cell)")
    ap.add_argument("--timeout-min", type=int, default=0, help="first / timed: every cell unit's time limit (minutes)")
    ap.add_argument("--no-windows", action="store_true",
                    help="timed: one block to end_by (pc/chain.ps1 opens no duty window for other lanes' notes)")
    args = ap.parse_args()
    cs = cells()
    if args.what == "list":
        for c in cs:
            print(f"{c['cell']:28s} {c['device']:5s} {c['set']:5s} group {c['group']}{'  probe' if c['probe'] else ''}")
        return
    if args.what == "steps":
        steps()
        return
    if args.what == "redo":
        src = {}
        for fl in [x for x in args.from_list.split(",") if x]:   # several lists: steps keep the order of --names
            for st in json.loads((K / "pc" / "steps" / f"{fl}.json").read_text(encoding="utf-8")):
                src.setdefault(st["name"], st)
        names = [n for n in args.names.split(",") if n]
        lst = [src[n] for n in names if n in src]
        if len(lst) != len(names):
            sys.exit(f"not in {args.from_list}: {sorted(set(names) - set(src))}")
        p = K / "pc" / "steps" / f"{args.out or 'redo-' + args.from_list.split(',')[0]}.json"
        p.write_text(json.dumps(lst, indent=1) + "\n", encoding="utf-8")
        print(f"{p.relative_to(K)}: {len(lst)} steps ({', '.join(names)})")
        return
    if args.what == "writeback":
        writeback_steps(args.sitting, [x for x in args.cells.split(",") if x])
        return
    devs = [d for d in args.devices.split(",") if d]
    pick = [c for c in cs if (not args.cells or c["cell"] in args.cells.split(",")) and (not args.group or c["group"] == args.group)
            and (not devs or c["device"] in devs)]
    if args.cells:
        missing = set(args.cells.split(",")) - {c["cell"] for c in pick}
        if missing:
            sys.exit(f"unknown cells: {sorted(missing)}")
    plan = {"sitting": args.sitting, "made": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")}
    if args.what == "first":
        plan["mode"] = "first"
        plan["units"] = [unit(c, first=args.first_n) for c in pick]
        for u in plan["units"]:
            u["timeout_min"] = max(15, u["timeout_min"] // 2) if u["device"] != "npu" else 30
            if args.timeout_min:
                u["timeout_min"] = args.timeout_min
    else:
        plan["mode"] = "timed"
        srcs = [s for s in args.est_from.split(",") if s]
        units = [dict(CONTROL, cell="control-warm", role="control-warm", **({"runs": args.warm_runs} if args.warm_runs > 1 else {})),
                 dict(CONTROL, cell="control-head", role="control-head", runs=1 if args.known else 3)]
        # the control row is the first pass of whisper/ovg-int8-cpu (the same 30 clips): its run time is the control's
        ce = None
        for s_ in srcs:
            mj = K / "raw" / s_ / "whisper" / "ovg-int8-cpu" / "run1.meta.json"
            if mj.exists():
                ce = round(float(json.loads(mj.read_text(encoding="utf-8-sig"))["engine"]["elapsedSeconds"]))
                break
        if args.control_est:
            ce = args.control_est
        if ce:
            for u in units:
                u["est_sec"] = ce
        cells_ = []
        for c in pick:
            u = unit(c)
            e = est_from(srcs, c)
            if e:
                u["est_sec"] = e
                u["timeout_min"] = max(u["timeout_min"], int(e * 2 / 60) + 5)
            if args.timeout_min:
                u["timeout_min"] = max(u["timeout_min"], args.timeout_min)
            cells_.append(u)
        # the shortest expected run first: an end_by leaves every cell after the first one that does not fit, so the
        # long cells go last (and to the later sitting), not the short ones behind them (11-09)
        units += sorted(cells_, key=lambda u: u.get("est_sec") or 60 * u["timeout_min"])
        units.append(dict(CONTROL, cell="control-tail", role="control-tail", **({"est_sec": ce} if ce else {})))
        plan["units"] = units
        if args.known:
            plan["control_known_rtf"] = args.known
        if args.slots:
            plan["slots"] = json.loads(args.slots)
        if args.end_by:
            plan["end_by"] = __import__("datetime").datetime.fromisoformat(args.end_by).isoformat()
        if args.no_windows:
            plan["windows"] = False
    d = K / "pc" / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.sitting}.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    tot = sum(u.get("est_sec") or 0 for u in plan["units"])
    print(f"pc/plans/{args.sitting}.json: {len(plan['units'])} units, estimated {tot / 60:.0f} min of runs")


if __name__ == "__main__":
    main()
