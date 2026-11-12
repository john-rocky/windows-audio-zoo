"""One engine process of an ONNX Runtime GenAI Whisper cell: a builder export (encoder.onnx + decoder.onnx +
genai_config.json) on one execution provider, one pass over a clip set after one untimed warm-up clip; one JSON.

    python run_asr_genai.py --model-dir <export> --ep cpu|cuda|openvino|webgpu|config [--ov-device GPU|NPU] [--encoder-ep config|same|openvino-gpu|openvino-npu]
        [--ep-option key=value ...] --set S150 --fixtures <dir> --out run.json [--first N] [--max-new 440]
        [--langs en_us]

--langs keeps the clips of those languages only (default: every language of the set): distil-large-v3 is an
English model, so its S150 cells run the 50 en_us clips (the form of pc/run_asr_ort_parakeet.py).

--encoder-ep config|same|openvino-gpu|openvino-npu: where the encoder runs. GenAI 0.19.0's clear_providers,
append_provider and set_provider_option change the decoder's session options only (src/config.cpp), and the Whisper
model gives its encoder the encoder's own session options from genai_config.json when the file has them
(src/models/whisper.cpp), which the builder writes: so with config (the default, the form of every run before
2026-11-12) a `-e cpu` export keeps its encoder on the CPU EP whatever --ep says. same = the encoder gets the
decoder's provider and options; openvino-gpu / openvino-npu = the encoder on the OpenVINO EP on that device. same and
openvino-* write a copy of genai_config.json with those encoder options into <the run JSON's folder>/cache/model, next
to hard links to the export's other files (GenAI reads an encoder's providers from the file at load only;
og.Config.overlay does not refresh them), and the run JSON records the encoder's session options.

--ep config (from 2026-11-12): the folder's genai_config.json as it is, no provider set in code (the form of a Hub
repo whose genai_config.json names its providers for the encoder and the decoder); a plugin EP the file names
(OpenVINO, WebGPU) is registered first, and the run JSON records the file's provider options (config_providers).

The loop of GenAI 0.19.0's examples/python/whisper.py: og.Audios.open(path) -> processor([prompt], audios) ->
Generator (greedy, num_beams 1, max_length = 4 + max_new) -> processor.decode(sequence). The prompt fixes the language
per clip: <|startoftranscript|><|xx|><|transcribe|><|notimestamps|> (PLAN section 2). GenAI takes the clip as a path,
so the timed span includes reading the WAV (about 1 MB). First token = from Audios.open to the first
generate_next_token's end.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


NAMES = {"openvino": "OpenVINO", "webgpu": "WebGPU", "cuda": "cuda"}


def encoder_model_dir(og, args, rec):
    """The folder og.Config reads: the export itself (--encoder-ep config), else a copy of its genai_config.json whose
    encoder session options name the chosen provider, next to hard links to the export's other files."""
    if args.encoder_ep == "config" or (args.encoder_ep == "same" and args.ep == "cpu"):
        return args.model_dir
    if args.encoder_ep == "same":
        name = NAMES[args.ep]
        opts = {"device_type": args.ov_device} if args.ep == "openvino" else {}
        for item in args.ep_option:
            key, value = item.split("=", 1)
            opts[key] = value
    else:
        name, opts = "OpenVINO", {"device_type": args.encoder_ep.split("-", 1)[1].upper()}
        if args.ep != "openvino":
            import onnxruntime_ep_openvino as ep
            og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
    src = pathlib.Path(args.model_dir)
    dst = pathlib.Path(args.out).resolve().parent / "cache" / "model"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file() and p.name != "genai_config.json":
            os.link(p, dst / p.name)
    cfg = json.loads((src / "genai_config.json").read_text(encoding="utf-8"))
    so = cfg["model"]["encoder"].setdefault("session_options", {})
    so["provider_options"] = [{name: opts}]
    (dst / "genai_config.json").write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    rec["encoder_session_options"] = so
    return str(dst)


def file_providers(model_dir):
    """--ep config: the provider options of the encoder and the decoder in the folder's genai_config.json."""
    cfg = json.loads((pathlib.Path(model_dir) / "genai_config.json").read_text(encoding="utf-8"))
    return {part: ((cfg["model"].get(part) or {}).get("session_options") or {}).get("provider_options", [])
            for part in ("encoder", "decoder")}


def make_model(og, args, rec):
    if args.ep == "config":
        rec["config_providers"] = fp = file_providers(args.model_dir)
        named = {k for opts in fp.values() for o in opts for k in o}
        if "OpenVINO" in named:
            import onnxruntime_ep_openvino as ep
            og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        if "WebGPU" in named:
            import onnxruntime_ep_webgpu as ep
            og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        return og.Model(og.Config(args.model_dir))
    config = og.Config(encoder_model_dir(og, args, rec))
    names = NAMES
    if args.ep == "openvino":
        import onnxruntime_ep_openvino as ep
        og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        config.clear_providers()
        config.append_provider("OpenVINO")
        config.set_provider_option("OpenVINO", "device_type", args.ov_device)
    elif args.ep == "webgpu":
        import onnxruntime_ep_webgpu as ep
        og.register_execution_provider_library(ep.get_ep_name(), ep.get_library_path())
        config.clear_providers()
        config.append_provider("WebGPU")
    elif args.ep == "cuda":
        config.clear_providers()
        config.append_provider("cuda")
    elif args.ep == "cpu":
        config.clear_providers()
    for item in args.ep_option:
        key, value = item.split("=", 1)
        config.set_provider_option(names.get(args.ep, args.ep), key, value)
    return og.Model(config)


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ep", required=True, choices=["cpu", "cuda", "openvino", "webgpu", "config"])
    ap.add_argument("--ov-device", default="GPU")
    ap.add_argument("--encoder-ep", default="config", choices=["config", "same", "openvino-gpu", "openvino-npu"],
                    help="where the encoder runs (the docstring): config = its genai_config.json providers (default)")
    ap.add_argument("--ep-option", action="append", default=[])
    ap.add_argument("--max-new", type=int, default=440)
    ap.add_argument("--langs", default="", help="only these languages of the set (comma list; default: all)")
    args = ap.parse_args()
    if args.ep == "config" and (args.ep_option or args.encoder_ep != "config"):
        ap.error("--ep config takes the file's providers as they are: no --ep-option, no --encoder-ep")
    rec = al.Record("run_asr_genai.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime", "onnxruntime-gpu",
                                       "onnxruntime-ep-openvino", "onnxruntime-ep-webgpu", "numpy"])  # read before the imports: a failed import keeps them
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()
        import onnxruntime_genai as og
        rows = al.clips(args.set, args.fixtures, args.first)
        if args.langs:
            langs = set(args.langs.split(","))
            rows = [r for r in rows if r["lang"] in langs]
        t = time.perf_counter()
        model = make_model(og, args, rec)
        processor = model.create_multimodal_processor()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["device_type"] = getattr(model, "device_type", None)
        rec["workload"] = {"set": args.set, "langs": sorted(set(args.langs.split(","))) if args.langs else None,
                           "first": args.first, "max_new_tokens": args.max_new, "decode": "greedy",
                           "language": "fixed per clip", "task": "transcribe", "timestamps": False,
                           "input": "WAV path (og.Audios.open), read inside the timed span"}

        def run(path, lang):
            code_ = al.LANG[lang][0] if lang in al.LANG else al.S220_LANG[lang]
            prompt = f"<|startoftranscript|><|{code_}|><|transcribe|><|notimestamps|>"
            t0 = time.perf_counter()
            audios = og.Audios.open(path)
            inputs = processor([prompt], audios=audios)
            params = og.GeneratorParams(model)
            params.set_search_options(do_sample=False, num_beams=1, num_return_sequences=1,
                                      max_length=4 + args.max_new, batch_size=1)
            gen = og.Generator(model, params)
            gen.set_inputs(inputs)
            first = None
            while not gen.is_done():
                gen.generate_next_token()
                if first is None:
                    first = time.perf_counter() - t0
            seq = gen.get_sequence(0)
            text = processor.decode(seq)
            wall = time.perf_counter() - t0
            n = len(seq)
            del gen, params
            return text, wall, first, n

        if not args.no_warmup:
            _, ws, _, _ = run(al.warmup_wav(args.fixtures), "en_us")
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            text, wall, first, n = run(r["path"], r["lang"])
            rec.row({"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall, "text": text.strip(),
                     "first_token_ms": first * 1000 if first is not None else None, "sequence_tokens": n})
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
