"""One engine process of an ONNX Runtime GenAI Whisper cell: a builder export (encoder.onnx + decoder.onnx +
genai_config.json) on one execution provider, one pass over a clip set after one untimed warm-up clip; one JSON.

    python run_asr_genai.py --model-dir <export> --ep cpu|cuda|openvino|webgpu [--ov-device GPU|NPU]
        [--ep-option key=value ...] --set S150 --fixtures <dir> --out run.json [--first N] [--max-new 440]

The loop of GenAI 0.19.0's examples/python/whisper.py: og.Audios.open(path) -> processor([prompt], audios) ->
Generator (greedy, num_beams 1, max_length = 4 + max_new) -> processor.decode(sequence). The prompt fixes the language
per clip: <|startoftranscript|><|xx|><|transcribe|><|notimestamps|> (PLAN section 2). GenAI takes the clip as a path,
so the timed span includes reading the WAV (about 1 MB). First token = from Audios.open to the first
generate_next_token's end.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


def make_model(og, args):
    config = og.Config(args.model_dir)
    names = {"openvino": "OpenVINO", "webgpu": "WebGPU", "cuda": "cuda"}
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
    ap.add_argument("--ep", required=True, choices=["cpu", "cuda", "openvino", "webgpu"])
    ap.add_argument("--ov-device", default="GPU")
    ap.add_argument("--ep-option", action="append", default=[])
    ap.add_argument("--max-new", type=int, default=440)
    args = ap.parse_args()
    rec = al.Record("run_asr_genai.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime-genai", "onnxruntime-genai-cuda", "onnxruntime", "onnxruntime-gpu",
                                       "onnxruntime-ep-openvino", "onnxruntime-ep-webgpu", "numpy"])  # read before the imports: a failed import keeps them
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()
        import onnxruntime_genai as og
        rows = al.clips(args.set, args.fixtures, args.first)
        t = time.perf_counter()
        model = make_model(og, args)
        processor = model.create_multimodal_processor()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["device_type"] = getattr(model, "device_type", None)
        rec["workload"] = {"set": args.set, "first": args.first, "max_new_tokens": args.max_new, "decode": "greedy",
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
