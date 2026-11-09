"""One engine process of an ONNX Runtime Parakeet TDT cell: istupakov's ONNX export (encoder + decoder_joint, fp32 or
int8) with onnx-asr's NeMo preprocessor (its NumPy form when built on the CPU provider; the class is recorded) and onnx-asr's TDT greedy loop; the encoder and decoder_joint
sessions run on one execution provider. One pass over a clip set after one untimed warm-up clip; one JSON.

    python run_asr_ort_parakeet.py --model-dir <onnx folder> [--quant int8] --ep cpu|cuda|openvino|webgpu
        [--ov-device GPU|NPU] [--decoder-ep cpu] --set S150 --fixtures <dir> --first 0 --langs en_us --out run.json

onnx-asr 0.12.0 builds the model with CPU sessions; this runner then replaces model.asr._encoder and
model.asr._decoder_joint with sessions on the chosen EP (OpenVINO = the plugin EP 1.8.1 through
SessionOptions.add_provider_for_devices, the form the vision zoo used on this PC). Parakeet v3's languages do not
include Mandarin or Japanese: the S150 cells run the 50 en_us clips (--langs en_us). Timing per clip:
model.recognize(samples, sample_rate=16000) wall clock (features, encoder, decode).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


def device_session(ort, path, args, ep, rec_dev):
    so = ort.SessionOptions()
    if ep == "cpu":
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    if ep == "cuda":
        s = ort.InferenceSession(path, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in s.get_providers():
            raise RuntimeError(f"CUDA EP not in the session providers: {s.get_providers()}")
        return s
    if ep == "openvino":
        import onnxruntime_ep_openvino as ov_ep
        if not rec_dev.get("registered_ov"):
            ort.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
            rec_dev["registered_ov"] = True
        # the plugin names its devices as OpenVINO does: the Intel GPU is "GPU.0" (11-09: NPU, GPU.0, CPU), so "GPU"
        # also matches "GPU.<n>" (the first one listed)
        want = args.ov_device
        devs = [d for d in ort.get_ep_devices() if d.ep_name == ov_ep.get_ep_name()
                and (str(d.ep_metadata.get("ov_device")) == want or str(d.ep_metadata.get("ov_device")).startswith(want + "."))]
        if not devs:
            raise RuntimeError(f"no OpenVINO EP device {args.ov_device}: "
                               f"{[d.ep_metadata.get('ov_device') for d in ort.get_ep_devices()]}")
        meta = dict(devs[0].ep_metadata)
        rec_dev["openvino"] = {k: meta.get(k) for k in ("FULL_DEVICE_NAME", "ov_device", "version", "os_driver_version",
                                                         "NPU_COMPILER_VERSION", "DEVICE_LUID")}
        so.add_provider_for_devices(devs, dict(x.split("=", 1) for x in args.ep_option))
        return ort.InferenceSession(path, sess_options=so)
    if ep == "webgpu":
        import onnxruntime_ep_webgpu as wg
        if not rec_dev.get("registered_wg"):
            ort.register_execution_provider_library(wg.get_ep_name(), wg.get_library_path())
            rec_dev["registered_wg"] = True
        devs = [d for d in ort.get_ep_devices() if d.ep_name == wg.get_ep_name()]
        so.add_provider_for_devices(devs[:1], dict(x.split("=", 1) for x in args.ep_option))
        return ort.InferenceSession(path, sess_options=so)
    raise ValueError(ep)


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--quant", default="", choices=["", "int8"])
    ap.add_argument("--ep", required=True, choices=["cpu", "cuda", "openvino", "webgpu"])
    ap.add_argument("--ov-device", default="GPU")
    ap.add_argument("--ep-option", action="append", default=[])
    ap.add_argument("--decoder-ep", default="", help="run decoder_joint on another EP (default: the same as --ep)")
    ap.add_argument("--langs", default="en_us")
    args = ap.parse_args()
    rec = al.Record("run_asr_ort_parakeet.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-openvino",
                                       "onnxruntime-ep-webgpu", "onnx-asr", "numpy"])  # read before the imports: a failed import keeps them
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()
        import onnx_asr
        import onnxruntime as ort
        langs = set(args.langs.split(","))
        rows = [r for r in al.clips(args.set, args.fixtures, args.first) if r["lang"] in langs]
        audio = {r["id"]: al.read_wav(r["path"]) for r in rows}
        t = time.perf_counter()
        model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", args.model_dir, quantization=args.quant or None,
                                    providers=["CPUExecutionProvider"])
        suffix = f".{args.quant}" if args.quant else ""
        dev = {}
        enc = os.path.join(args.model_dir, f"encoder-model{suffix}.onnx")
        dec = os.path.join(args.model_dir, f"decoder_joint-model{suffix}.onnx")
        model.asr._encoder = device_session(ort, enc, args, args.ep, dev)
        model.asr._decoder_joint = device_session(ort, dec, args, args.decoder_ep or args.ep, dev)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        dev.pop("registered_ov", None)
        dev.pop("registered_wg", None)
        rec["device"] = dev
        rec["session_providers"] = {"encoder": model.asr._encoder.get_providers(),
                                    "decoder_joint": model.asr._decoder_joint.get_providers()}
        pre = getattr(model.asr, "_preprocessor", None)
        pre_name = type(pre).__name__ + (f"({type(pre.preprocessor).__name__})" if hasattr(pre, "preprocessor") else "")
        rec["workload"] = {"set": args.set, "langs": sorted(langs), "first": args.first, "decode": "TDT greedy (onnx-asr)",
                           "files": [os.path.basename(enc), os.path.basename(dec)],
                           "preprocessor": f"onnx-asr {pre_name} (built with the CPU providers)"}

        def run(samples):
            t0 = time.perf_counter()
            text = model.recognize(samples, sample_rate=16000)
            return text, time.perf_counter() - t0

        if not args.no_warmup:
            _, ws = run(al.read_wav(al.warmup_wav(args.fixtures)))
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            text, wall = run(audio[r["id"]])
            rec.row({"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall,
                     "text": (text or "").strip(), "first_token_ms": None})
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
