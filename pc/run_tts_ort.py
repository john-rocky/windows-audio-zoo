"""One engine process of an ONNX Runtime TTS cell: Kokoro-82M (onnx-community's ONNX, fp32 or fp16) through kokoro-onnx
(its espeak-ng phonemes, its batching) with the session on one execution provider; the T50 sentences after one
untimed warm-up sentence; waves to <out dir>/wav/<id>.wav for the write-back ASR.

    python run_tts_ort.py --model <model.onnx> --voice-bin <voices/af_heart.bin> --ep cpu|cuda|openvino|webgpu
        [--ov-device GPU|NPU] --set T50 --fixtures <dir> --out run.json

The voice file of onnx-community (float32, [510, 1, 256]) goes into a one-voice .npz for kokoro-onnx; language en-us,
speed 1.0. Timing per sentence: Kokoro.create wall clock (phonemes + model); RTF = synthesis wall / wave seconds.
"""
import argparse
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402
from run_asr_ort_parakeet import device_session  # noqa: E402
from run_tts_ovg import WARMUP_TEXT, write_wav  # noqa: E402


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model", required=True)
    ap.add_argument("--voice-bin", required=True)
    ap.add_argument("--ep", required=True, choices=["cpu", "cuda", "openvino", "webgpu"])
    ap.add_argument("--ov-device", default="GPU")
    ap.add_argument("--ep-option", action="append", default=[])
    args = ap.parse_args()
    rec = al.Record("run_tts_ort.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["onnxruntime", "onnxruntime-gpu", "onnxruntime-ep-openvino",
                                       "onnxruntime-ep-webgpu", "kokoro-onnx", "espeakng-loader", "phonemizer", "numpy"])  # read before the imports: a failed import keeps them
        rec["cuda_dll_dirs"] = al.add_cuda_dirs()
        import numpy as np
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        rows = al.clips(args.set, args.fixtures, args.first)
        wavdir = pathlib.Path(args.out).parent / "wav"
        wavdir.mkdir(parents=True, exist_ok=True)
        voice = np.fromfile(args.voice_bin, dtype=np.float32).reshape(-1, 1, 256)
        npz = os.path.join(tempfile.mkdtemp(prefix="z3voice_"), "voices.npz")
        np.savez(npz, af_heart=voice)
        t = time.perf_counter()
        dev = {}
        sess = device_session(ort, args.model, args, args.ep, dev)
        kok = Kokoro.from_session(sess, npz)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        dev.pop("registered_ov", None)
        dev.pop("registered_wg", None)
        rec["device"] = dev
        rec["session_providers"] = sess.get_providers()
        rec["workload"] = {"set": args.set, "voice": "af_heart", "language": "en-us", "speed": 1.0,
                           "g2p": "kokoro-onnx (espeak-ng through espeakng-loader)", "model": os.path.basename(args.model)}

        def run(text):
            t0 = time.perf_counter()
            audio, rate = kok.create(text, voice=voice, speed=1.0, lang="en-us")
            return np.asarray(audio, dtype=np.float32).reshape(-1), int(rate), time.perf_counter() - t0

        if not args.no_warmup:
            _, _, ws = run(WARMUP_TEXT)
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            data, rate, wall = run(r["ref"])
            path = wavdir / f"{r['id']}.wav"
            write_wav(path, data, rate)
            rec.row({"id": r["id"], "text_in": r["ref"], "wall_s": wall, "audio_s": len(data) / rate, "rate": rate,
                     "peak_abs": float(np.abs(data).max()) if len(data) else 0.0,
                     "all_silent": bool(len(data) == 0 or float(np.abs(data).max()) < 1e-3), "wav": str(path)})
        rec["summary"] = rec.summarize("tts")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
