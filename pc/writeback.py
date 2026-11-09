"""The TTS write-back (PLAN section 1): every <run dir>/wav/<id>.wav (24 kHz) resampled to 16 kHz (scipy resample_poly
2/3) and transcribed by one fixed ASR = OpenVINO GenAI's ASRPipeline with whisper-large-v3-turbo-fp16-ov on the CPU,
language <|en|>, greedy, no timestamps. Not timed. Writes <run>.writeback.json next to each run JSON (id -> text).

    python writeback.py --asr-dir <whisper-large-v3-turbo-fp16-ov folder> --run-json <run.json> [<run.json> ...] [--device CPU]
"""
import argparse
import json
import os
import pathlib
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


def read_any(path):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    return pcm, rate


def main():
    al.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--run-json", required=True, nargs="+")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    al.high_qos()   # untimed, but the same state as the runs
    import openvino_genai as ov_genai
    from scipy.signal import resample_poly
    pipe = ov_genai.ASRPipeline(args.asr_dir, args.device)
    for rj in args.run_json:
        if os.path.exists(rj):
            one(pipe, args, rj, read_any, resample_poly)


def one(pipe, args, run_json, read_any, resample_poly):
    run = json.loads(pathlib.Path(run_json).read_text(encoding="utf-8"))
    out = {"asr": {"pipeline": "openvino_genai.ASRPipeline", "model": os.path.basename(args.asr_dir.rstrip("\\/")),
                   "device": args.device, "language": "<|en|>",
                   "versions": al.versions(["openvino-genai", "openvino", "scipy", "numpy"])},
           "source": run_json, "texts": {}, "started": al.now()}
    for r in run.get("rows", []):
        wav = r.get("wav")
        if not wav or not os.path.exists(wav):
            out["texts"][r["id"]] = None
            continue
        x, rate = read_any(wav)
        if rate != 16000:
            g = {24000: (2, 3)}.get(rate)
            if not g:
                raise ValueError(f"{wav}: rate {rate}")
            x = resample_poly(x, g[0], g[1]).astype("float32")
        res = pipe.generate(x.tolist(), language="<|en|>", task="transcribe", return_timestamps=False, max_new_tokens=440)
        out["texts"][r["id"]] = (res.texts[0] if res.texts else "").strip()
    out["ended"] = al.now()
    p = pathlib.Path(run_json).with_name(pathlib.Path(run_json).stem + ".writeback.json")
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"writeback: {len(out['texts'])} texts -> {p}")


if __name__ == "__main__":
    main()
