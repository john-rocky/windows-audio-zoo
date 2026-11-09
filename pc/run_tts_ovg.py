"""One engine process of an OpenVINO GenAI TTS cell: Text2SpeechPipeline with Kokoro-82M (OpenVINO's IR) on one device,
the T50 sentences after one untimed warm-up sentence; the waves go to <out dir>/wav/<id>.wav for the write-back ASR.

    python run_tts_ovg.py --model-dir <IR folder> --device CPU|GPU.0|GPU.1|NPU --set T50 --fixtures <dir>
        --out run.json [--voice af_heart] [--fallback-dir <OpenVINO G2P folder>] [--cache-dir <dir>]

Voice = <model dir>/voices/<voice>.bin reshaped to pipe.get_speaker_embedding_shape(); language en-us; speed 1.0.
G2P = the pipeline's own (built-in dictionary; out-of-dictionary words go to --fallback-dir or espeak-ng when one is
there): written in the JSON. Timing per sentence: pipe.generate wall clock; RTF = synthesis wall / wave seconds.
"""
import argparse
import os
import pathlib
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402

WARMUP_TEXT = "This sentence warms up the speech engine before the timed sentences."


def write_wav(path, samples, rate):
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--fallback-dir", default="")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    rec = al.Record("run_tts_ovg.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["openvino-genai", "openvino", "openvino-tokenizers", "numpy"])  # read before the imports: a failed import keeps them
        import numpy as np
        import openvino as ov
        import openvino_genai as ov_genai
        rows = al.clips(args.set, args.fixtures, args.first)
        wavdir = pathlib.Path(args.out).parent / "wav"
        wavdir.mkdir(parents=True, exist_ok=True)
        props = {}
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            props["CACHE_DIR"] = args.cache_dir
        t = time.perf_counter()
        pipe = ov_genai.Text2SpeechPipeline(args.model_dir, args.device, **props)
        emb = np.fromfile(os.path.join(args.model_dir, "voices", f"{args.voice}.bin"), dtype=np.float32)
        spk = ov.Tensor(emb.reshape(pipe.get_speaker_embedding_shape()))
        if args.fallback_dir:
            config = pipe.get_generation_config()
            config.phonemize_fallback_model_dir = args.fallback_dir
            pipe.set_generation_config(config)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["workload"] = {"set": args.set, "voice": args.voice, "language": "en-us", "speed": 1.0,
                           "g2p": "pipeline built-in dictionary + " + (f"OpenVINO fallback {args.fallback_dir}"
                                                                         if args.fallback_dir else "espeak-ng fallback if present")}

        def run(text):
            t0 = time.perf_counter()
            res = pipe.generate(text, spk, language="en-us")   # the sample's form (text2speech.py)
            wall = time.perf_counter() - t0
            data = np.array(res.speeches[0].data, dtype=np.float32).reshape(-1)
            return data, int(res.output_sample_rate), wall

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
