"""The Kokoro-82M reference (PLAN section 1): hexgrad's kokoro package (KModel + KPipeline, misaki's English G2P), fp32,
the PC's CPU, the T50 sentences, voice af_heart, speed 1; waves to <out dir>/wav/<id>.wav for the write-back ASR.

    python ref_kokoro_pt.py --model-dir <hexgrad/Kokoro-82M folder> --set T50 --fixtures <dir> --out ref.json
"""
import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402
from run_tts_ovg import write_wav  # noqa: E402


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    rec = al.Record("ref_kokoro_pt.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["kokoro", "misaki", "torch", "transformers", "spacy", "numpy"])  # read before the imports: a failed import keeps them
        import numpy as np
        import torch
        from kokoro import KModel, KPipeline
        rows = al.clips(args.set, args.fixtures, args.first)
        wavdir = pathlib.Path(args.out).parent / "wav"
        wavdir.mkdir(parents=True, exist_ok=True)
        md = pathlib.Path(args.model_dir)
        t = time.perf_counter()
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(md / "config.json"), model=str(md / "kokoro-v1_0.pth")).eval()
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
        voice = str(md / "voices" / "af_heart.pt")
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["workload"] = {"set": args.set, "voice": "af_heart", "lang_code": "a (en-us)", "speed": 1.0,
                           "g2p": "misaki en (KPipeline)", "dtype": "float32", "device": "cpu"}
        for r in rows:
            t0 = time.perf_counter()
            parts = []
            with torch.inference_mode():
                for _, _, audio in pipe(r["ref"], voice=voice, speed=1):
                    if audio is not None:
                        parts.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
            wall = time.perf_counter() - t0
            data = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
            path = wavdir / f"{r['id']}.wav"
            write_wav(path, data, 24000)
            rec.row({"id": r["id"], "text_in": r["ref"], "wall_s": wall, "audio_s": len(data) / 24000, "rate": 24000,
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
