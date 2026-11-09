"""One engine process of an OpenVINO GenAI ASR cell: ASRPipeline (Whisper or Qwen3-ASR) on one device, one pass over a
clip set after one untimed warm-up clip; one JSON.

    python run_asr_ovg.py --model-dir <IR folder> --family whisper|qwen3asr --device CPU|GPU.0|GPU.1|NPU
        --set S150 --fixtures <dir> --out run.json [--first N] [--cache-dir <dir>] [--max-new 440]

Whisper: language fixed per clip (<|en|> / <|zh|> / <|ja|>), task transcribe, no timestamps (PLAN section 2).
Qwen3-ASR: no language (the pipeline's own prompt = the official prompt of earlier Qwen3-ASR work); the text is the part
after <asr_text>, the language tag is kept in the row. Greedy (the pipeline's default, num_beams 1).
Timing per clip: pipe.generate(samples) wall clock; first token = perf_metrics.get_ttft().
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


def metric(pm, name):
    try:
        v = getattr(pm, name)()
        return {"mean": float(v.mean), "std": float(v.std)} if hasattr(v, "mean") else float(v)
    except Exception as e:  # not every metric exists for every model family
        return f"n/a ({type(e).__name__})"


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--family", required=True, choices=["whisper", "qwen3asr"])
    ap.add_argument("--device", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--max-new", type=int, default=0, help="0 = 440 for Whisper, 512 for Qwen3-ASR")
    args = ap.parse_args()
    max_new = args.max_new or (440 if args.family == "whisper" else 512)
    rec = al.Record("run_asr_ovg.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["openvino-genai", "openvino", "openvino-tokenizers", "numpy"])  # read before the imports: a failed import keeps them
        import openvino_genai as ov_genai
        rows = al.clips(args.set, args.fixtures, args.first)
        audio = {r["id"]: al.read_wav(r["path"]).tolist() for r in rows}  # read before the timed loop
        props = {}
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            props["CACHE_DIR"] = args.cache_dir
        t = time.perf_counter()
        pipe = ov_genai.ASRPipeline(args.model_dir, args.device, **props)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["workload"] = {"set": args.set, "first": args.first, "max_new_tokens": max_new, "decode": "greedy",
                           "language": "fixed per clip" if args.family == "whisper" else "none (auto)",
                           "task": "transcribe" if args.family == "whisper" else None, "timestamps": False}
        try:
            gc = pipe.get_generation_config()
            rec["generation_config"] = {k: str(getattr(gc, k)) for k in ("max_new_tokens", "num_beams", "do_sample",
                                                                          "language", "task", "return_timestamps")
                                        if hasattr(gc, k)}
        except Exception as e:
            rec["generation_config"] = f"n/a ({e})"

        def run(samples, lang):
            kw = {"max_new_tokens": max_new}
            if args.family == "whisper":
                kw.update(language=f"<|{al.LANG[lang][0]}|>" if lang in al.LANG else f"<|{al.S220_LANG[lang]}|>",
                          task="transcribe", return_timestamps=False)
            t0 = time.perf_counter()
            res = pipe.generate(samples, **kw)
            return res, time.perf_counter() - t0

        if not args.no_warmup:
            w = al.read_wav(al.warmup_wav(args.fixtures)).tolist()
            _, ws = run(w, "en_us")
            rec["warmup_s"] = round(ws, 3)
        for r in rows:
            res, wall = run(audio[r["id"]], r["lang"])
            pm = res.perf_metrics
            ttft = metric(pm, "get_ttft")
            row = {"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall,
                   "text": res.texts[0] if res.texts else "",
                   "first_token_ms": ttft["mean"] if isinstance(ttft, dict) else None,
                   "tokens": metric(pm, "get_num_generated_tokens"),
                   "tpot_ms": metric(pm, "get_tpot"), "generate_ms": metric(pm, "get_generate_duration")}
            langs = getattr(res, "languages", None)
            if langs:
                row["language"] = langs[0]
            rec.row(row)
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
