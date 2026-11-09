"""The PyTorch references (PLAN section 1): transformers fp32 on the PC's CPU, one clip at a time, greedy; one JSON in the
runners' form (rows with text). Not a timed cell: it runs under the shared PC's yield wrapper.

    python ref_hf.py --family whisper --model-dir <openai/whisper-large-v3-turbo folder> --set S150 --fixtures <dir> --out ref.json
    python ref_hf.py --family parakeet --model-dir <nvidia/parakeet-tdt-0.6b-v3 folder> --set S150 --langs en_us ...

Whisper: WhisperForConditionalGeneration.generate(input_features, language=<clip's>, task="transcribe",
max_new_tokens=440, num_beams=1, do_sample=False, return_timestamps=False), eager attention.
Parakeet: AutoModelForTDT.generate(**processor(samples)) and processor.decode(skip_special_tokens=True) (the
transformers model card's form for nvidia/parakeet-tdt-0.6b-v3).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--family", required=True, choices=["whisper", "parakeet"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--langs", default="")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    rec = al.Record("ref_hf.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["torch", "transformers", "numpy"])  # read before the imports: a failed import keeps them
        import torch
        import transformers
        if args.threads:
            torch.set_num_threads(args.threads)
        rec["torch_threads"] = torch.get_num_threads()
        rows = al.clips(args.set, args.fixtures, args.first)
        if args.langs:
            rows = [r for r in rows if r["lang"] in set(args.langs.split(","))]
        t = time.perf_counter()
        if args.family == "whisper":
            proc = transformers.WhisperProcessor.from_pretrained(args.model_dir)
            model = transformers.WhisperForConditionalGeneration.from_pretrained(
                args.model_dir, dtype=torch.float32, attn_implementation="eager").eval()
        else:
            proc = transformers.AutoProcessor.from_pretrained(args.model_dir)
            model = transformers.AutoModelForTDT.from_pretrained(args.model_dir, dtype=torch.float32).eval()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["workload"] = {"set": args.set, "first": args.first, "dtype": "float32", "device": "cpu", "decode": "greedy",
                           "max_new_tokens": 440 if args.family == "whisper" else None,
                           "language": "fixed per clip" if args.family == "whisper" else None}
        for r in rows:
            x = al.read_wav(r["path"])
            t0 = time.perf_counter()
            with torch.inference_mode():
                if args.family == "whisper":
                    feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features
                    lang = al.LANG[r["lang"]][0] if r["lang"] in al.LANG else al.S220_LANG[r["lang"]]
                    ids = model.generate(feats, language=lang, task="transcribe", max_new_tokens=440, num_beams=1,
                                         do_sample=False, return_timestamps=False)
                    seq = ids[0].tolist()
                    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
                else:
                    inputs = proc([x], sampling_rate=16000, return_tensors="pt")
                    out = model.generate(**inputs, return_dict_in_generate=True)
                    seq = out.sequences[0].tolist()
                    text = proc.decode(out.sequences, skip_special_tokens=True)
                    text = text[0] if isinstance(text, list) else text
            wall = time.perf_counter() - t0
            rec.row({"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall, "text": text.strip(),
                     "gen_ids": seq})
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
