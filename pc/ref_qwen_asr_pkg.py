"""A second Qwen3-ASR reference, to place the OpenVINO cells' gap from the reference of earlier Qwen3-ASR work: the qwen-asr package (0.0.6)
with the checkpoint optimum-intel exports (`Qwen/Qwen3-ASR-1.7B`, not the transformers `-hf` one that reference
used), PyTorch fp32 on the PC's CPU, one clip at a time, the package's own prompt (`_build_text_prompt`: empty
context, no forced language) and processor, greedy (do_sample=False, num_beams=1), max_new_tokens 512, the output
parsed by the package's `parse_asr_output`. Not a timed cell.

    python ref_qwen_asr_pkg.py --model-dir <Qwen/Qwen3-ASR-1.7B folder> --set S150 --first 10 --fixtures <dir> --out <json>

Same errors as the OpenVINO cells = the gap is in this checkpoint and its processor; same as that reference = the
gap comes from the export or the runtime.
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
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    rec = al.Record("ref_qwen_asr_pkg.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["qwen-asr", "torch", "transformers", "numpy"])  # read before the imports
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwen_asr.inference.utils import parse_asr_output
        if args.threads:
            torch.set_num_threads(args.threads)
        rec["torch_threads"] = torch.get_num_threads()
        rows = al.clips(args.set, args.fixtures, args.first)
        t = time.perf_counter()
        m = Qwen3ASRModel.from_pretrained(args.model_dir, dtype=torch.float32, max_inference_batch_size=1,
                                          max_new_tokens=512)
        m.model.eval()
        rec["load_s"] = round(time.perf_counter() - t, 3)
        prompt = m._build_text_prompt(context="", force_language=None)
        rec["prompt"] = prompt
        rec["workload"] = {"set": args.set, "first": args.first, "dtype": str(m.model.dtype), "device": "cpu",
                           "decode": "greedy", "max_new_tokens": 512, "language": "none (auto)"}
        for r in rows:
            x = al.read_wav(r["path"])
            t0 = time.perf_counter()
            with torch.inference_mode():
                inputs = m.processor(text=[prompt], audio=[x], return_tensors="pt", padding=True)
                inputs = inputs.to(m.model.device).to(m.model.dtype)
                out = m.model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
                seqs = out.sequences if hasattr(out, "sequences") else out
                new = seqs[:, inputs["input_ids"].shape[1]:]
                raw = m.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            lang, text = parse_asr_output(raw, user_language=None)
            wall = time.perf_counter() - t0
            rec.row({"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall, "text": text.strip(),
                     "language": lang, "raw": raw, "gen_ids": new[0].tolist()})
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
