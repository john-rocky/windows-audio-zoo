"""The second step of placing the Qwen3-ASR OpenVINO cells' gap (after pc/ref_qwen_asr_pkg.py): the exported IR (the
folder the OpenVINO GenAI cells load) through optimum-intel's own Python class, OVModelForSpeechSeq2Seq, fed the
qwen-asr package's processor inputs (its chat template with empty context and no forced language, input_features,
feature_attention_mask), the form of optimum-intel 2.2.0's tests/openvino/test_asr.py (test_compare_to_transformers:
ov_config INFERENCE_PRECISION_HINT f32 on CPU); greedy, max_new_tokens 512; decoded and parsed as the package does.
Not a timed cell.

    python ref_qwen_asr_ovpy.py --ir-dir <exports\\qwen3asr-ov-fp16> --model-dir <models\\Qwen__Qwen3-ASR-1.7B>
        --set S150 --first 10 --fixtures <dir> --out <json> [--precision f32|default]

Same text as the package's PyTorch run = the IR holds the model and the gap is in OpenVINO GenAI's pipeline; same as
the OpenVINO GenAI cells = the gap is in the IR (the export, or its fp16 / int8 weights).
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
    ap.add_argument("--ir-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="the checkpoint folder: the package's processor")
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--precision", default="f32", choices=["f32", "default"])
    args = ap.parse_args()
    rec = al.Record("ref_qwen_asr_ovpy.py", args)
    al.banner()
    code = 0
    try:
        rec["versions"] = al.versions(["optimum-intel", "openvino", "qwen-asr", "transformers", "torch", "numpy"])
        import openvino as ov
        import qwen_asr.inference.qwen3_asr  # noqa: F401  (registers qwen3_asr with AutoConfig / AutoModel / AutoProcessor)
        from optimum.intel import OVModelForSpeechSeq2Seq
        from qwen_asr.inference.utils import parse_asr_output
        from transformers import AutoProcessor
        rec["device_default_precision"] = str(ov.Core().get_property(args.device, "INFERENCE_PRECISION_HINT"))
        rows = al.clips(args.set, args.fixtures, args.first)
        t = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_dir, fix_mistral_regex=True)
        ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if args.precision == "f32" else {}
        model = OVModelForSpeechSeq2Seq.from_pretrained(args.ir_dir, device=args.device, ov_config=ov_config,
                                                        trust_remote_code=True)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        prompt = processor.apply_chat_template([{"role": "system", "content": ""},
                                                {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
                                               add_generation_prompt=True, tokenize=False)
        rec["prompt"] = prompt
        rec["workload"] = {"set": args.set, "first": args.first, "device": args.device, "ov_config": ov_config,
                           "decode": "greedy", "max_new_tokens": 512, "language": "none (auto)"}
        for r in rows:
            x = al.read_wav(r["path"])
            t0 = time.perf_counter()
            inputs = processor(text=[prompt], audio=[x], return_tensors="pt", padding=True)
            ids = model.generate(input_features=inputs["input_features"], decoder_input_ids=inputs["input_ids"],
                                 attention_mask=inputs["feature_attention_mask"], max_new_tokens=512, do_sample=False,
                                 num_beams=1)
            ids = ids.sequences if hasattr(ids, "sequences") else ids
            new = ids[:, inputs["input_ids"].shape[1]:]
            raw = processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
