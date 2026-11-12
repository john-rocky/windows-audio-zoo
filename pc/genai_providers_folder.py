"""A folder that runs an ONNX Runtime GenAI Whisper export on one execution provider with no provider set in code:
hard links to the export's files and a copy of its genai_config.json whose encoder and decoder entries both name the
provider. GenAI 0.19.0 takes a Whisper encoder's provider from this file only (pc/run_asr_genai.py's docstring).

    python genai_providers_folder.py --src <export> --dst <folder> --provider OpenVINO [--option device_type=CPU ...]
        [--encoder-option key=value ...] [--decoder-option key=value ...]

--option goes to both entries; --encoder-option and --decoder-option go to one entry only, on top of --option (a
device for one part, or the OpenVINO EP's reshape_input, the input shapes of that part's model).

The copy is the export's file with its two "provider_options" values replaced in the text (the builder writes each
on one line), so every other byte stays the builder's; the result is read back as JSON and must equal the export's
config with those two values changed, else nothing is written (exit 1). The folder runs with pc/run_asr_genai.py
--ep config. One JSON line on stdout: the folder, its files, the provider and each entry's options.
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the builder's export folder")
    ap.add_argument("--dst", required=True, help="the new folder (replaced if it exists)")
    ap.add_argument("--provider", required=True, help="the provider's name in genai_config.json, e.g. OpenVINO")
    ap.add_argument("--option", action="append", default=[], help="key=value for both entries, e.g. device_type=CPU")
    ap.add_argument("--encoder-option", action="append", default=[], help="key=value for the encoder's entry only")
    ap.add_argument("--decoder-option", action="append", default=[], help="key=value for the decoder's entry only")
    a = ap.parse_args()
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    text = (src / "genai_config.json").read_bytes().decode("utf-8")   # bytes: the builder's line endings stay (CRLF)
    cfg = json.loads(text)
    both = dict(o.split("=", 1) for o in a.option)
    opts = {"encoder": {**both, **dict(o.split("=", 1) for o in a.encoder_option)},
            "decoder": {**both, **dict(o.split("=", 1) for o in a.decoder_option)}}
    want = json.loads(text)
    for part in ("encoder", "decoder"):
        want["model"][part]["session_options"]["provider_options"] = [{a.provider: opts[part]}]
    new = None
    for order in (("encoder", "decoder"), ("decoder", "encoder")):   # which entry's line comes first in the file
        seen = []

        def put(m, order=order, seen=seen):
            seen.append(m)
            part = order[len(seen) - 1] if len(seen) <= 2 else None
            return m.group(1) + json.dumps([{a.provider: opts[part]}]) if part else m.group(0)

        cand, n = re.subn(r'("provider_options":\s*)\[[^\n]*\]', put, text)
        if n == 2 and json.loads(cand) == want:
            new = cand
            break
    if new is None:
        sys.exit(f"genai_config.json: {n} provider_options lines replaced, or the result is not the export's config with "
                 f"the encoder's and the decoder's provider set: nothing written")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name != "genai_config.json":
            os.link(p, dst / p.name)
    (dst / "genai_config.json").write_bytes(new.encode("utf-8"))
    print(json.dumps({"dst": str(dst), "files": sorted(p.name for p in dst.iterdir()), "provider": a.provider,
                      "encoder_options": opts["encoder"], "decoder_options": opts["decoder"],
                      "encoder_before": cfg["model"]["encoder"]["session_options"].get("provider_options"),
                      "decoder_before": cfg["model"]["decoder"]["session_options"].get("provider_options")}))


if __name__ == "__main__":
    main()
