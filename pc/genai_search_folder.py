"""A folder that runs an ONNX Runtime GenAI export with one value of its genai_config.json's "search" section changed:
hard links to the export's files and a copy of its genai_config.json with that one value replaced in the text, so
every other byte stays the builder's.

    python genai_search_folder.py --src <export> --dst <folder> --set past_present_share_buffer=true

The value is read as JSON (true, 448, ...). The key must appear once in the file; the copy is read back as JSON and
must equal the export's config with that one value changed, else nothing is written (exit 1). One JSON line on
stdout: the folder, its files and the change.
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
    ap.add_argument("--set", required=True, help="key=value of the search section, the value as JSON")
    a = ap.parse_args()
    key, raw = a.set.split("=", 1)
    value = json.loads(raw)
    src, dst = pathlib.Path(a.src), pathlib.Path(a.dst)
    text = (src / "genai_config.json").read_bytes().decode("utf-8")   # bytes: the builder's line endings stay (CRLF)
    cfg = json.loads(text)
    if key not in cfg.get("search", {}):
        sys.exit(f"genai_config.json: no {key} in the search section: nothing written")
    want = json.loads(text)
    want["search"][key] = value
    new, n = re.subn(r'("' + re.escape(key) + r'":\s*)([^,\r\n}]+)', lambda m: m.group(1) + json.dumps(value), text)
    if n != 1 or json.loads(new) != want:
        sys.exit(f"genai_config.json: {n} {key} values replaced, or the result is not the export's config with that "
                 f"one value changed: nothing written")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.is_file() and p.name != "genai_config.json":
            os.link(p, dst / p.name)
    (dst / "genai_config.json").write_bytes(new.encode("utf-8"))
    print(json.dumps({"dst": str(dst), "files": sorted(p.name for p in dst.iterdir()),
                      "search": {key: value}, "before": {key: cfg["search"][key]}}))


if __name__ == "__main__":
    main()
