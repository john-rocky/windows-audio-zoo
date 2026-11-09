#!/usr/bin/env python3
"""Pin every Hub file the Z3 cells download: repo, revision (pinned below, read from the Hub API on 2026-11-09), path, size,
sha256 (lfs.oid, the Hub's own hash of the bytes) or the git blob sha1 for small files. Writes shelf/hub_manifest.json
(pc/fetch.py reads it on the PC).

    python3 tools/shelf_manifest.py
"""
import fnmatch
import json
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]
# (repo, revision, include globs; empty = the whole repo except README / .gitattributes)
SOURCES = [
    # Whisper large-v3-turbo: OpenVINO's three IRs (ASRPipeline), the checkpoint (ORT GenAI's builder, the transformers reference)
    ("OpenVINO/whisper-large-v3-turbo-fp16-ov", "0250c28d68c7c10d6b5cb39707e876c0c66ab6f8", []),
    ("OpenVINO/whisper-large-v3-turbo-int8-ov", "b568445dd5dc8c695bde596f8acbb4694fd6ba64", []),
    ("OpenVINO/whisper-large-v3-turbo-int4-ov", "ae50b4d9a9dbaf16f2df59c23f3984e42f864dfc", []),
    ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
     ["config.json", "generation_config.json", "model.safetensors", "*token*", "vocab.json", "merges.txt",
      "normalizer.json", "preprocessor_config.json", "added_tokens.json", "special_tokens_map.json"]),
    # Qwen3-ASR-1.7B: the qwen-asr checkpoint (optimum-intel 2.2.0's export loads it through qwen-asr 0.0.6 with
    # transformers 4.57.6; the -hf form is transformers 5's), the LiteRT-LM file (30 s window)
    ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", []),
    ("litert-community/Qwen3-ASR-1.7B", "2ad21a0bf38433cc96eb9aa54b95c5d70d7cbbe9", ["Qwen3-ASR-1.7B.litertlm"]),
    # Granite Speech 5.0 2B: the LiteRT-LM file
    ("litert-community/granite-speech-5.0-2b", "c344fa2ac34eacb943906a9b1959f389f76a61f1",
     ["granite-speech-5.0-2b.litertlm", "litertlm_manifest.json"]),
    # Parakeet TDT 0.6B v3: the ONNX export (fp32 and int8), the checkpoint (transformers reference)
    ("istupakov/parakeet-tdt-0.6b-v3-onnx", "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce", []),
    ("nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
     ["config.json", "generation_config.json", "model.safetensors", "processor_config.json", "tokenizer.json",
      "tokenizer_config.json"]),
    # Kokoro-82M: OpenVINO's two IRs (Text2SpeechPipeline), onnx-community's ONNX (kokoro-onnx form), the checkpoint (reference)
    ("OpenVINO/Kokoro-82M-fp16-ov", "787375b6f2bea4e25555bc6686a43fe52083713a", []),
    ("OpenVINO/Kokoro-82M-int8-ov", "9c035d0bfab136f0c1c525a5841d3411d7220c66", []),
    ("onnx-community/Kokoro-82M-v1.0-ONNX", "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
     ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model.onnx", "onnx/model_fp16.onnx", "voices/af_heart.bin"]),
    ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987", ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"]),
]


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    out, total = [], 0
    for repo, sha, inc in SOURCES:
        info = get(f"https://huggingface.co/api/models/{repo}")
        tree = get(f"https://huggingface.co/api/models/{repo}/tree/{sha}?recursive=true")
        files = []
        for f in tree:
            p = f["path"]
            if f["type"] != "file" or p == ".gitattributes" or p.endswith("README.md"):
                continue
            if inc and not any(fnmatch.fnmatch(p, g) for g in inc):
                continue
            lfs = f.get("lfs") or {}
            files.append({"path": p, "size": f["size"], "sha256": lfs.get("oid"), "blob": f.get("oid")})
        missing = [g for g in inc if not any(fnmatch.fnmatch(x["path"], g) for x in files)]
        if missing:
            sys.exit(f"STOP: {repo}@{sha[:10]} has no file for {missing}")
        size = sum(x["size"] for x in files)
        total += size
        out.append({"repo": repo, "folder": "", "revision": sha, "license": (info.get("cardData") or {}).get("license"),
                    "main_now": info.get("sha"), "bytes": size, "files": files})
        print(f"{repo:42s} {sha[:10]} {size/2**30:6.2f} GiB {len(files):3d} files"
              + ("" if info.get("sha") == sha else f"  (main is now {info.get('sha', '')[:10]})"))
    (HERE / "shelf").mkdir(exist_ok=True)
    (HERE / "shelf" / "hub_manifest.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"total {total/2**30:.2f} GiB")


if __name__ == "__main__":
    sys.exit(main())
