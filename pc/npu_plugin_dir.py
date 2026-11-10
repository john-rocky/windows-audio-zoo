"""A folder for LiteRT-LM's NPU backend (run_asr_litertlm.py --npu-lib-dir) that holds an Intel OpenVINO compiler
plugin beside a dispatch library.

LiteRT-LM 0.19.1 hands LiteRT one folder as both the dispatch library and the compiler plugin directory
(runtime/util/litert_util.cc, PopulateNpuOptions), and its Windows wheel ships only the dispatch library
(litert_lm\\vendors\\intel_openvino\\dispatch\\LiteRtDispatch.dll). LiteRT's Windows finder takes any
*LiteRtCompilerPlugin*.dll in that folder (litert/core/dynamic_loading_windows.cc). The plugin here comes from the
ai-edge-litert-nightly wheel 2.4.0.dev20261024 for Python 3.12 on Windows: its ai-edge-litert-sdk-intel-nightly of the
same date pins openvino==2026.4.1 (the version in this repo's LiteRT-LM venv), and LiteRT-LM 0.19.1 builds against
LiteRT dcb657e8a5 of 2026-10-24.

    python npu_plugin_dir.py --out <dir> --dispatch litertlm|nightly [--wheels <dir>]

--dispatch litertlm: LiteRT-LM's own dispatch library (built with the same LiteRT as its runtime) beside the plugin;
nightly: the dispatch library from the same nightly wheel as the plugin. The wheel is downloaded once into --wheels
(default <out>\\..\\wheels) and checked against PyPI's sha256. Writes <out>\\manifest.json (each file: where it came
from, bytes, sha256).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import urllib.request
import zipfile

WHEEL = "ai_edge_litert_nightly-2.4.0.dev20261024-cp312-cp312-win_amd64.whl"
WHEEL_URL = ("https://files.pythonhosted.org/packages/4f/07/a5502da4a4a265ee172aa3334bd329a19b8e993d7628383d888158b16d58/"
             + WHEEL)
WHEEL_SHA256 = "d46091d1111f4168ef4fc9dcfedcc65efc269e09b23f37b3fed32d981afa7815"
PLUGIN_IN_WHEEL = "ai_edge_litert/vendors/intel_openvino/compiler/LiteRtCompilerPlugin.dll"
DISPATCH_IN_WHEEL = "ai_edge_litert/vendors/intel_openvino/dispatch/LiteRtDispatch.dll"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wheel_path(wheels):
    os.makedirs(wheels, exist_ok=True)
    p = os.path.join(wheels, WHEEL)
    if not os.path.exists(p) or sha256(p) != WHEEL_SHA256:
        tmp = p + ".part"
        with urllib.request.urlopen(WHEEL_URL, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256(tmp)
        if got != WHEEL_SHA256:
            os.remove(tmp)
            raise SystemExit(f"FAIL {WHEEL}: sha256 {got}, PyPI says {WHEEL_SHA256}")
        os.replace(tmp, p)
    return p


def litertlm_dispatch():
    spec = importlib.util.find_spec("litert_lm")   # the package's folder, without importing it
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("FAIL litert_lm is not installed in this interpreter")
    p = os.path.join(list(spec.submodule_search_locations)[0], "vendors", "intel_openvino", "dispatch",
                     "LiteRtDispatch.dll")
    if not os.path.exists(p):
        raise SystemExit(f"FAIL no dispatch library at {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dispatch", required=True, choices=["litertlm", "nightly"])
    ap.add_argument("--wheels", default="")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    wheels = args.wheels or os.path.join(os.path.dirname(out), "wheels")
    whl = wheel_path(wheels)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    files = []
    with zipfile.ZipFile(whl) as z:
        names = [PLUGIN_IN_WHEEL] + ([DISPATCH_IN_WHEEL] if args.dispatch == "nightly" else [])
        for n in names:
            dst = os.path.join(out, os.path.basename(n))
            with z.open(n) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            files.append({"name": os.path.basename(n), "from": f"{WHEEL}:{n}", "wheel_sha256": WHEEL_SHA256})
    if args.dispatch == "litertlm":
        src = litertlm_dispatch()
        shutil.copy2(src, os.path.join(out, "LiteRtDispatch.dll"))
        files.append({"name": "LiteRtDispatch.dll", "from": src})
    for f in files:
        p = os.path.join(out, f["name"])
        f["bytes"] = os.path.getsize(p)
        f["sha256"] = sha256(p)
    man = {"dispatch": args.dispatch, "python": sys.version.split()[0], "files": files}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    for f in files:
        print(f"OK {f['name']} {f['bytes']} {f['sha256']} <- {f['from']}")


if __name__ == "__main__":
    main()
