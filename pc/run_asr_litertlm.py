"""One engine process of a LiteRT-LM audio cell (litert-lm-api, the released Python API): a .litertlm with an audio
encoder (Qwen3-ASR, Granite Speech) on one backend, one pass over a clip set after one untimed warm-up clip; one JSON.

    python run_asr_litertlm.py --bundle <file.litertlm> --family qwen3asr|granite --backend cpu|gpu|npu
        [--audio-backend cpu|gpu|npu] [--gpu-pref 1|2 [--gpu-pref-names real|asis]] [--npu-lib-dir <dir>]
        --set S150|S220 --fixtures <dir> --out run.json [--first N]

Per clip: a new conversation (greedy: top_k 1, top_p 1.0, temperature 0), one user message with the clip as
Content.AudioFile(<path>) (the API reads the file: the timed span includes the read), the streamed reply
(send_message_async); first token = the first non-empty chunk. Max output tokens: Qwen3-ASR 512, Granite 256.
Qwen3-ASR: chat_template = the checkpoint's official template (the same prompt string as the earlier Qwen3-ASR rows and OpenVINO
GenAI), the language tag split off like qwen_asr's parse_asr_output; Granite: the bundle's own
template with the instruction text of earlier Granite Speech work after the audio.
GPU: the Intel GPU rows get Windows' per-app GPU preference (HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences,
"GpuPreference=1;" = power saving = the Intel GPU) from pc/chain.ps1, written before this process starts under the
paths the venv's launcher and its interpreter run by (the unit's "gpu_pref"); without a value LiteRT-LM 0.19.1's GPU
backend takes the RTX. --gpu-pref writes the same value from inside this process (before the runtime's import) and
removes it after: --gpu-pref-names asis names the interpreter as Python has it (uv's minor-version junction kept),
which reached the Intel GPU (11-10); the default "real" (os.path.realpath, the junction's target) is the name the
Intel GPU runs of 11-09/10 carried, and every one of them took the RTX. The venvs come from a Python that only this
repo uses, so no other job's python.exe shares the value. The run's log names the adapter ("Selected adapter: ...");
tools/results.py reads it and marks an Intel GPU cell that took another adapter WRONG DEVICE.
NPU: LiteRT-LM 0.19.1's NPU() (no argument) points LiteRT at <litert_lm>\\vendors\\intel_openvino\\dispatch\\, the one
folder it uses as both the dispatch library and the compiler plugin directory (runtime/util/litert_util.cc), and the
wheel ships only the dispatch library there. A .litertlm that packages no TF_LITE_AUX model (the auxiliary model
LiteRT-LM's specialized NPU executor reads; the litert-community Qwen3-ASR and Granite bundles) takes LiteRT's generic
compiler-plugin path, finds no plugin and runs on the CPU through XNNPACK (the log: "No compiler plugin found",
"XNNPACK delegate for CPU"; the NPU's memory stays 0).
--npu-lib-dir <dir> passes NPU(litert_dispatch_lib_dir=<dir>), a folder made by pc/npu_plugin_dir.py with a compiler
plugin beside a dispatch library, and puts the openvino package's libs folder on PATH the way NPU() does; the run's
JSON lists the folder's files with their sha256.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402

GRANITE_INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."
ASR_TEXT_TAG = "<asr_text>"
# the checkpoint's chat template for [system?, user(audio)] + generation prompt, message by message (the same text as
# the JINJA_OFFICIAL of an earlier Qwen3-ASR LiteRT-LM conversion)
JINJA_OFFICIAL = (
    "{%- set ns = namespace(has_system=false, prefix='') -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{%- set ns.has_system = true -%}"
    "{{- '<|im_start|>system\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'user' -%}"
    "{%- if not ns.has_system -%}{{- '<|im_start|>system\\n<|im_end|>\\n' -}}{%- set ns.has_system = true -%}"
    "{%- endif -%}"
    "{{- '<|im_start|>user\\n' -}}"
    "{%- set ns.prefix = '' -%}"
    "{%- if m.content is string -%}{%- set ns.prefix = m.content -%}{%- else -%}"
    "{%- for c in m.content -%}"
    "{%- if c.type == 'audio' -%}{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}"
    "{%- elif c.type == 'text' -%}{%- set ns.prefix = ns.prefix + c.text -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if m.content is string -%}{{- m.content -}}{%- else -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{{- c.text -}}{%- endif -%}{%- endfor -%}"
    "{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' + ns.prefix -}}{%- endif -%}"
)


def split_raw(raw):
    """qwen_asr's parse_asr_output for the no-forced-language case (the LiteRT lane's common.split_raw)."""
    s = (raw or "").strip()
    if ASR_TEXT_TAG not in s:
        return "", s
    meta, text = s.split(ASR_TEXT_TAG, 1)
    lang = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            lang = line[len("language "):].strip()
            break
    if "language none" in meta.lower():
        lang = ""
    return lang, text.strip()


GPUPREF_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"


def set_gpu_pref(value, names="real"):
    """Write GpuPreference=<value>; for this interpreter's exe (and the venv launcher); returns what was written.
    names "real": the paths through os.path.realpath (Z3's form: uv's minor-version junction resolved, so the name is
    ...\\cpython-3.12.16-...\\python.exe; the GPU backend stayed on the RTX); "asis": the paths as Python has them (the
    junction kept, ...\\cpython-3.12-...\\python.exe: the path the venv's launcher started the interpreter by, which
    Get-Process's .Path shows; with this name the GPU backend took the Intel GPU, Z3b 11-10).
    QueryFullProcessImageNameW gives the junction's target, the name that did not work."""
    import winreg
    raw = (getattr(sys, "_base_executable", sys.executable), sys.executable)
    exes = sorted({(os.path.realpath(p) if names == "real" else os.path.abspath(p)) for p in raw if p})
    k = winreg.CreateKey(winreg.HKEY_CURRENT_USER, GPUPREF_KEY)
    for e in exes:
        winreg.SetValueEx(k, e, 0, winreg.REG_SZ, f"GpuPreference={value};")
    back = {e: winreg.QueryValueEx(k, e)[0] for e in exes}
    winreg.CloseKey(k)
    return back


def _image_path():
    """QueryFullProcessImageNameW for this process: on this PC the junction's target (cpython-3.12.16-...), not the
    name a per-app GPU preference must carry (set_gpu_pref); recorded beside sys._base_executable."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD)]
    buf = ctypes.create_unicode_buffer(1024)
    n = wintypes.DWORD(1024)
    if not k32.QueryFullProcessImageNameW(k32.GetCurrentProcess(), 0, buf, ctypes.byref(n)):
        return None
    return buf.value


def clear_gpu_pref(exes):
    import winreg
    try:
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, GPUPREF_KEY, 0, winreg.KEY_SET_VALUE)
        for e in exes:
            try:
                winreg.DeleteValue(k, e)
            except FileNotFoundError:
                pass
        winreg.CloseKey(k)
    except FileNotFoundError:
        pass


def npu_lib_dir_record(path):
    """The folder handed to NPU(litert_dispatch_lib_dir=...): each file's name, size and sha256, and the openvino
    package's libs folder put on PATH (what NPU() does when it picks its own folder) with the devices OpenVINO sees."""
    import hashlib
    files = []
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                files.append({"name": name, "bytes": os.path.getsize(p), "sha256": hashlib.sha256(f.read()).hexdigest()})
    import json
    man = os.path.join(path, "manifest.json")   # pc/npu_plugin_dir.py: where each file came from
    manifest = None
    if os.path.isfile(man):
        with open(man, encoding="utf-8") as f:
            manifest = json.load(f)
    import openvino as ov
    libs = os.path.join(os.path.dirname(ov.__file__), "libs")
    cur = os.environ.get("PATH", "")
    if libs not in cur.split(os.pathsep):
        os.environ["PATH"] = f"{cur}{os.pathsep}{libs}" if cur else libs
    return {"path": os.path.abspath(path), "files": files, "manifest": manifest, "openvino_libs_on_path": libs,
            "openvino_devices": list(ov.Core().available_devices)}


def backend_of(interfaces, name, npu_lib_dir=""):
    if name == "cpu":
        return interfaces.CPU()
    if name == "gpu":
        return interfaces.GPU()
    if npu_lib_dir:
        return interfaces.NPU(litert_dispatch_lib_dir=os.path.abspath(npu_lib_dir))
    return interfaces.NPU()


def main():
    al.utf8_stdout()
    ap = al.base_args(argparse.ArgumentParser())
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--family", required=True, choices=["qwen3asr", "granite"])
    ap.add_argument("--backend", required=True, choices=["cpu", "gpu", "npu"])
    ap.add_argument("--audio-backend", default="", choices=["", "cpu", "gpu", "npu"])
    ap.add_argument("--gpu-pref", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--gpu-pref-names", default="real", choices=["real", "asis"])
    ap.add_argument("--npu-lib-dir", default="")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    max_new = 512 if args.family == "qwen3asr" else 256
    rec = al.Record("run_asr_litertlm.py", args)
    al.banner()
    code = 0
    pref = {}
    try:
        rec["versions"] = al.versions(["litert-lm-api", "openvino", "numpy"])  # read before the imports: a failed import keeps them
        rec["process_paths"] = {"sys._base_executable": getattr(sys, "_base_executable", None),
                                "sys.executable": sys.executable,
                                "QueryFullProcessImageNameW": _image_path() if os.name == "nt" else None}
        if args.gpu_pref:
            pref = set_gpu_pref(args.gpu_pref, args.gpu_pref_names)
            rec["gpu_preference_written"] = pref
        if args.npu_lib_dir:
            rec["npu_lib_dir"] = npu_lib_dir_record(args.npu_lib_dir)
        import litert_lm
        from litert_lm import interfaces
        rows = al.clips(args.set, args.fixtures, args.first)
        audio_backend = args.audio_backend or args.backend
        cache = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "lm_cache")
        os.makedirs(cache, exist_ok=True)
        t = time.perf_counter()
        eng = litert_lm.Engine(args.bundle, backend=backend_of(interfaces, args.backend, args.npu_lib_dir),
                               audio_backend=backend_of(interfaces, audio_backend, args.npu_lib_dir), cache_dir=cache)
        rec["load_s"] = round(time.perf_counter() - t, 3)
        rec["workload"] = {"set": args.set, "first": args.first, "max_new_tokens": max_new, "decode": "greedy (top_k 1)",
                           "backend": args.backend, "audio_backend": audio_backend, "cache_dir": cache,
                           "prompt": "official chat template" if args.family == "qwen3asr" else
                           f"bundle template, instruction after the audio: {GRANITE_INSTRUCTION!r}",
                           "input": "Content.AudioFile(path), read inside the timed span"}
        sampler = interfaces.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

        def run(path):
            kw = {"sampler_config": sampler, "max_output_tokens": max_new}
            if args.family == "qwen3asr":
                kw["chat_template"] = JINJA_OFFICIAL
            conv = eng.create_conversation(**kw)
            parts = [litert_lm.Content.AudioFile(os.path.abspath(path))]
            if args.family == "granite":
                parts.append(litert_lm.Content.Text(GRANITE_INSTRUCTION))
            msg = litert_lm.Message.user(litert_lm.Contents.of(*parts))
            render = conv.render_message_to_string(msg)
            t0 = time.perf_counter()
            first, chunks = None, []
            for m in conv.send_message_async(msg):
                s = str(m)
                if s and first is None:
                    first = time.perf_counter() - t0
                chunks.append(s)
            wall = time.perf_counter() - t0
            n = conv.token_count
            conv.close()
            return "".join(chunks), wall, first, n, render

        if not args.no_warmup:
            _, ws, _, _, render = run(al.warmup_wav(args.fixtures))
            rec["warmup_s"] = round(ws, 3)
            rec["render_warmup"] = render
        for r in rows:
            raw, wall, first, n, _ = run(r["path"])
            row = {"id": r["id"], "lang": r["lang"], "seconds": r["seconds"], "wall_s": wall,
                   "first_token_ms": first * 1000 if first is not None else None, "token_count": n, "raw": raw}
            if args.family == "qwen3asr":
                row["language"], row["text"] = split_raw(raw)
            else:
                row["text"] = raw.strip()
            rec.row(row)
        rec["summary"] = rec.summarize("asr")
        rec["status"] = "OK"
    except Exception:
        rec.fail()
        code = 1
    finally:
        if pref:
            clear_gpu_pref(list(pref))
    rec.write()
    os._exit(code)


if __name__ == "__main__":
    main()
