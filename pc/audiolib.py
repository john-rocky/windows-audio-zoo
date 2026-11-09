"""Shared parts of the Z3 runners (one engine process per cell; pc/chain.ps1 starts it and watches the pid it prints).

Clip sets (fixtures/, rebuilt by tools/fetch_clips.py), read from <fixtures>:
  S150  fleurs/<config>/<id>.wav + meta.json      FLEURS test en_us / cmn_hans_cn / ja_jp, 50 each (the set of earlier Qwen3-ASR work)
  S220  s220/clips/<id>.wav + s220/clips.json      the set of earlier Granite Speech work (en-LS 20, en-FL 50, ja 50, fr / de / es / pt 25)
  T50   the en_us 50 of S150, as text (raw_transcription) for the TTS cells
A runner reads every WAV before its timed loop (PCM16 mono 16 kHz -> float32 in [-1, 1]); the timed span of a clip
starts when the runtime gets the samples (or the path, for a runtime that only takes a path) and ends with its text.
"""
import os

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # before any onnxruntime import

import ctypes  # noqa: E402
import ctypes.wintypes  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402

# FLEURS config -> (Whisper language code, Qwen3-ASR language name)
LANG = {"en_us": ("en", "English"), "cmn_hans_cn": ("zh", "Chinese"), "ja_jp": ("ja", "Japanese")}
# S220 set -> Whisper code (Granite needs none; kept for the record)
S220_LANG = {"en_ls": "en", "en_fl": "en", "ja": "ja", "fr": "fr", "de": "de", "es": "es", "pt": "pt"}


def utf8_stdout():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def now():
    return dt.datetime.now().astimezone().isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clips(set_name, fixtures, first=0):
    """The rows of one clip set: id, lang, path, seconds, ref (reference text), sha256. first = N per language."""
    fx = pathlib.Path(fixtures)
    rows = []
    if set_name in ("S150", "T50"):
        meta = json.loads((fx / "meta.json").read_text(encoding="utf-8"))
        for c in meta["clips"]:
            if not c["path"].startswith("fleurs/"):
                continue
            lang = c["source"]["config"]
            if set_name == "T50" and lang != "en_us":
                continue
            rows.append({"id": c["name"], "lang": lang, "path": str(fx / c["path"]), "seconds": float(c["duration_s"]),
                         "sha256": c["sha256"], "ref": c["reference_text"]["raw_transcription"]})
    elif set_name == "S220":
        meta = json.loads((fx / "s220" / "clips.json").read_text(encoding="utf-8"))
        for c in meta["clips"]:
            rows.append({"id": c["id"], "lang": c["set"], "path": str(fx / "s220" / "clips" / f"{c['id']}.wav"),
                         "seconds": float(c["seconds"]), "sha256": c["sha256"], "ref": c["ref"]})
    else:
        raise ValueError(f"unknown clip set {set_name}")
    if first:
        seen, keep = {}, []
        for r in rows:
            seen[r["lang"]] = seen.get(r["lang"], 0) + 1
            if seen[r["lang"]] <= first:
                keep.append(r)
        rows = keep
    return rows


def warmup_wav(fixtures):
    """The untimed warm-up clip (not scored): the en example of the fixture folder (Fun-ASR-Nano's example, 16 kHz)."""
    return str(pathlib.Path(fixtures) / "examples" / "en.wav")


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 16000:
            raise ValueError(f"{path}: want PCM16 mono 16 kHz, got width {w.getsampwidth()} ch {w.getnchannels()} "
                             f"rate {w.getframerate()}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0)


def peak_working_set():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    if os.name != "nt":
        return {"peak_working_set_bytes": None, "peak_pagefile_bytes": None}
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    gpmi = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    gpmi.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    gpmi.restype = ctypes.wintypes.BOOL
    try:
        ok = gpmi(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    except Exception:
        ok = False
    return {"peak_working_set_bytes": pmc.PeakWorkingSetSize if ok else None,
            "peak_pagefile_bytes": pmc.PeakPagefileUsage if ok else None}


def versions(names):
    out = {"python": sys.version.split()[0]}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def add_cuda_dirs():
    """The CUDA wheels keep their DLLs under site-packages\\nvidia\\...\\bin, which is not on PATH."""
    sp = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not sp.is_dir():
        return []
    dirs = sorted({str(p.parent) for p in sp.rglob("*.dll")})
    for d in dirs:
        os.add_dll_directory(d)
    os.environ["PATH"] = ";".join(dirs) + ";" + os.environ.get("PATH", "")
    return dirs


class _PPTS(ctypes.Structure):  # PROCESS_POWER_THROTTLING_STATE
    _fields_ = [("Version", ctypes.wintypes.ULONG), ("ControlMask", ctypes.wintypes.ULONG),
                ("StateMask", ctypes.wintypes.ULONG)]


def high_qos():
    """Take this process out of Windows' execution-speed power throttling (High QoS). A process started from ssh / WMI
    with no window runs throttled from about 3 s after its creation, its threads on the efficiency cores: on this PC
    GenAI's decode 1.9x (CUDA EP) / 2.0x (Intel GPU) slower and a fixed Python loop 2-3x slower
    (an earlier finding on this PC, and the same call it used). Every runner calls it first (Record.__init__), so
    every timed number on this PC is taken out of that state; the run JSON says what was set."""
    if os.name != "nt":
        return None
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_int, ctypes.POINTER(_PPTS),
                                          ctypes.wintypes.DWORD]
    k32.SetProcessInformation.restype = ctypes.wintypes.BOOL
    st = _PPTS(1, 0x1, 0x0)   # version 1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED in the mask, off in the state
    ok = bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))  # ProcessPowerThrottling
    return {"execution_speed_throttling": "off (High QoS)", "set_ok": ok, "error": 0 if ok else ctypes.get_last_error(),
            "s_after_audiolib_import": round(time.perf_counter() - _T0, 3)}


_T0 = time.perf_counter()   # module import, close to the process start


def banner():
    print(f"ENGINE_PID {os.getpid()}", flush=True)  # chain.ps1 watches this pid (the venv python.exe is a launcher)


class Record:
    """One cell run's JSON: header, per-clip rows, summary; written on every exit path (status OK / ERROR)."""

    def __init__(self, runner, args):
        qos = high_qos()   # first: before the runtime's import and the model's load
        self.rec = {"runner": runner, "started": now(), "argv": sys.argv[1:], "pid": os.getpid(), "status": "ERROR",
                    "power_throttling": qos, "rows": []}
        self.out = pathlib.Path(args.out)
        self.t0 = time.perf_counter()

    def __setitem__(self, k, v):
        self.rec[k] = v

    def __getitem__(self, k):
        return self.rec[k]

    def row(self, r):
        self.rec["rows"].append(r)
        n = len(self.rec["rows"])
        if n % 10 == 0 or n <= 2:
            print(json.dumps({"n": n, "id": r.get("id"), "wall_s": round(r.get("wall_s") or 0, 3),
                              "text": (r.get("text") or "")[:60]}, ensure_ascii=False), flush=True)

    def summarize(self, kind="asr"):
        rows = [r for r in self.rec["rows"] if r.get("wall_s") is not None]
        s = {"clips": len(rows)}
        if kind == "asr":
            audio = sum(r["seconds"] for r in rows)
            wall = sum(r["wall_s"] for r in rows)
            s.update({"audio_s": round(audio, 3), "wall_s": round(wall, 3), "rtf": round(wall / audio, 5) if audio else None,
                      "empty": sum(1 for r in rows if not (r.get("text") or "").strip())})
            ft = [r["first_token_ms"] for r in rows if r.get("first_token_ms") is not None]
            s["first_token_ms_mean"] = round(sum(ft) / len(ft), 2) if ft else None
            odd = [r for i, r in enumerate(rows) if i % 2 == 0]
            even = [r for i, r in enumerate(rows) if i % 2 == 1]
            if odd and even:
                ro = sum(r["wall_s"] for r in odd) / sum(r["seconds"] for r in odd)
                re_ = sum(r["wall_s"] for r in even) / sum(r["seconds"] for r in even)
                s["split_half_rtf"] = [round(ro, 5), round(re_, 5)]
        else:  # tts
            audio = sum(r.get("audio_s") or 0 for r in rows)
            wall = sum(r["wall_s"] for r in rows)
            s.update({"audio_s": round(audio, 3), "wall_s": round(wall, 3), "rtf": round(wall / audio, 5) if audio else None,
                      "silent": sum(1 for r in rows if (r.get("audio_s") or 0) < 0.2 or r.get("all_silent"))})
        self.rec["summary"] = s
        return s

    def fail(self):
        self.rec["status"] = "ERROR"
        self.rec["error"] = traceback.format_exc()
        print(self.rec["error"], file=sys.stderr, flush=True)

    def write(self):
        self.rec.update(peak_working_set())
        self.rec["ended"] = now()
        self.rec["process_s"] = round(time.perf_counter() - self.t0, 3)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.rec, indent=1, ensure_ascii=False), encoding="utf-8")
        sys.stdout.flush()
        sys.stderr.flush()


def base_args(ap):
    ap.add_argument("--set", required=True, choices=["S150", "S220", "T50"])
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--first", type=int, default=0, help="only the first N clips per language (control rows, probes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-warmup", action="store_true")
    return ap
