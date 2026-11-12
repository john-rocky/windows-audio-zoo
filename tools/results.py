#!/usr/bin/env python3
"""The Z3 results: every cell's run scored against its model's reference, one JSON per cell under results/, and the
table RESULTS_AUDIO_WINDOWS.md. Needs a Python with openai-whisper 20250625 (the S220 scorer's normalizers); reads
only this repo (fixtures/, raw/).

    python tools/results.py build --runs first-a,first-b,s1,s2 [--refs raw/refs]
    python tools/results.py table

Scores (PLAN.md section 1):
  S150  en_us WER on norm_text, cmn_hans_cn / ja_jp CER on norm_chars (tools/fleurs_score.py); line = each language
        <= the model's reference + 0.5 pt (counts, not rounded rates), 0 empty answers. References: Whisper / Parakeet
        = transformers fp32 on this PC's CPU (raw/refs/<model>/ref.json), Qwen3-ASR = transformers fp32
        (raw/refs/qwen3asr-fp32/ref_transformers_official.jsonl).
  S220  tools/s220_score.py (EnglishTextNormalizer / BasicTextNormalizer / ja chars); line = en_fl, ja and
        fr+de+es+pt each <= the fp32 reference (raw/refs/granite-fp32/R_fp32_cpu.jsonl) + 0.5 pt, 0 empty, 0 loop.
  T50   write-back WER (pc/writeback.py's text vs the sentence, norm_text); line = the PyTorch reference's write-back
        WER + 1.0 pt, 0 silent waves.
A run = <sitting>/<cell>/run1.json (the runner) + run1.meta.json (the chain). The row of a cell takes its latest timed
run with status OK, else its latest first-pass run (first 10 clips per language, marked "first pass"). A run that left
no runner JSON (the chain's time limit, a crash, a run stopped by hand) is still a row: its status comes from the
chain's record (TIMEOUT / CRASH / NO JSON) and its record path is the meta file. A LiteRT-LM GPU run names the adapter
it took in its log ("Selected adapter: ..."): an Intel GPU cell whose run took another adapter is WRONG DEVICE (its
scores stay in the JSON, the table shows none). A LiteRT-LM NPU run says in its log whether a compiler plugin was
loaded and in the chain's record how much NPU memory it held: no plugin, the XNNPACK CPU delegate and 0 MiB on the NPU
= the run went through the CPU (doc "ran_on": "cpu"; the table's device cell says so, the numbers stay).
"""
import argparse
import json
import pathlib
import sys

K = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(K / "tools"))
import fleurs_score as FS  # noqa: E402

# the references of the models without one in raw/refs/<model>/ref.json: Qwen3-ASR = transformers fp32
# (Qwen/Qwen3-ASR-1.7B-hf, the header row names the stack), Granite Speech = its fp32 CPU rows (the S220 scorer's R)
QREF = K / "raw/refs/qwen3asr-fp32/ref_transformers_official.jsonl"
GREF = K / "raw/refs/granite-fp32/R_fp32_cpu.jsonl"
# the clip lists (tools/fetch_clips.py rebuilds the clips from them)
S150_META = K / "fixtures/S150.meta.json"
S220_INDEX = K / "fixtures/S220.clips.json"
LANGS = ["en_us", "cmn_hans_cn", "ja_jp"]
RUNTIME = {"ovg": "OpenVINO GenAI 2026.4.1", "genai": "ONNX Runtime GenAI 0.19.0", "litertlm": "LiteRT-LM 0.19.1",
           "ort": "ONNX Runtime 1.31.1"}
DEVICE = {"cpu": "CPU (Core Ultra 9 386H)", "igpu": "Intel GPU", "rtx": "RTX 5080 Laptop", "npu": "Intel NPU"}


def shown(p):
    """A path as a record names it: relative to this repo, else from the home folder (~/...), never absolute."""
    p = pathlib.Path(p).resolve()
    for base, pre in ((K, ""), (pathlib.Path.home(), "~/")):
        try:
            return pre + str(p.relative_to(base))
        except ValueError:
            pass
    return str(p)


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))


def s150_refs():
    m = {c["name"]: c for c in jload(S150_META)["clips"] if c["path"].startswith("fleurs/")}
    return m


def ref_texts(model, refs_dir):
    """clip id -> reference system's text (the line's reference)."""
    if model == "qwen3asr":
        out = {}
        for l in QREF.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r.get("type") in ("header", "footer"):
                continue
            out[r["clip"]] = r.get("text") or ""
        return out, shown(QREF)
    p = pathlib.Path(refs_dir) / model / "ref.json"
    if not p.exists():
        return None, shown(p)
    return {r["id"]: r.get("text") or "" for r in jload(p)["rows"]}, shown(p)


def score_s150(rows, meta):
    agg = {}
    for r in rows:
        c = meta.get(r["id"])
        if not c:
            continue
        cfg = c["source"]["config"]
        e, n = FS.score(cfg, c["reference_text"]["raw_transcription"], r.get("text") or "")
        a = agg.setdefault(cfg, {"metric": "WER" if cfg == "en_us" else "CER", "err": 0, "n": 0, "clips": 0, "empty": 0})
        a["err"] += e
        a["n"] += n
        a["clips"] += 1
        a["empty"] += int(not (r.get("text") or "").strip())
    for a in agg.values():
        a["rate_pct"] = round(100.0 * a["err"] / a["n"], 3) if a["n"] else None
    return agg


def line_s150(run_agg, ref_agg, margin=0.5):
    out, ok = {}, True
    for cfg, a in run_agg.items():
        b = ref_agg.get(cfg)
        if not b or not b["n"]:
            continue
        p = a["err"] * b["n"] * 100 <= (b["err"] * 100 + margin * b["n"]) * a["n"]
        out[cfg] = {"run": a["rate_pct"], "ref": b["rate_pct"], "bar": round(100.0 * b["err"] / b["n"] + margin, 3), "pass": bool(p)}
        ok &= p
    empty = sum(a["empty"] for a in run_agg.values())
    out["empty"] = {"n": empty, "pass": empty == 0}
    out["pass"] = bool(ok and empty == 0)
    return out


def score_s220(rows):
    import s220_score as S  # noqa: E402
    clips = S.load_clips(S220_INDEX)
    ids = {r["id"] for r in rows}
    run = S.score_rows({r["id"]: {"text": r.get("text") or ""} for r in rows}, clips, None)
    ref_rows = {k: v for k, v in S.read_jsonl(GREF).items() if k in ids}
    ref = S.score_rows(ref_rows, clips, None)
    line = S.line(run, ref)
    line.pop("eos", None)   # the LiteRT-LM API gives no count of the answer's tokens: EOS not checked
    line["pass"] = all(v["pass"] for k, v in line.items() if isinstance(v, dict))
    return {k: v for k, v in run.items() if k != "per_clip"}, line, shown(GREF)


def score_tts(run_json, meta, ref_wb):
    wb_path = pathlib.Path(run_json).with_name(pathlib.Path(run_json).stem + ".writeback.json")
    if not wb_path.exists():
        return None, None, str(wb_path)
    wb = jload(wb_path)["texts"]
    err = n = 0
    for cid, text in wb.items():
        c = meta[cid]
        e, k = FS.score("en_us", c["reference_text"]["raw_transcription"], text or "")
        err += e
        n += k
    return {"metric": "write-back WER", "err": err, "n": n, "rate_pct": round(100.0 * err / n, 3) if n else None,
            "texts": len(wb)}, None, str(wb_path)


def cell_parts(cell):
    """<model>/<runtime>-<form>-<device>; a LiteRT-LM cell has no form part (the published .litertlm bundle)."""
    model, rest = cell.split("/", 1)
    parts = rest.split("-")
    if len(parts) == 2:
        return model, parts[0], "bundle", parts[1]
    rt, form, dev = parts
    return model, rt, form, dev


# runs that left no runner JSON, from the lane's notes (the chain's record does not say why an engine ended)
NO_JSON_NOTE = {"granite/litertlm-npu": "stopped by hand 22 min in with 60 to 69 of the 70 first-pass clips done: the "
                                        "stop read the runner's progress lines (clips 1, 2 and every 10th) as 8 clips "
                                        "(ended at 15:12 on 11-09, its progress lines in run1.stdout.kept.txt; "
                                        "corrected 11-10)"}


def progress(stdout_path):
    """How far a run without runner JSON got, from its progress lines (pc/audiolib.py prints clips 1, 2 and every 10th
    as {"n", "id", "wall_s"}): the last clip number printed, and the RTF over the printed clips (not a timing)."""
    if not stdout_path.exists():
        return None
    rows = [json.loads(ln) for ln in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.startswith('{"n"')]
    if not rows:
        return None
    sec = {c["name"]: float(c["duration_s"]) for c in jload(S150_META)["clips"]}
    sec.update({c["id"]: float(c["seconds"]) for c in jload(S220_INDEX)["clips"]})
    if not all(r["id"] in sec for r in rows):
        return {"printed": len(rows), "last_n": rows[-1]["n"]}
    return {"printed": len(rows), "last_n": rows[-1]["n"],
            "wall_s_printed": round(sum(r["wall_s"] for r in rows), 3),
            "audio_s_printed": round(sum(sec[r["id"]] for r in rows), 3),
            "rtf_printed": round(sum(r["wall_s"] for r in rows) / sum(sec[r["id"]] for r in rows), 4)}


def npu_luids(sitting):
    """The NPU's adapter LUIDs of a sitting, from its chain log's "adapters:" line (pc/chain.ps1 names an adapter whose
    only engine type is Neural "npu?"); LUIDs change at a reboot, so never a fixed value."""
    log = K / "raw" / sitting / "chain.log"
    if not log.exists():
        return []
    for ln in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "] adapters: " in ln:
            return [kv.split("=", 1)[0].strip() for kv in ln.split("] adapters: ", 1)[1].split(",")
                    if kv.split("=", 1)[-1].strip().startswith("npu")]
    return []


def npu_check(err_log, meta, sitting):
    """Where a LiteRT-LM NPU-backend run went: its log's compiler-plugin lines and the NPU's memory peak."""
    text = err_log.read_text(encoding="utf-8", errors="replace") if err_log.exists() else ""
    g = (meta.get("engine") or {}).get("gpuMiBPeak") or {}
    luids = npu_luids(sitting)
    mib = [(g.get(lu) or {}).get("shared", 0) + (g.get(lu) or {}).get("dedicated", 0) for lu in luids if lu in g]
    applied = [ln.split("compiler plugins were applied successfully", 1)[0].rsplit("]", 1)[-1].strip()
               for ln in text.splitlines() if "compiler plugins were applied successfully" in ln]
    c = {"plugin_loaded": [ln.split("Loaded plugin at:", 1)[1].strip() for ln in text.splitlines()
                           if "Loaded plugin at:" in ln],
         "partitioned": [ln.split("] ", 1)[-1].strip() for ln in text.splitlines() if "Partitioned subgraph" in ln],
         "plugins_applied": int(applied[-1]) if applied and applied[-1].isdigit() else None,
         "npu_compile_failed": [ln.strip() for ln in text.splitlines() if "vclAllocatedExecutableCreate" in ln],
         "no_compiler_plugin": "No compiler plugin found" in text,
         "xnnpack_cpu_delegate": "XNNPACK delegate for CPU" in text,
         "npu_luids": luids, "npu_mib_peak": max(mib) if mib else None}
    if (c["no_compiler_plugin"] or c["plugins_applied"] == 0) and c["npu_mib_peak"] == 0:
        c["ran_on"] = "cpu"   # no plugin, or a plugin whose compile failed: the model stays on the CPU (XNNPACK)
    elif c["plugins_applied"] and (c["npu_mib_peak"] or 0) > 0:
        c["ran_on"] = "npu"
    else:
        c["ran_on"] = "unclear"
    return c


def latest(runs_dirs, cell):
    """(run json path or None, meta path, sitting, mode) of the row: the latest timed OK run, else the latest
    first-pass run; a run with only its meta file (no runner JSON) counts last. A timed run of a sitting whose control
    tail did not pass (raw/<sitting>/control.json verdict other than PASS, distil-s1c of 11-12) is not a row."""
    timed, first = [], []
    for s in runs_dirs:
        d = K / "raw" / s / cell
        rj, mj = d / "run1.json", d / "run1.meta.json"
        if not mj.exists():
            continue
        m = jload(mj)
        if m.get("mode") == "timed":
            ctl = K / "raw" / s / "control.json"
            if not ctl.exists() or not str(jload(ctl).get("verdict") or "").startswith("PASS"):
                continue
        (timed if m.get("mode") == "timed" else first).append((rj if rj.exists() else None, mj, s, m.get("mode")))
    for lst in (timed, first):
        ok = [x for x in lst if x[0] and jload(x[0]).get("status") == "OK"]
        if ok:
            return ok[-1]
    allr = timed + first
    return allr[-1] if allr else None


def build(args):
    runs = [s for s in args.runs.split(",") if s]
    meta = s150_refs()
    cells = sorted({p.parent.name and str(p.parent.relative_to(K / "raw" / s)) for s in runs
                    for p in (K / "raw" / s).glob("*/*/run1.meta.json")} - {""})
    cells = [c for c in cells if "/" in c and not c.startswith("control")]
    refs_cache = {}
    first_texts = {}
    for s in runs:
        for p in (K / "raw" / s).glob("*/*/run1.json"):
            m = (p.with_name("run1.meta.json"))
            if m.exists() and jload(m).get("mode") == "first":
                c = str(p.parent.relative_to(K / "raw" / s))
                try:
                    first_texts[c] = {r["id"]: r.get("text") for r in jload(p).get("rows", [])}
                except Exception:
                    pass
    for cell in cells:
        pick = latest(runs, cell)
        if not pick:
            continue
        rj, mj, sitting, mode = pick
        r, m = (jload(rj) if rj else {}), jload(mj)
        if not rj:  # no runner JSON: the chain's record says how the engine ended
            e = m.get("engine") or {}
            code = e.get("exitCode")
            if e.get("timedOut"):
                r = {"status": "TIMEOUT", "error": f"no runner JSON: the chain's {e.get('elapsedSeconds', 0) / 60:.0f}-min "
                                                    f"limit, try {m.get('tries')}"}
            elif code == -1073741819:
                r = {"status": "CRASH", "error": f"no runner JSON: exit 0xC0000005 after {e.get('elapsedSeconds', 0):.0f} s"}
            else:
                r = {"status": "NO JSON", "error": NO_JSON_NOTE.get(cell) or f"no runner JSON: exit {code} after "
                                                                             f"{e.get('elapsedSeconds', 0):.0f} s"}
            p = progress(mj.with_name("run1.stdout.txt"))
            if p:
                r["progress"] = p
        model, rt, form, dev = cell_parts(cell)
        doc = {"cell": cell, "model": model, "runtime": RUNTIME.get(rt, rt), "form": form, "device": dev,
               "device_name": DEVICE.get(dev, dev), "set": r.get("argv_set") or (r.get("workload") or {}).get("set"),
               "mode": mode, "sitting": sitting, "run_start": m.get("start"), "run_end": m.get("end"),
               "status": r.get("status"),
               "record": {"run": str(rj.relative_to(K)) if rj else None, "meta": str(mj.relative_to(K)),
                          "chain_log": f"raw/{sitting}/chain.log"},
               "error": r.get("error"), "versions": r.get("versions"), "load_s": r.get("load_s"),
               "summary": r.get("summary"), "gate": m.get("gate"), "power": m.get("power"), "desktop_locked": m.get("desktopLocked"),
               "engine": {k: (m.get("engine") or {}).get(k) for k in ("elapsedSeconds", "peakWorkingSetBytes", "gpuMiBPeak",
                                                                     "exitCode", "timedOut", "suspendedSeconds")},
               "tries": m.get("tries")}
        if r.get("progress"):
            doc["progress"] = r["progress"]
        # LiteRT-LM's GPU log names its adapter (environment.cc "Selected adapter: ..."). Found 11-10: every Intel GPU run
        # of LiteRT-LM 0.19.1 in Z3 (first-a, s4) took the RTX: pc/run_asr_litertlm.py wrote GpuPreference=1 under the
        # interpreter's junction-resolved path (os.path.realpath), not the path its process ran by (README, 11-10)
        err_log = mj.with_name("run1.stderr.txt")
        if err_log.exists():
            for ln in err_log.read_text(encoding="utf-8", errors="replace").splitlines():
                if "Selected adapter:" in ln:
                    doc["adapter"] = ln.split("Selected adapter:", 1)[1].strip()
                    break
        if m.get("gpuPreference"):
            doc["gpu_preference"] = m["gpuPreference"]
        if dev == "igpu" and doc.get("adapter") and "intel" not in doc["adapter"].lower():
            doc["status"] = "WRONG DEVICE"
            doc["error"] = (f"ran on {doc['adapter'].split(',')[0]} (log: Selected adapter); GpuPreference=1 was "
                            "written under the interpreter's junction-resolved path, not the path its process ran by")
        # LiteRT-LM's NPU backend (found 11-10): a bundle that packages no TF_LITE_AUX model takes LiteRT's generic
        # compiler-plugin path, and LiteRT-LM 0.19.1's wheel ships no compiler plugin = the model runs on the CPU
        if dev == "npu" and rt == "litertlm":
            doc["npu_check"] = npu_check(err_log, m, sitting)
            if doc["npu_check"]["ran_on"] == "cpu":
                doc["ran_on"] = "cpu"
                why = ("no compiler plugin" if doc["npu_check"]["no_compiler_plugin"] else
                       "the compiler plugin's NPU compile failed")
                doc["device_name"] = f"CPU (Intel NPU requested; {why}, XNNPACK)"
            if r.get("npu_lib_dir"):
                doc["npu_lib_dir"] = r["npu_lib_dir"]
        # ONNX Runtime GenAI (found 11-12): pc/run_asr_genai.py sets --ep in code (clear_providers / append_provider),
        # which in 0.19.0 changes the decoder's session options only (src/config.cpp); the Whisper encoder takes the
        # session options of its own genai_config.json entry (src/models/whisper.cpp), and the builder's -e cpu exports
        # these cells load name no provider there = the CPU EP. A run that wrote no encoder options (--encoder-ep config,
        # every run before 11-12) ran its encoder on the CPU.
        av = r.get("argv") or []
        if rt == "genai" and "--ep" in av and av[av.index("--ep") + 1] in ("openvino", "webgpu") \
                and not r.get("encoder_session_options"):
            doc["encoder_on"] = "cpu"
        if r.get("encoder_session_options"):
            doc["encoder_session_options"] = r["encoder_session_options"]
        if r.get("config_providers") is not None:   # --ep config: the folder's genai_config.json as it is
            doc["config_providers"] = r["config_providers"]
        if mode == "timed":
            ctl = K / "raw" / sitting / "control.json"
            doc["control"] = jload(ctl) if ctl.exists() else None
        rows = r.get("rows") or []
        if r.get("status") == "OK" and rows:
            cset = doc["set"] or ("T50" if model == "kokoro" else "S220" if model == "granite" else "S150")
            doc["set"] = cset
            if cset == "S150":
                if model not in refs_cache:
                    refs_cache[model] = ref_texts(model, args.refs)
                rt_, rpath = refs_cache[model]
                agg = score_s150(rows, meta)
                doc["accuracy"] = agg
                if rt_:
                    ids = {x["id"] for x in rows}
                    ref_agg = score_s150([{"id": k, "text": v} for k, v in rt_.items() if k in ids], meta)
                    doc["reference"] = {"path": rpath, "scores": ref_agg}
                    doc["line"] = line_s150(agg, ref_agg)
            elif cset == "S220":
                acc, line, rpath = score_s220(rows)
                doc["accuracy"], doc["line"], doc["reference"] = acc, line, {"path": rpath}
            else:
                acc, _, wbp = score_tts(rj, meta, None)
                doc["accuracy"] = acc
                doc["record"]["writeback"] = str(pathlib.Path(wbp).relative_to(K)) if acc else None
                refwb = pathlib.Path(args.refs) / "kokoro" / "ref.writeback.json"
                if acc and refwb.exists():
                    wb = jload(refwb)["texts"]
                    e = n = 0
                    for cid in {x["id"] for x in rows}:
                        a, b = FS.score("en_us", meta[cid]["reference_text"]["raw_transcription"], wb.get(cid) or "")
                        e += a
                        n += b
                    silent = (r.get("summary") or {}).get("silent", 0)
                    p = acc["err"] * n * 100 <= (e * 100 + 1.0 * n) * acc["n"]
                    doc["reference"] = {"path": shown(refwb), "writeback_wer": round(100.0 * e / n, 3) if n else None}
                    doc["line"] = {"writeback": {"run": acc["rate_pct"], "ref": doc["reference"]["writeback_wer"],
                                                 "bar": round(100.0 * e / n + 1.0, 3), "pass": bool(p)},
                                   "silent": {"n": silent, "pass": not silent}, "pass": bool(p and not silent)}
            if mode == "timed" and cell in first_texts:
                ft = first_texts[cell]
                same = sum(1 for x in rows if x["id"] in ft and ft[x["id"]] == x.get("text"))
                doc["first_vs_timed_equal"] = f"{same}/{sum(1 for x in rows if x['id'] in ft)}"
        out = K / "results" / model / f"{cell.split('/', 1)[1]}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{cell:28s} {mode:5s} {r.get('status')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "table"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--refs", default=str(K / "raw" / "refs"))
    args = ap.parse_args()
    if args.what == "build":
        build(args)
    else:
        import table
        table.main()


if __name__ == "__main__":
    main()
