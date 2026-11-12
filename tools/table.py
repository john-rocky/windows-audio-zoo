#!/usr/bin/env python3
"""RESULTS_AUDIO_WINDOWS.md's table from results/<model>/*.json (this lane's cells, tools/results.py) and
results/existing/*.json (cells collected from earlier work on this PC; kept local, not in this repo). The text above and below the table (conclusion,
notes) is kept as written in RESULTS_AUDIO_WINDOWS.md; only the part between the markers is replaced.

    python3 tools/table.py
"""
import json
import pathlib

K = pathlib.Path(__file__).resolve().parents[1]
OUT = K / "RESULTS_AUDIO_WINDOWS.md"
BEGIN, END = "<!-- table:begin -->", "<!-- table:end -->"
MODEL = {"whisper": "Whisper large-v3-turbo", "qwen3asr": "Qwen3-ASR 1.7B", "granite": "Granite Speech 5.0 2B",
         "parakeet": "Parakeet TDT 0.6B v3", "kokoro": "Kokoro-82M", "distil-large-v3": "Distil-Whisper distil-large-v3"}
ORDER = ["whisper", "distil-large-v3", "qwen3asr", "granite", "parakeet", "kokoro"]
DEV = {"cpu": "CPU", "igpu": "Intel GPU", "rtx": "RTX 5080 Laptop", "npu": "Intel NPU"}
# a LiteRT-LM cell whose NPU backend got the Intel OpenVINO compiler plugin beside its dispatch library (11-10)
FORM = {"plugin": "bundle + Intel compiler plugin"}


def jload(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def acc_text(d):
    a = d.get("accuracy") or {}
    if d.get("set") == "S150":
        parts = []
        for cfg, lab in (("en_us", "en WER"), ("cmn_hans_cn", "zh CER"), ("ja_jp", "ja CER")):
            if cfg in a and a[cfg].get("rate_pct") is not None:
                parts.append(f"{lab} {a[cfg]['rate_pct']:.2f}")
        return " / ".join(parts)
    if d.get("set") == "S220":
        g = a.get("groups") or {}
        parts = []
        for k, lab in (("en_fl", "en-FL WER"), ("ja", "ja CER"), ("eu", "fr+de+es+pt WER")):
            if k in g and g[k].get("rate_pct") is not None:
                parts.append(f"{lab} {g[k]['rate_pct']:.2f}")
        return " / ".join(parts)
    if d.get("set") == "T50" and a.get("rate_pct") is not None:
        return f"write-back WER {a['rate_pct']:.2f}"
    return ""


def line_text(d):
    ln = d.get("line")
    if not ln:
        return "—"
    if ln.get("pass"):
        return "PASS"
    bad = [k for k, v in ln.items() if isinstance(v, dict) and not v.get("pass")]
    return "FAIL " + ", ".join(bad)


def mem_text(d):
    e = d.get("engine") or {}
    ws = e.get("peakWorkingSetBytes")
    out = f"{ws / 2**30:.1f}" if ws else ""
    g = e.get("gpuMiBPeak") or {}
    # the adapter that held the most (an Intel GPU run also opens the RTX for a few MiB while it picks its adapter,
    # Z3b 11-10: RTX 13 MiB dedicated beside 2235 MiB shared on the Intel GPU)
    ads = [(v.get("dedicated", 0) + v.get("shared", 0), v.get("dedicated", 0), v.get("shared", 0))
           for v in g.values() if isinstance(v, dict)]
    if ads and max(ads)[0] > 0:
        _, dd, sh = max(ads)
        out += f" / GPU {dd / 1024:.1f} ded" if dd >= sh else f" / GPU {sh / 1024:.1f} shared"
    return out


def row_this(d):
    s = d.get("summary") or {}
    st = d.get("status")
    mode = d.get("mode")
    note = []
    if mode == "first":
        tts = d.get("set") == "T50" or d.get("model") == "kokoro"
        full = {"S150": 150, "S220": 220, "T50": 50}.get(d.get("set") or "")
        if d.get("model") == "distil-large-v3":   # an English model: its first pass is the 50 en_us clips (11-12)
            note.append("first pass, the 50 en_us clips, untimed")
        elif full and (d.get("summary") or {}).get("clips") == full:   # a first pass over every clip of its set (11-10)
            note.append("first pass, every clip, untimed")
        else:
            note.append("first pass, 10 sentences, untimed" if tts else "first pass, 10 clips per language, untimed")
    timed_ok = mode == "timed"   # a timed row counts as a timing only when its sitting's tail row passed
    if mode == "timed":
        c = d.get("control") or {}
        v = c.get("verdict") or ("no tail row" if c else "no control record")
        if not v.startswith("PASS"):
            note.append(f"control: {v}; RTF not a timing")
            timed_ok = False
    if st != "OK":
        err = (d.get("error") or "").strip().splitlines()
        note.append(("ERROR" if st in (None, "ERROR") else st) + ": " + (err[-1][:120] if err else "no runner JSON"))
    rtf = s.get("rtf")
    ft = s.get("first_token_ms_mean")
    rec = d["record"]["run"] or d["record"]["meta"]
    dev = DEV.get(d["device"], d["device"])
    nc = d.get("npu_check") or {}
    if d.get("ran_on") == "cpu" and nc.get("no_compiler_plugin"):   # LiteRT-LM's NPU backend, no plugin (11-10)
        dev = "CPU (Intel NPU requested)"
        note.append("the NPU backend ran on the CPU: no compiler plugin (LiteRT-LM 0.19.1's wheel ships the Intel "
                    "dispatch library only), XNNPACK CPU delegate, 0 MiB on the NPU")
    elif d.get("ran_on") == "cpu":   # a compiler plugin loaded, its NPU compile failed: no plugin applied
        dev = "CPU (Intel NPU requested)"
        note.append("the NPU backend ran on the CPU: the Intel compiler plugin's NPU compile failed "
                    f"({len(nc.get('npu_compile_failed') or [])} partition), 0 compiler plugins applied, XNNPACK CPU "
                    "delegate, 0 MiB on the NPU")
    elif d.get("encoder_on") == "cpu":   # ONNX Runtime GenAI: --ep reached the decoder only (11-12)
        note.append(f"decoder on the {dev}, encoder on the CPU (the provider was set in code; the export's "
                    "genai_config.json names none for the encoder)")
        dev = f"{dev} (decoder)"
    elif d.get("encoder_session_options"):   # --encoder-ep same / openvino-gpu / openvino-npu (11-12)
        prov = "; ".join(" ".join([k] + [f"{a}={b}" for a, b in v.items()])
                         for o in d["encoder_session_options"].get("provider_options", []) for k, v in o.items())
        note.append(f"encoder on {prov} (`--encoder-ep`: a copy of the export's genai_config.json)")
    elif d.get("config_providers") is not None:   # --ep config (11-12): the folder's genai_config.json as it is
        prov = "; ".join(f"{part} " + (", ".join(" ".join([k] + [f"{a}={b}" for a, b in v.items()])
                                                 for o in opts for k, v in o.items()) or "CPU")
                         for part, opts in d["config_providers"].items())
        note.append(f"providers from the folder's genai_config.json: {prov}")
    elif nc.get("ran_on") == "npu":
        src = sorted({f.get("from", "").split(":")[0] for f in ((d.get("npu_lib_dir") or {}).get("manifest") or {})
                      .get("files", []) if "CompilerPlugin" in f.get("name", "")})
        note.append("the Intel OpenVINO compiler plugin" + (f" of {', '.join(src)}" if src else "") +
                    f" beside LiteRT-LM's dispatch library ({len(nc.get('partitioned') or [])} subgraphs partitioned), "
                    f"{nc.get('npu_mib_peak')} MiB on the NPU")
    if st == "OK":
        acc = acc_text(d)
        rtf_cell = f"{rtf:.4f}" if (rtf and timed_ok) else (f"({rtf:.4f})" if rtf else "")
    else:   # no score and no timing: the cell says so and how the run ended (the note has the detail)
        acc = rtf_cell = "none: " + ("ERROR" if st in (None, "ERROR") else st)
        p = d.get("progress") or {}
        if p.get("rtf_printed"):
            rtf_cell = f"({p['rtf_printed']:.4f}, the {p['printed']} printed clips)"
    return [MODEL[d["model"]], d["runtime"], FORM.get(d["form"], d["form"]), dev, d.get("set") or "",
            acc, line_text(d) if st == "OK" else "—", rtf_cell,
            f"{ft:.0f}" if (ft and st == "OK") else "", mem_text(d) if st == "OK" else "",
            f"`{rec}`", "; ".join(note)]


def main():
    rows = []
    for m in ORDER:
        for p in sorted((K / "results" / m).glob("*.json")):
            rows.append(row_this(jload(p)))
        for p in sorted((K / "results" / "existing").glob(f"{m}__*.json")):
            d = jload(p)
            rows.append([MODEL[m], d["runtime"], d["form"], DEV.get(d["device"], d["device"]), d["set"], d["accuracy_text"],
                         d["line"], d.get("rtf_text", ""), d.get("first_token_ms_text", ""), d.get("memory_text", ""),
                         f"`{d['record']}`", d.get("note", "")])
    head = ["model", "runtime", "form", "device", "clips", "WER / CER (%)", "line", "RTF", "first token ms",
            "peak RAM GB / GPU GB", "record", "note"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    table = "\n".join(lines)
    text = OUT.read_text(encoding="utf-8") if OUT.exists() else f"# Audio cells on one Windows PC\n\n{BEGIN}\n{END}\n"
    if BEGIN not in text:
        text += f"\n{BEGIN}\n{END}\n"
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    OUT.write_text(text[:a] + "\n" + table + "\n" + text[b:], encoding="utf-8")
    print(f"{OUT.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
