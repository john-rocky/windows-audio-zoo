"""The S220 scorer (the Granite Speech set): the same normalization and line as the scorer of its reference rows
(raw/refs/granite-fp32/R_fp32_cpu.jsonl), kept here so tools/results.py needs no other folder.

  en_ls / en_fl       WER after openai-whisper 20250625 whisper.normalizers.EnglishTextNormalizer()
  fr / de / es / pt   pooled WER after whisper.normalizers.BasicTextNormalizer()
  ja                  CER: NFKC -> lower() -> drop Unicode P* / Z* characters and whitespace

Rates = sum of edit distances / sum of reference lengths, rounded once. Also counted: empty answers, loops
((.{4,}?)\\1{3,}), answers not stopped by EOS (the 256-token cap reached). Line: en_fl, ja and eu (fr+de+es+pt) each
<= the reference + 0.5 pt (counts compared, not rounded rates); empty, loop and not-EOS 0.
"""
import json
import os
import re
import unicodedata

MAX_NEW = 256
LOOP = re.compile(r"(.{4,}?)\1{3,}", re.S)
GROUPS = {"en_fl": ["en_fl"], "ja": ["ja"], "eu": ["fr", "de", "es", "pt"], "en_ls": ["en_ls"],
          "en70": ["en_ls", "en_fl"]}
LINE = ("en_fl", "ja", "eu")
MARGIN = 0.5
_EN = None
_BASIC = None


def normalizers():
    global _EN, _BASIC
    if _EN is None:
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

        _EN, _BASIC = EnglishTextNormalizer(), BasicTextNormalizer()
    return _EN, _BASIC


def norm_chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return [ch for ch in s if not (unicodedata.category(ch)[0] in "PZ" or ch.isspace())]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[-1]


def units(set_name, s):
    en, basic = normalizers()
    if set_name in ("en_ls", "en_fl"):
        return en(s).split()
    if set_name == "ja":
        return norm_chars(s)
    return basic(s).split()


def load_clips(index):
    """The clips of one clips.json (id, set, ref, ...), in the file's order."""
    return json.load(open(index, encoding="utf-8"))["clips"]


def read_jsonl(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def score_rows(rows, clips, ref_rows=None, cap=MAX_NEW):
    fx = {c["id"]: c for c in clips}
    per_set, per_clip = {}, []
    for cid, r in rows.items():
        c = fx[cid]
        ref_units, hyp_units = units(c["set"], c["ref"]), units(c["set"], r["text"])
        e, n = lev(ref_units, hyp_units), len(ref_units)
        s = per_set.setdefault(c["set"], {"err": 0, "n": 0, "clips": 0, "empty": 0, "loop": 0, "not_eos": 0,
                                          "ids_vs_ref": 0, "ids_known": 0})
        empty = not r["text"].strip()
        loop = bool(LOOP.search(r["text"]))
        nt = r.get("n_tokens")
        not_eos = bool(r.get("hit_cap")) or (nt is not None and nt >= cap) or (r.get("eos") is False)
        for k, v in (("err", e), ("n", n), ("clips", 1), ("empty", empty), ("loop", loop), ("not_eos", not_eos)):
            s[k] += v
        pc = {"id": cid, "set": c["set"], "err": e, "n": n, "empty": empty, "loop": loop, "n_tokens": nt,
              "not_eos": not_eos}
        if ref_rows is not None and cid in ref_rows and r.get("gen_ids") is not None:
            rr = ref_rows[cid]["gen_ids"]
            s["ids_known"] += 1
            same = r["gen_ids"] == rr
            s["ids_vs_ref"] += same
            pc["ids_vs_ref"] = same
            if not same:
                pc["first_id_diff"] = next((i for i, (a, b) in enumerate(zip(r["gen_ids"], rr)) if a != b),
                                           min(len(r["gen_ids"]), len(rr)))
        per_clip.append(pc)
    groups = {}
    for g, sets in GROUPS.items():
        e = sum(per_set[x]["err"] for x in sets if x in per_set)
        n = sum(per_set[x]["n"] for x in sets if x in per_set)
        k = sum(per_set[x]["clips"] for x in sets if x in per_set)
        if k:
            groups[g] = {"err": e, "n": n, "rate_pct": round(100.0 * e / n, 3) if n else None, "clips": k}
    for s in per_set.values():
        s["rate_pct"] = round(100.0 * s["err"] / s["n"], 3) if s["n"] else None
    tot = {k: sum(s[k] for s in per_set.values()) for k in ("clips", "empty", "loop", "not_eos", "ids_vs_ref",
                                                            "ids_known")}
    return {"groups": groups, "sets": per_set, "totals": tot, "per_clip": per_clip}


def line(run, ref):
    out = {}
    for g in LINE:
        if g in run["groups"] and g in ref["groups"]:
            a, b = run["groups"][g], ref["groups"][g]
            ok = a["err"] * b["n"] * 100 <= (b["err"] * 100 + MARGIN * b["n"]) * a["n"]
            out[g] = {"run": a["rate_pct"], "ref": b["rate_pct"], "bar": round(100.0 * b["err"] / b["n"] + MARGIN, 3),
                      "pass": bool(ok)}
    t = run["totals"]
    out["empty"] = {"n": t["empty"], "pass": t["empty"] == 0}
    out["loop"] = {"n": t["loop"], "pass": t["loop"] == 0}
    out["eos"] = {"not_eos": t["not_eos"], "pass": t["not_eos"] == 0}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out
