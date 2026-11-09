#!/usr/bin/env python3
"""Rebuild the two clip folders the cells read from their public sources, and check every clip (README step 1).

    python tools/fetch_clips.py --out <folder> [--cache <folder>] [--fixtures <folder>]

Writes <folder>/s150 (meta.json, fleurs/<config>/<id>.wav, examples/en.wav) and <folder>/s220 (clips.json,
clips/<id>.wav), the layout `tools/pc.sh pushfix <folder>/s150 <folder>/s220` sends to the PC. Needs numpy, pyarrow,
soundfile and huggingface_hub, and ffmpeg on PATH (the warm-up clip). Downloads about 3 GB (seven FLEURS test parquets).

Sources, pinned:
  FLEURS       google/fleurs@70bb2e84... parquet-data/<config>/test-00000-of-00001.parquet (CC BY 4.0): the clips of
               fixtures/S150.meta.json (en_us, cmn_hans_cn, ja_jp, 50 each) and fixtures/S220.clips.json (the same en_us
               and ja_jp clips; fr_fr, de_de, es_419, pt_br 25 each). Rule (S150.meta.json "selection"): rows sorted by
               (id, path), the first row of each id, ascending id, the first N not over 30 s; int16 audio as it is,
               float audio as round(x * 32768) clipped; written as PCM16 mono WAV by soundfile.
  LibriSpeech  hf-internal-testing/librispeech_asr_dummy@5be91486... clean/validation-00000-of-00001.parquet
               (LibriSpeech dev-clean, CC BY 4.0): the 20 utterances of fixtures/S220.librispeech.json.
  Warm-up      FunAudioLLM/Fun-ASR-Nano-2512@272c57b8... example/en.mp3 (Apache-2.0), decoded by
               ffmpeg -ac 1 -ar 16000 -sample_fmt s16; untimed and not scored.
Checks (exit 1 when a scored clip is missing or differs):
  - each FLEURS config: the rule's first N ids = the index's ids, and each clip's file sha256 = the index's;
  - each LibriSpeech clip: its samples (sha256 of the int16 data) = those of the file the cells read; that file was a
    Core Audio WAV, so the container differs and the clips.json written here carries the new files' sha256;
  - the warm-up clip against the index's sha256: another ffmpeg build writes other bytes (a warning, not a failure).
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

K = pathlib.Path(__file__).resolve().parents[1]
FLEURS = ("google/fleurs", "70bb2e84b976b7e960aa89f1c648e09c59f894dd")
LIBRI = ("hf-internal-testing/librispeech_asr_dummy", "5be91486e11a2d616f4ec5db8d3fd248585ac07a")
FUNASR = ("FunAudioLLM/Fun-ASR-Nano-2512", "272c57b82523ada6fd87095e955f8e29100979ab")
SR = 16000
MAX_SAMPLES = 30 * SR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pcm_sha256(samples):
    return hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()


def pcm16(audio):
    """int16 stays as it is; float audio in [-1, 1) becomes round(x * 32768), clipped."""
    if audio.dtype == np.int16:
        return audio
    return np.clip(np.round(np.asarray(audio, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def decode(cell):
    """A parquet audio cell (a struct with the file's bytes) -> (mono samples, sample rate)."""
    with sf.SoundFile(io.BytesIO(cell["bytes"])) as f:
        data = f.read(dtype="int16" if f.subtype == "PCM_16" else "float64", always_2d=True)
        rate, channels = f.samplerate, f.channels
    if channels != 1:
        raise ValueError(f"{channels} channels, want mono")
    return pcm16(data[:, 0]), rate


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, SR, subtype="PCM_16")


def fetch(repo, rev, filename, repo_type, cache):
    return pathlib.Path(hf_hub_download(repo_id=repo, filename=filename, revision=rev, repo_type=repo_type,
                                        cache_dir=cache))


def fleurs_pick(config, n, cache):
    """The rule's first n clips of one config: {id: int16 samples}."""
    table = pq.read_table(fetch(FLEURS[0], FLEURS[1], f"parquet-data/{config}/test-00000-of-00001.parquet",
                                "dataset", cache))
    first = {}
    for i, r in enumerate(table.select(["id", "path"]).to_pylist()):
        rid, path = int(r["id"]), str(r.get("path") or "")
        if rid not in first or path < first[rid][1]:
            first[rid] = (i, path)
    audio, picked = table.column("audio"), {}
    for rid in sorted(first):
        samples, rate = decode(audio[first[rid][0]].as_py())
        if rate != SR:
            raise ValueError(f"FLEURS {config} id {rid}: {rate} Hz, want {SR}")
        if samples.shape[0] > MAX_SAMPLES:
            continue
        picked[rid] = samples
        if len(picked) == n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="folder for s150/ and s220/")
    ap.add_argument("--cache", default=None, help="Hub download cache (default: huggingface_hub's own)")
    ap.add_argument("--fixtures", default=str(K / "fixtures"), help="the folder with the three index files")
    a = ap.parse_args()
    out, fx = pathlib.Path(a.out).resolve(), pathlib.Path(a.fixtures)
    s150, s220 = out / "s150", out / "s220"
    m150 = json.loads((fx / "S150.meta.json").read_text(encoding="utf-8"))
    m220 = json.loads((fx / "S220.clips.json").read_text(encoding="utf-8"))
    libri = json.loads((fx / "S220.librispeech.json").read_text(encoding="utf-8"))
    bad, warn = [], []

    # FLEURS: each config's ids and where each clip goes (S220's en_us / ja_jp clips are S150's)
    want = {}
    for c in m150["clips"]:
        if c["path"].startswith("fleurs/"):
            src = c["source"]
            want.setdefault(src["config"], {}).setdefault(int(src["id"]), []).append((s150 / c["path"], c["sha256"]))
    for c in m220["clips"]:
        if c["set"] != "en_ls":
            config, rid = c["id"].rsplit("_", 1)
            want.setdefault(config, {}).setdefault(int(rid), []).append((s220 / "clips" / f"{c['id']}.wav",
                                                                         c["sha256"]))
    for config in sorted(want):
        n = len(want[config])
        picked = fleurs_pick(config, n, a.cache)
        if sorted(picked) != sorted(want[config]):
            bad.append(f"FLEURS {config}: the rule's first {n} ids are not the index's")
        for rid, dests in sorted(want[config].items()):
            if rid not in picked:
                bad.append(f"FLEURS {config} {rid}: not picked by the rule")
                continue
            for dst, sha in dests:
                write_wav(dst, picked[rid])
                if sha256_file(dst) != sha:
                    bad.append(f"{dst.relative_to(out)}: sha256 differs from the index")
        print(f"FLEURS {config}: {n} clips", flush=True)

    # LibriSpeech: samples checked against the files the cells read; the new WAVs' sha256 go into clips.json
    table = pq.read_table(fetch(LIBRI[0], LIBRI[1], "clean/validation-00000-of-00001.parquet", "dataset", a.cache))
    utts = table.column("id").to_pylist()
    new_sha = {}
    for e in libri["clips"]:
        if e["utt"] not in utts:
            bad.append(f"{e['id']}: utterance {e['utt']} not in the parquet")
            continue
        samples, rate = decode(table.column("audio")[utts.index(e["utt"])].as_py())
        if rate != SR:
            bad.append(f"{e['id']}: {rate} Hz, want {SR}")
        if pcm_sha256(samples) != e["pcm_sha256"]:
            bad.append(f"{e['id']}: samples differ from the measured file's")
        dst = s220 / "clips" / f"{e['id']}.wav"
        write_wav(dst, samples)
        new_sha[e["id"]] = sha256_file(dst)
    missing = [c["id"] for c in m220["clips"] if c["set"] == "en_ls" and c["id"] not in new_sha]
    bad += [f"{i}: no LibriSpeech clip written" for i in missing]
    print(f"LibriSpeech: {len(new_sha)} clips", flush=True)

    # the warm-up clip (untimed, not scored)
    ex = next(c for c in m150["clips"] if c["path"] == "examples/en.wav")
    mp3 = fetch(FUNASR[0], FUNASR[1], "example/en.mp3", "model", a.cache)
    if sha256_file(mp3) != ex["source"]["file_sha256"]:
        bad.append("example/en.mp3: sha256 differs from the index")
    dst = s150 / "examples" / "en.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", str(SR),
                    "-sample_fmt", "s16", str(dst)], check=True)
    if sha256_file(dst) != ex["sha256"]:
        warn.append("examples/en.wav (the warm-up clip, not scored): sha256 differs from the index (another ffmpeg "
                    "build writes other bytes)")

    # the index files the runners read
    m150["fixtures_root"] = str(s150)
    (s150 / "meta.json").write_text(json.dumps(m150, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for c in m220["clips"]:
        if c["id"] in new_sha:
            c["sha256"] = new_sha[c["id"]]
    m220["source"] = ("tools/fetch_clips.py; the en_ls clips' sha256 are of the WAVs written here (their samples "
                      "checked against the measured files)")
    s220.mkdir(parents=True, exist_ok=True)
    (s220 / "clips.json").write_text(json.dumps(m220, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for w in warn:
        print("WARN " + w)
    for b in bad:
        print("BAD  " + b)
    n150 = sum(1 for c in m150["clips"] if c["path"].startswith("fleurs/"))
    print(f"{'FAIL' if bad else 'PASS'}: {n150} S150 + {len(m220['clips'])} S220 clips, {len(bad)} bad, "
          f"{len(warn)} warning(s) -> {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
