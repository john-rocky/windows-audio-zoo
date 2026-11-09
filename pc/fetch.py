"""Download the Hub files of shelf/hub_manifest.json into <root>\\models\\<repo>\\<path> at the pinned revisions and
check every file against the Hub's hash (sha256 = lfs.oid for LFS files, the git blob sha1 for small ones).

    python fetch.py --manifest shelf\\hub_manifest.json --dest C:\\src\\audiozoo\\models [--only <repo substring> ...]

curl.exe, resumable (-C -), cut and resumed when one connection falls under 100 KB/s for 60 s (seen on this PC: one
connection can crawl while the next is fast). A file whose hash already matches is skipped. Writes
<dest>\\fetch_report.json and exits 1 when any file does not match after 6 tries.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time


def blob_sha1(path):
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(path, f):
    if not path.is_file() or path.stat().st_size != f["size"]:
        return False
    return (sha256(path) == f["sha256"]) if f.get("sha256") else (blob_sha1(path) == f["blob"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    man = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    dest = pathlib.Path(args.dest)
    report, bad = [], 0
    for src in man:
        if args.only and not any(o in src["repo"] + "/" + src["folder"] for o in args.only):
            continue
        for f in src["files"]:
            path = dest / src["repo"].replace("/", "__") / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{f['path']}"
            t0, tries = time.time(), 0
            while not ok(path, f) and tries < 6:
                tries += 1
                if path.is_file() and path.stat().st_size > f["size"]:
                    path.unlink()
                subprocess.run(["curl.exe", "-sSL", "--fail", "-C", "-", "--speed-limit", "102400", "--speed-time", "60",
                                "--retry", "3", "-o", str(path), url])
            good = ok(path, f)
            bad += not good
            row = {"repo": src["repo"], "revision": src["revision"], "path": f["path"], "size": f["size"],
                   "hash": f.get("sha256") or f["blob"], "ok": good, "tries": tries, "seconds": round(time.time() - t0, 1)}
            report.append(row)
            print(json.dumps(row), flush=True)
    (dest / "fetch_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"fetch done: {len(report)} files, {bad} not matching", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
