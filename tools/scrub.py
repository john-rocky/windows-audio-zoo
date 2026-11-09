#!/usr/bin/env python3
"""Replace account names with USER in the text files under the given paths, in place, before they are committed (the
bench's Windows records write C:\\Users\\USER the same way). The names are given on the command line, so this file
carries none of them. A JSON file that parsed before must still parse after.

    python3 tools/scrub.py --names <name>[,<name>...] <path> [<path> ...]      # prints the files it changed
    python3 tools/scrub.py --names <name>[,<name>...] --check <path> [...]     # exit 1 when a name is still there
"""
import argparse
import json
import pathlib
import re
import sys

TEXT = {".json", ".jsonl", ".txt", ".log", ".md", ".csv", ".tsv", ".ps1", ".py", ".sh", ".cells", ".xml", ""}


def files(paths):
    for p in map(pathlib.Path, paths):
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and ".git" not in f.parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    names = [n for n in args.names.split(",") if n]
    missing = [p for p in args.paths if not pathlib.Path(p).exists()]
    if missing:  # a path that is not there would pass the check unread (zsh hands an unsplit "$P" as one path)
        print("no such path: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)
    # a whole word only: a longer handle that contains a name (a public account) is left as it is
    pats = [re.compile(r"(?i)(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])") for n in names]
    left = changed = 0
    for f in files(args.paths):
        if f.suffix.lower() not in TEXT:
            continue
        try:
            s = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                s = f.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
        hits = sum(len(p.findall(s)) for p in pats)
        if not hits:
            continue
        if args.check:
            print(f"{f}: {hits}")
            left += 1
            continue
        parsed = None
        if f.suffix.lower() == ".json":
            try:
                json.loads(s.lstrip("\ufeff"))
                parsed = True
            except ValueError:
                parsed = False
        t = s
        for p in pats:
            t = p.sub("USER", t)
        if parsed:
            json.loads(t.lstrip("\ufeff"))  # raises if the replacement broke the file
        f.write_text(t, encoding="utf-8")
        changed += 1
        print(f"scrubbed {f} ({hits})")
    if args.check and left:
        sys.exit(1)
    if not args.check:
        print(f"{changed} file(s) changed")


if __name__ == "__main__":
    main()
