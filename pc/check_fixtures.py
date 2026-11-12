"""Check the clip sets on this PC against their own lists: every WAV's sha256 (S150 meta.json, S220 clips.json) and the
warm-up clip. Exit 1 when one is missing or differs.

    python check_fixtures.py --fixtures <PC folder>\\fixtures [--sets S150]

--sets: only these clip sets (default both; a folder for the distil-large-v3 cells has S150 only).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audiolib as al  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fixtures", required=True)
ap.add_argument("--sets", default="S150,S220")
a = ap.parse_args()
bad, n = [], 0
for s in a.sets.split(","):
    for r in al.clips(s, a.fixtures):
        n += 1
        if not os.path.exists(r["path"]):
            bad.append(f"{s} {r['id']}: missing")
        elif al.sha256_file(r["path"]) != r["sha256"]:
            bad.append(f"{s} {r['id']}: sha256 differs")
w = al.warmup_wav(a.fixtures)
if not os.path.exists(w):
    bad.append(f"warm-up clip missing: {w}")
print(f"{n} clips checked, {len(bad)} bad")
for b in bad:
    print("  " + b)
sys.exit(1 if bad else 0)
