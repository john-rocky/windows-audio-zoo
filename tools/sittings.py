#!/usr/bin/env python3
"""The timed sittings from the first passes: every cell whose first-pass run ended OK goes to the sitting of its device
(s1 = CPU, s2 = Intel GPU and NPU, s3 = RTX), with its expected run time scaled from that run (tools/make_plan.py timed).
A cell whose first pass failed keeps its first-pass record as its row (the error) and is not timed.

    python3 tools/sittings.py --first first-a,first-an,first-b [--known <control rtf>] [--only s1,s2,s3] [--slots <json>] [--dry]

--known: the control row's known RTF (from the first sitting's head: runs\\s1\\control.json known_rtf); without it the
sitting's control head runs 3 times and sets it.
--end-by: the ISO time no cell may run past (the chain leaves the rest to a later sitting, control.json left_cells).
--cells with a new sitting name (e.g. --only s4 --cells <left cells>): a later sitting of those cells, any device.
--no-windows: the sitting is one block to --end-by (the PC's order of the night 11-09/10): the chain opens no duty
window for other lanes' notes (a slot_now.txt request from the Mac still applies).
--slots: a fixed slot for another lane in every sitting's plan (tools/make_plan.py timed --slots), e.g. another lane's block:
    '[{"who": "another lane", "release_at": "2026-11-09T21:07:00+09:00", "give_up_at": "2026-11-09T21:37:00+09:00"}]'
"""
import argparse
import json
import pathlib
import subprocess
import sys

K = pathlib.Path(__file__).resolve().parents[1]
GROUPS = {"s1": ("cpu",), "s2": ("igpu", "npu"), "s3": ("rtx",)}


def first_ok(sittings):
    ok, bad = {}, {}
    for s in sittings:
        for rj in sorted((K / "raw" / s).glob("*/*/run1.json")):
            cell = str(rj.parent.relative_to(K / "raw" / s))
            try:
                st = json.loads(rj.read_text(encoding="utf-8-sig")).get("status")
            except Exception as e:  # noqa: BLE001
                st = f"unreadable ({e})"
            if st == "OK":   # an OK run in a later first pass (a redo) replaces an earlier failure
                ok[cell] = s
                bad.pop(cell, None)
            elif cell not in ok:
                bad[cell] = s
        for mj in sorted((K / "raw" / s).glob("*/*/run1.meta.json")):   # a run that wrote no runner JSON
            cell = str(mj.parent.relative_to(K / "raw" / s))
            if cell not in ok and not (mj.parent / "run1.json").exists():
                bad[cell] = s
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--known", type=float, default=0)
    ap.add_argument("--only", default="s1,s2,s3")
    ap.add_argument("--slots", default="")
    ap.add_argument("--end-by", default="")
    ap.add_argument("--no-windows", action="store_true", help="one block to end_by: no duty window (make_plan.py)")
    ap.add_argument("--cells", default="", help="only these cells (a later sitting of the cells an end_by left)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    sits = [s for s in args.first.split(",") if s]
    ok, bad = first_ok(sits)
    sys.path.insert(0, str(K / "tools"))
    import make_plan  # noqa: E402
    dev = {c["cell"]: c["device"] for c in make_plan.cells()}
    for c, s in sorted(bad.items()):
        print(f"not timed (first pass not OK in {s}): {c}")
    for name in [x for x in args.only.split(",") if x]:
        only = set(x for x in args.cells.split(",") if x)
        cells = [c for c in sorted(ok) if dev.get(c) in GROUPS.get(name, ("cpu", "igpu", "npu", "rtx")) and (not only or c in only)]
        if not cells:
            print(f"{name}: no cells")
            continue
        cmd = [sys.executable, str(K / "tools" / "make_plan.py"), "timed", "--sitting", name, "--cells", ",".join(cells),
               "--est-from", ",".join(sits)]
        if args.known:
            cmd += ["--known", str(args.known)]
        if args.slots:
            cmd += ["--slots", args.slots]
        if args.end_by:
            cmd += ["--end-by", args.end_by]
        if args.no_windows:
            cmd += ["--no-windows"]
        print(f"{name}: {len(cells)} cells: {' '.join(cells)}")
        if not args.dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
