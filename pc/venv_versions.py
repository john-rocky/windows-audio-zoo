"""Every distribution of this interpreter's environment, with the Python version and the interpreter's paths
(pc/steps.ps1 runs it once per venv after its install).

    python venv_versions.py --out logs\\versions-<venv>.txt
"""
import argparse
import importlib.metadata as md
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
lines = [f"python {sys.version.split()[0]}", f"executable {sys.executable}",
         f"base_executable {getattr(sys, '_base_executable', '')}"]
lines += sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{len(lines) - 3} distributions -> {a.out}")
