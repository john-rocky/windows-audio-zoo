#!/usr/bin/env python3
"""Draw the figure at the top of README.md from the result files.

    python3 tools/readme_figure.py

Reads results/<model>/*.json. Writes docs/images/results-light.svg and docs/images/results-dark.svg, and prints the
sentence the figure's alt text uses. It counts results by model and verdict only, so the figure names no runtime or device.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "pass": "#2a78d6", "fail": "#eb6834", "none": "#b4b2a7"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "pass": "#3987e5", "fail": "#d95926", "none": "#6f6e66"},
}
MODELS = [("whisper", "Whisper large-v3-turbo"), ("distil-large-v3", "Distil-Whisper distil-large-v3"),
          ("qwen3asr", "Qwen3-ASR 1.7B"), ("granite", "Granite Speech 5.0 2B"), ("parakeet", "Parakeet TDT 0.6B v3"),
          ("kokoro", "Kokoro-82M")]
KEYS = [("pass", "Matched the original model"), ("fail", "Did not match"), ("none", "Did not finish")]
TITLE = "Results per model, checked against the original model"
WIDTH, BAR, GAP = 860, 24, 2


def count():
    """Return {model: {pass, fail, none}} over results/<model>/*.json."""
    counts = {}
    for model, _ in MODELS:
        files = sorted((ROOT / "results" / model).glob("*.json"))
        if not files:
            sys.exit(f"readme_figure: no results for {model}")
        c = {"pass": 0, "fail": 0, "none": 0}
        for f in files:
            line = json.loads(f.read_text(encoding="utf-8")).get("line")
            c["none" if not line else "pass" if line.get("pass") else "fail"] += 1
        counts[model] = c
    return counts


def text(x, y, body, size, fill, weight="400"):
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}">{body}</text>'


def summary(v):
    words = [("pass", "matched"), ("fail", "did not match"), ("none", "did not finish")]
    return ", ".join(f"{v[key]} {word}" for key, word in words if v[key])


def draw(counts, theme, desc):
    c = THEMES[theme]
    unit = WIDTH / max(sum(v.values()) for v in counts.values())
    parts = [text(0, 28, TITLE, 20, c["text"], "600")]
    x = 0
    for key, label in KEYS:
        parts.append(f'<rect x="{x:g}" y="42" width="14" height="14" rx="4" fill="{c[key]}"/>')
        parts.append(text(x + 20, 54, label, 14, c["muted"]))
        x += 20 + 8.2 * len(label) + 20
    y = 92
    for model, label in MODELS:
        v = counts[model]
        parts.append(text(0, y, f"{label}: {summary(v)}", 15, c["text"]))
        x = 0
        for key, _ in KEYS:
            if v[key]:
                parts.append(f'<rect x="{x:g}" y="{y + 8:g}" width="{v[key] * unit - GAP:g}" height="{BAR}" rx="4" fill="{c[key]}"/>')
                x += v[key] * unit
        y += 62
    height = y - 62 + 8 + BAR + 8
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:g}" width="{WIDTH}" height="{height:g}" '
        f'role="img" aria-labelledby="t d" font-family="{FONT}">\n'
        f'<title id="t">{TITLE}</title>\n<desc id="d">{desc}</desc>\n' + "\n".join(parts) + "\n</svg>\n"
    )


def main():
    counts = count()
    desc = " ".join(f"{label}: {summary(counts[m])}." for m, label in MODELS)
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (OUT / f"results-{theme}.svg").write_text(draw(counts, theme, desc), encoding="utf-8")
    print(desc)


if __name__ == "__main__":
    main()
