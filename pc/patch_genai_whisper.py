"""ONNX Runtime GenAI 0.19.0's model builder, Whisper: the encoder graph gets one cross-attention K / V output per
*encoder* layer, each made from decoder.layers[i].encoder_attn (builders/whisper.py, WhisperEncoder: the output names
and make_postprocessing_nodes loop over range(self.num_layers) = encoder_layers). A model whose decoder has fewer layers
than its encoder (whisper-large-v3-turbo: 32 encoder layers, 4 decoder layers) stops there:
    IndexError: index 4 is out of range
The runtime reads those outputs by the decoder's layer count (src/models/io/cross_kv_cache.cpp, v0.19.0:
layer_count_ = model.config_->model.decoder.num_hidden_layers; names from encoder.outputs.cross_present_*_names), so the
encoder must make one pair per decoder layer. This edits the three loops of WhisperEncoder to count decoder layers, in
the venv's installed copy of builders/whisper.py; the original stays beside it as whisper.py.orig. The lines this script
edits are unchanged on the builder's main branch at 75fa2250c9 (2026-11-12). A fix for the builder itself is proposed in
https://github.com/microsoft/onnxruntime-genai/pull/2940 (a draft pull request, open and not merged on 2026-11-12).

    python patch_genai_whisper.py            # patch (a patched file is left as it is)
    python patch_genai_whisper.py --check    # print the state only

Refuses a file in which the three lines are not found as written (another builder version).
"""
import argparse
import importlib.util
import pathlib
import shutil
import sys

MARK = "# Z3 patch: cross K/V per decoder layer"
EDITS = [
    ("        config.num_hidden_layers = config.encoder_layers\n",
     "        config.num_hidden_layers = config.encoder_layers\n"
     f"        self.num_cross_layers = config.decoder_layers  {MARK}\n"),
    ('            "present_key_cross": {i: f"present_key_cross_{i}" for i in range(self.num_layers)},\n',
     '            "present_key_cross": {i: f"present_key_cross_{i}" for i in range(self.num_cross_layers)},\n'),
    ('            "present_value_cross": {i: f"present_value_cross_{i}" for i in range(self.num_layers)},\n',
     '            "present_value_cross": {i: f"present_value_cross_{i}" for i in range(self.num_cross_layers)},\n'),
    ('        for i in range(self.num_layers):\n            for proj_type in ["k_proj", "v_proj"]:\n',
     '        for i in range(self.num_cross_layers):\n            for proj_type in ["k_proj", "v_proj"]:\n'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    spec = importlib.util.find_spec("onnxruntime_genai")   # the package's folder, without importing it
    if not spec or not spec.submodule_search_locations:
        sys.exit("onnxruntime_genai is not installed in this interpreter")
    f = pathlib.Path(list(spec.submodule_search_locations)[0]) / "models" / "builders" / "whisper.py"
    text = f.read_text(encoding="utf-8")
    if MARK in text:
        print(f"PATCHED already: {f}")
        return
    missing = [i for i, (old, _) in enumerate(EDITS) if text.count(old) != 1]
    if missing:
        sys.exit(f"REFUSED: edit(s) {missing} not found exactly once in {f} (another builder version?)")
    if args.check:
        print(f"NOT PATCHED (all {len(EDITS)} edits apply): {f}")
        return
    shutil.copy2(f, f.with_suffix(".py.orig"))
    for old, new in EDITS:
        text = text.replace(old, new)
    f.write_text(text, encoding="utf-8")
    print(f"PATCHED: {f} ({len(EDITS)} edits; the original is {f.name}.orig)")


if __name__ == "__main__":
    main()
