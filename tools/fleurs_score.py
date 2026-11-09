"""The S150 scorer's normalization and edit distance, verbatim from the scorer of earlier Qwen3-ASR
work on this PC (itself verbatim from an earlier FLEURS scorer): en_us WER on norm_text, cmn_hans_cn / ja_jp CER on norm_chars,
corpus rate = sum of distances / sum of reference lengths. Pure python.
"""
import re
import unicodedata

# ---- verbatim from the earlier FLEURS scorer ------------------------------------------------------------
CER_FORMULA = ("CER = sum(levenshtein(norm(ref), norm(hyp))) / sum(len(norm(ref))); norm(s) = NFKC(s).lower() with every "
               "char of Unicode category P* or Z* and every whitespace char removed; ref = FLEURS raw_transcription")
WER_FORMULA = ("WER = sum(levenshtein_words(norm_text(ref), norm_text(hyp))) / sum(len(norm_text(ref))); norm_text = "
               "upper(), '-' -> ' ', [^A-Z0-9' ] -> ' ', split; ref = FLEURS raw_transcription")


def norm_text(s):
    s = s.upper().replace("-", " ")
    return re.sub(r"[^A-Z0-9' ]+", " ", s).split()


def norm_chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return [ch for ch in s if not (unicodedata.category(ch)[0] in "PZ" or ch.isspace())]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def score(config, ref, hyp):
    if config == "en_us":
        r, h = norm_text(ref), norm_text(hyp)
    else:
        r, h = norm_chars(ref), norm_chars(hyp)
    return lev(r, h), len(r)
# ---- end verbatim ------------------------------------------------------------------------------------------------
