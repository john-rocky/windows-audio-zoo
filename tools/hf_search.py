#!/usr/bin/env python3
"""List Hub model repos by author and search text (the Hub API, no login).
    python3 tools/hf_search.py <author> <text> [<author> <text> ...]"""
import json
import sys
import urllib.parse
import urllib.request

a = sys.argv[1:]
for i in range(0, len(a), 2):
    author, text = a[i], a[i + 1]
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({"author": author, "search": text, "limit": 30})
    with urllib.request.urlopen(url, timeout=30) as r:
        rows = json.load(r)
    print(f"== {author} / {text}: {len(rows)}")
    for m in rows:
        print("  ", m["id"], m.get("downloads"), m.get("lastModified", "")[:10])
