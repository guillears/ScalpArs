#!/usr/bin/env python3
"""LITE OB study (2026-10-15) — step 0: merge every BOOK row from ~/Downloads/scalpars_decisions_*.csv, dedup on (t, pair).
Research only. Output reports/lite_ob/book.pkl"""
import glob, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "lite_ob"); os.makedirs(OUT, exist_ok=True)
fr = []
for f in sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"))):
    d = pd.read_csv(f, low_memory=False)
    d = d[d.e == "BOOK"]
    if len(d):
        d = d[["t", "pair"] + [c for c in d.columns if c.startswith("ob_")]]
        d["file"] = os.path.basename(f)
        fr.append(d)
B = pd.concat(fr, ignore_index=True)
B["t"] = pd.to_datetime(B.t, utc=True)
B = B.drop_duplicates(["t", "pair"], keep="last").sort_values(["pair", "t"]).reset_index(drop=True)
B.to_pickle(os.path.join(OUT, "book.pkl"))
print(len(B), B.pair.nunique(), B.t.min(), B.t.max())
print(B.t.dt.second.value_counts().head())
print(B.describe().T.to_string())
