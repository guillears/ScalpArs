#!/usr/bin/env python3
"""FRENZY W/L screen 2026-10-19 — journal union (read-only). Unions every decisions export in ~/Downloads + the archived
B22 journal, keeps BOOK rows (FRENZY order-book minute snapshots), FRENZY-family BLOCK rows, WILLY events, OPEN rows.
Output: scratch pickle (not a report). Dedup key (t, e, pair, gate, src)."""
import glob, os, sys
import pandas as pd
OUT = sys.argv[1]
files = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"))) + \
        ["/Users/guillearslanian/Downloads/NOFA AI/reports/BASELINE22_batch1014-1016_LIVE_decisions.csv"]
keep = []
for f in files:
    try:
        j = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print("skip", f, e); continue
    mode = "live" if "_live_" in f or "LIVE" in f else "paper"
    g = j.gate.astype(str)
    m = (j.e == "BOOK") | ((j.e == "BLOCK") & g.str.contains("FRENZY|LITE|WILLY")) | j.e.isin(["WILLY", "OPEN", "WILLY_HOLD"])
    k = j[m].copy(); k["mode"] = mode; k["file"] = os.path.basename(f)
    keep.append(k)
    print(os.path.basename(f), j.t.min(), j.t.max(), len(k))
J = pd.concat(keep, ignore_index=True)
J = J.drop_duplicates(subset=["t", "e", "pair", "gate", "src", "mode"])
J.to_pickle(OUT)
print(J.shape); print(J.groupby(["mode", "e"]).size())
print("BOOK span", J[J.e == "BOOK"].t.min(), J[J.e == "BOOK"].t.max())
