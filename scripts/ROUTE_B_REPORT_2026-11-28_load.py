#!/usr/bin/env python3
"""Route B report 2026-11-28 — step 1: pull (cached) every hourly per-minute summary + the recorder's event log from S3 (read-only),
write reports/ROUTE_B_REPORT_2026-11-28_data/{features_all.pkl.gz, events.json, positions_from_bot_list.csv}."""
import json, os, sys, io
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b"))
import daily_report as DR  # noqa: E402

OUT = os.path.join(ROOT, "reports", "ROUTE_B_REPORT_2026-11-28_data")
os.makedirs(OUT, exist_ok=True)
LAST_HOUR = os.environ.get("RB_LAST_HOUR", "features/2026-11-28/01.csv.gz")   # the latest COMPLETE hourly summary (the current hour is still being written)
fk = sorted(k for k in DR.s3_keys("features/") if k.endswith(".csv.gz") and k <= LAST_HOUR)
print("hour files", len(fk), fk[0], fk[-1])
fr = []
CACHE = os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data", "s3")        # the loader's local copy of the same S3 keys (read first)
for k in fk:
    lp = os.path.join(CACHE, k)
    if os.path.exists(lp):
        with open(lp, "rb") as fh:
            b = fh.read()
    else:
        b = DR.fetch(k)
    if not b:
        print("missing", k); continue
    d = pd.read_csv(io.BytesIO(b), compression="gzip")
    d["hourfile"] = k
    fr.append(d)
F = pd.concat(fr, ignore_index=True)
num = [c for c in F.columns if c not in ("sym", "hourfile")]
F[num] = F[num].apply(pd.to_numeric, errors="coerce")
F.to_pickle(os.path.join(OUT, "features_all.pkl.gz"))
ek = [k for k in DR.s3_keys("data/events/") if k.endswith(".jsonl.gz")]
ev = DR.read_events([b for b in (DR.fetch(k) for k in ek) if b])
with open(os.path.join(OUT, "events.json"), "w") as f:
    json.dump([{k: v for k, v in e.items() if k != "host"} for e in ev], f)
pos = DR.positions(ev, None)
pos.to_csv(os.path.join(OUT, "positions_from_bot_list.csv"), index=False)
print(len(F), F.sym.nunique(), len(ev), len(pos))
print(pos.to_string())
