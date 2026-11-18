#!/usr/bin/env python3
"""NOSPOT_SLEEVE (PREREG reports/NOSPOT_SLEEVE_2026-11-18_PREREG.md) — data: the missing Binance futures `metrics` daily zips and 5m
premiumIndexKlines (monthly, else daily for the current month) from data.binance.vision (0 API weight, generic UA, no account).
Input lists: <scratch>/metrics_todo.csv (pair,date) · prem_todo.csv (pair,month). Output reports/NOSPOT_SLEEVE_data/{metrics,premium}/."""
import concurrent.futures as cf, io, os, sys, zipfile
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import lead1_data as L   # noqa: E402  (get(): UA, 404 → None, stop on 403/418/429)
OD = os.path.join(R, "reports", "NOSPOT_SLEEVE_data")
VIS = "https://data.binance.vision/data/futures/um"


def met(x):
    p, d = x; fo = os.path.join(OD, "metrics", f"{p}.{d}.csv")
    if os.path.exists(fo):
        return 0
    z = L.get(f"{VIS}/daily/metrics/{p}/{p}-metrics-{d}.zip")
    if not z:
        open(fo, "w").write("missing\n"); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); pd.read_csv(zz.open(zz.namelist()[0])).to_csv(fo, index=False); return 1


def prem(x):
    p, m = x; fo = os.path.join(OD, "premium", f"{p}.{m}.csv")
    if os.path.exists(fo):
        return 0
    cols = ["open_time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"]
    z = L.get(f"{VIS}/monthly/premiumIndexKlines/{p}/5m/{p}-5m-{m}.zip")
    parts = []
    if z:
        parts.append(z)
    else:
        for d in pd.date_range(m + "-01", pd.Timestamp(m + "-01") + pd.offsets.MonthEnd(0)):
            zd = L.get(f"{VIS}/daily/premiumIndexKlines/{p}/5m/{p}-5m-{d:%Y-%m-%d}.zip")
            if zd:
                parts.append(zd)
    xs = []
    for z in parts:
        zz = zipfile.ZipFile(io.BytesIO(z)); raw = zz.open(zz.namelist()[0]).read().decode()
        hdr = 0 if raw.startswith("open_time") else None
        x = pd.read_csv(io.StringIO(raw), header=hdr)
        if hdr is None:
            x.columns = cols[:x.shape[1]]
        x = x.rename(columns={"close": "c"}); xs.append(x[["open_time", "c"]])
    if not xs:
        open(fo, "w").write("missing\n"); return 0
    pd.concat(xs).drop_duplicates("open_time").sort_values("open_time").to_csv(fo, index=False); return 1


if __name__ == "__main__":
    S = sys.argv[1]
    mt = pd.read_csv(os.path.join(S, "metrics_todo.csv")); pt = pd.read_csv(os.path.join(S, "prem_todo.csv"))
    with cf.ThreadPoolExecutor(8) as ex:
        print("metrics", sum(ex.map(met, zip(mt.pair, mt.date))), "of", len(mt), flush=True)
        print("premium", sum(ex.map(prem, zip(pt.pair, pt.month))), "of", len(pt), flush=True)
