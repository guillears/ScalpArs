#!/usr/bin/env python3
"""DIST_ENTRY_TIMING (PREREG reports/DIST_ENTRY_TIMING_2026-11-20_PREREG.md) — PART A sub-minute prices. READ-ONLY on the bot.
  --real   : the paper DIST fills (decisions exports: DISTRIBUTION OPENED row = signal close T + candle close; OPEN row = fill time + price)
             + the operator-reported KERNELUSDT 11-20 fill → aggTrades around T (data.binance.vision daily zip; 11-20 via fapi aggTrades,
             2 calls, weight 20 each, used-weight header read) → reports/DIST_ENTRY_TIMING_data/latency_real.csv
  --sample : a random sample (seed 1120) of ≤ 40 REF_B kept fills → aggTrades daily zip (skipped if > 80 MB) → price at T + s for
             s ∈ {0, 5, 10, 15, 20, 30, 45, 60, 90, 120} (last trade ≤ T + s; s = 0 → first trade ≥ T) → latency_sample.csv
Usage: venv/bin/python -I scripts/dist_entry_timing_latency.py --real|--sample"""
import argparse, glob, io, json, os, sys, time, zipfile, urllib.request, urllib.error
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(R, "reports", "DIST_ENTRY_TIMING_data"); os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research client)"}
VIS = "https://data.binance.vision/data/futures/um/daily/aggTrades"
SEC = (0, 5, 10, 15, 20, 30, 45, 60, 90, 120)
DAY = 86_400_000


def http(url, head=False):
    req = urllib.request.Request(url, headers=UA, method="HEAD" if head else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w is not None and int(w) > 1000:
                raise SystemExit(f"used weight {w} — stopping")
            return (r.headers.get("Content-Length") if head else r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"HTTP {e.code} — stopping")
        return None


def agg_window(p, T, lo_s=-180, hi_s=180):
    """aggTrades (ms, price) in [T + lo, T + hi] from the daily zip (or fapi for days not yet published)."""
    ds = pd.to_datetime(T, unit="ms").strftime("%Y-%m-%d")
    z = http(f"{VIS}/{p}/{p}-aggTrades-{ds}.zip")
    if z:
        zz = zipfile.ZipFile(io.BytesIO(z))
        df = pd.read_csv(zz.open(zz.namelist()[0]), header=None, usecols=[1, 5], low_memory=False)
        if not str(df.iloc[0, 1]).isdigit():
            df = df.iloc[1:]
        tt = df.iloc[:, 1].values.astype(np.int64); tt = np.where(tt > 10**14, tt // 1000, tt)
        px = df.iloc[:, 0].values.astype(float)
        m = (tt >= T + lo_s * 1000) & (tt <= T + hi_s * 1000)
        return tt[m], px[m], "vision"
    b = http(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={p}&startTime={T + lo_s * 1000}&endTime={T + hi_s * 1000}&limit=1000")
    if not b:
        return None, None, "none"
    a = json.loads(b)
    xs = [(int(r["T"]), float(r["p"])) for r in a]
    last = a[-1] if a else None
    for _ in range(30):                                     # page forward (weight 20 / call) until the window is covered
        if last is None or int(last["T"]) > T + hi_s * 1000:
            break
        a2 = json.loads(http(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={p}&fromId={last['a'] + 1}&limit=1000") or b"[]")
        if not a2:
            break
        xs += [(int(r["T"]), float(r["p"])) for r in a2 if int(r["T"]) <= T + hi_s * 1000]
        last = a2[-1]; time.sleep(0.3)
    xs = np.array(xs)
    return xs[:, 0].astype(np.int64), xs[:, 1], "fapi"


def px_at(tt, px, T, s):
    if s == 0:
        j = np.searchsorted(tt, T, "left"); return px[j] if j < len(tt) else np.nan
    j = np.searchsorted(tt, T + s * 1000, "right") - 1
    return px[j] if j >= 0 else np.nan


def real():
    rows = []
    fs = sorted(set(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")) +
                    glob.glob(os.path.join(R, "exports_archive", "scalpars_decisions_paper_*.csv"))))
    D = pd.concat([pd.read_csv(f, usecols=["t", "e", "pair", "gate", "strategy", "price"], low_memory=False) for f in fs]).drop_duplicates()
    op = D[(D.e == "DISTRIBUTION") & (D.gate == "OPENED")]
    fl = D[(D.e == "OPEN") & (D.strategy == "DISTRIBUTION_SHORT")]
    for _, r in op.iterrows():
        T = pd.Timestamp(r.t).value // 10**6
        f = fl[(fl.pair == r.pair) & (pd.to_datetime(fl.t).values.astype("datetime64[ms]").astype(np.int64) >= T)
               & (pd.to_datetime(fl.t).values.astype("datetime64[ms]").astype(np.int64) < T + 600_000)]
        if not len(f):
            continue
        f = f.iloc[0]
        rows.append(dict(pair=r.pair, T=T, close=float(r.price), fill_ms=pd.Timestamp(f.t).value // 10**6, fill_px=float(f.price), src="export"))
    # operator-reported (2026-11-20, not in an export yet): signal close 12:30:00 UTC, filled 12:30:27 at 0.05114
    rows.append(dict(pair="KERNELUSDT", T=pd.Timestamp("2026-11-20 12:30:00").value // 10**6, close=np.nan,
                     fill_ms=pd.Timestamp("2026-11-20 12:30:27").value // 10**6, fill_px=0.05114, src="operator"))
    out = []
    for r in rows:
        tt, px, src = agg_window(r["pair"], r["T"], -180, 180)
        d = dict(r); d["agg_src"] = src
        if tt is not None and len(tt):
            d["px_T0"] = px_at(tt, px, r["T"], 0)
            d["last_before_T"] = px[np.searchsorted(tt, r["T"], "left") - 1] if np.searchsorted(tt, r["T"], "left") > 0 else np.nan
            for s in (-120, -60):
                d[f"px_T{s}s"] = px_at(tt, px, r["T"], s)
            for s in SEC:
                d[f"px_{s}s"] = px_at(tt, px, r["T"], s)
            d["px_at_fill"] = px_at(tt, px, r["fill_ms"], 1)
        out.append(d); print(d, flush=True); time.sleep(0.5)
    X = pd.DataFrame(out)
    X["delay_s"] = (X.fill_ms - X["T"]) / 1000
    X["fill_vs_T0_bps"] = (X.px_T0 / X.fill_px - 1) * 1e4        # + = the fill is BELOW the T price = a cost to the short
    X["fill_vs_close_bps"] = (X.close / X.fill_px - 1) * 1e4
    X["Tm1_vs_fill_bps"] = (X["px_T-60s"] / X.fill_px - 1) * 1e4   # what a T − 60 s entry would have gained vs the fill
    X["Tm2_vs_fill_bps"] = (X["px_T-120s"] / X.fill_px - 1) * 1e4
    X.to_csv(os.path.join(OUT, "latency_real.csv"), index=False)
    print(X[["pair", "T", "delay_s", "close", "px_T0", "fill_px", "fill_vs_T0_bps", "fill_vs_close_bps", "Tm1_vs_fill_bps", "Tm2_vs_fill_bps"]].to_string())


def sample(n=40):
    TR = pd.read_csv(os.path.join(R, "reports", "DIST_EXIT_2026-11-18_trades.csv.gz"))
    TR = TR[TR.cell == "REF_B"].reset_index(drop=True)
    rng = np.random.default_rng(1120)
    order = rng.permutation(len(TR))
    out = []
    for i in order:
        if len(out) >= n:
            break
        r = TR.iloc[i]; p = r.pair; T = int(r.entry_ms)
        ds = pd.to_datetime(T, unit="ms").strftime("%Y-%m-%d")
        cl = http(f"{VIS}/{p}/{p}-aggTrades-{ds}.zip", head=True)
        if cl is None or int(cl) > 80e6:
            continue
        tt, px, src = agg_window(p, T, -150, 150)
        if tt is None or not len(tt):
            continue
        d = dict(pair=p, T=T, block=r.block, net=r.net)
        for s in SEC:
            d[f"px_{s}s"] = px_at(tt, px, T, s)
        for s in (-120, -60):
            d[f"px_T{s}s"] = px_at(tt, px, T, s)
        d["n_trades_60s"] = int(((tt >= T) & (tt < T + 60_000)).sum())
        out.append(d); print(len(out), p, ds, flush=True)
    X = pd.DataFrame(out)
    for s in SEC[1:]:
        X[f"cost_{s}s_bps"] = (X.px_0s / X[f"px_{s}s"] - 1) * 1e4     # + = price fell after T = a later short entry is worse
    X.to_csv(os.path.join(OUT, "latency_sample.csv"), index=False)
    print(X[[f"cost_{s}s_bps" for s in SEC[1:]]].describe().round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); ap.add_argument("--sample", action="store_true")
    A = ap.parse_args()
    if A.real:
        real()
    if A.sample:
        sample()
