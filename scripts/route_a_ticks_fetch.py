#!/usr/bin/env python3
"""ROUTE A — FETCH. Downloads Binance public USDT-M daily aggTrades zips (data.binance.vision, .CHECKSUM verified) for the sampled
trigger events, cuts the windows the study needs and deletes the zip:
  ticks  [sig − 75 min, sig + 10 min)  (time, price, qty, is_buyer_maker)
  1m OHLC from sig − 75 min to the end of the last downloaded day (exit path; the 5m cache continues it)
  baseline ticks around one random moment on the same pair-day ≥ 2 h from any trigger of any grid cell
Stratified selection with replacement-on-404 in the frozen order (route_a_ticks_sample.py). ≤ 4 parallel downloads, back-off.
Output reports/ROUTE_A_TICKS_data/ev/<PAIR>_<sig_ms>.npz, fetch_log.csv. Raw zips go to reports/ROUTE_A_TICKS_data/raw/ (temporary).
Usage: venv/bin/python -I scripts/route_a_ticks_fetch.py [--max-rounds 4]"""
import argparse, hashlib, io, os, sys, threading, time, urllib.request, urllib.error, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); RAW = os.path.join(D, "raw"); EV = os.path.join(D, "ev")
os.makedirs(RAW, exist_ok=True); os.makedirs(EV, exist_ok=True)
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
DAY, MIN = 86_400_000, 60_000
PRE, POST = 75 * MIN, 10 * MIN
LOCK = threading.Lock(); STATS = dict(bytes=0, files=0, miss=0)
UA = {"User-Agent": "research-client/1.0"}


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k * 2)
        except Exception:
            time.sleep(2 ** k * 2)
    raise RuntimeError("failed " + url)


def fetch_day(pair, d):
    """returns DataFrame (t, p, q, m) for one pair-day or None if the file does not exist"""
    url = URL.format(s=pair, d=d)
    b = get(url)
    if b is None:
        return None
    ck = get(url + ".CHECKSUM")
    if ck is not None and ck.split()[0].decode() != hashlib.sha256(b).hexdigest():
        b = get(url)
        if hashlib.sha256(b).hexdigest() != ck.split()[0].decode():
            raise RuntimeError("checksum " + url)
    with LOCK:
        STATS["bytes"] += len(b); STATS["files"] += 1
    z = zipfile.ZipFile(io.BytesIO(b)); name = z.namelist()[0]
    raw = z.read(name)
    first = raw[:200].split(b"\n")[0]
    hdr = 0 if first[:1].isalpha() else None
    df = pd.read_csv(io.BytesIO(raw), header=hdr, usecols=[1, 2, 5, 6], names=None if hdr == 0 else ["a", "p", "q", "f", "l", "t", "m"])
    df.columns = ["p", "q", "t", "m"]
    if df.m.dtype != bool:
        df["m"] = df.m.astype(str).str.lower().eq("true")
    return df[["t", "p", "q", "m"]].sort_values("t", kind="stable").reset_index(drop=True)


def bars1m(df, t0, t1):
    x = df[(df.t >= t0) & (df.t < t1)]
    if not len(x):
        return np.zeros((0, 5))
    k = (x.t.values // MIN) * MIN
    g = pd.DataFrame(dict(k=k, p=x.p.values)).groupby("k").p
    return np.c_[g.first().index.values, g.first().values, g.max().values, g.min().values, g.last().values]


def cut(df, a, b):
    x = df[(df.t >= a) & (df.t < b)]
    return x.t.values.astype("int64"), x.p.values.astype("float64"), x.q.values.astype("float64"), x.m.values.astype(bool)


def do_pair(args):
    pair, evs, trig = args
    cache = {}
    out = []
    for e in evs:
        f = os.path.join(EV, f"{pair}_{e['sig_ms']}.npz")
        if os.path.exists(f):
            out.append((e["sig_ms"], "ok")); continue
        s = int(e["sig_ms"])
        days = sorted({(s - PRE) // DAY, s // DAY, (s + POST) // DAY})
        dfs = []
        miss = False
        for dd in days:
            ds = pd.Timestamp(dd * DAY, unit="ms").strftime("%Y-%m-%d")
            if ds not in cache:
                try:
                    cache[ds] = fetch_day(pair, ds)
                except Exception as ex:
                    print("ERR", pair, ds, ex, flush=True); cache[ds] = None
            if cache[ds] is None:
                miss = True
            else:
                dfs.append(cache[ds])
        if miss or not dfs:
            with LOCK:
                STATS["miss"] += 1
            out.append((s, "missing")); continue
        df = pd.concat(dfs, ignore_index=True)
        tt, pp, qq, mm = cut(df, s - PRE, s + POST)
        if len(tt) < 50:
            out.append((s, "thin")); continue
        bars = bars1m(df, s - PRE, (max(days) + 1) * DAY)
        # baseline moment on the sig's day, window inside the downloaded data, ≥ 2 h from any trigger
        sday = s // DAY; rng = np.random.default_rng(s % (2 ** 32))
        lo, hi = max(sday * DAY + PRE, df.t.iloc[0] + PRE), min((sday + 1) * DAY - POST, df.t.iloc[-1] - POST)
        bms = -1
        for _ in range(60):
            if hi <= lo:
                break
            c = int(rng.integers(lo, hi)) // MIN * MIN
            if len(trig) == 0 or np.min(np.abs(trig - c)) >= 2 * 3_600_000:
                bms = c; break
        if bms > 0:
            bt, bp, bq, bm = cut(df, bms - PRE, bms + POST)
        else:
            bt = np.zeros(0, "int64"); bp = bq = np.zeros(0); bm = np.zeros(0, bool)
        np.savez_compressed(f, t=tt, p=pp, q=qq, m=mm, bars=bars, bms=np.int64(bms), bt=bt, bp=bp, bq=bq, bm=bm)
        out.append((s, "ok"))
    return pair, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-rounds", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    O = pd.read_csv(os.path.join(D, "order.csv"))
    T = pd.read_csv(os.path.join(D, "all_triggers.csv"))
    trig = {p: np.sort(g.sig_ms.values.astype("int64")) for p, g in T.groupby("pair")}
    logf = os.path.join(D, "fetch_log.csv")
    status = pd.read_csv(logf) if os.path.exists(logf) else pd.DataFrame(columns=["pair", "sig_ms", "status"])
    done = {(r.pair, int(r.sig_ms)): r.status for r in status.itertuples()}
    t0 = time.time()
    for rnd in range(a.max_rounds):
        O["st"] = [done.get((p, int(s)), "") for p, s in zip(O.pair, O.sig_ms)]
        ok = O[O.st == "ok"].groupby(["month", "runner"]).size()
        jobs = []
        for (mo, rn), g in O.groupby(["month", "runner"]):
            need = int(g.target.iloc[0]) - int(ok.get((mo, rn), 0))
            if need <= 0:
                continue
            cand = g[g.st == ""].sort_values("rank").head(need)
            jobs.append(cand)
        if not jobs:
            break
        J = pd.concat(jobs)
        print(time.strftime("%H:%M:%S"), f"round {rnd}: {len(J)} events, {J.pair.nunique()} pairs", flush=True)
        tasks = [(p, g.to_dict("records"), trig.get(p, np.zeros(0, "int64"))) for p, g in J.groupby("pair")]
        new = []
        with ThreadPoolExecutor(a.workers) as P:
            for k, (pair, res) in enumerate(P.map(do_pair, tasks)):
                for s, st in res:
                    done[(pair, int(s))] = st; new.append(dict(pair=pair, sig_ms=int(s), status=st))
                if k % 25 == 0:
                    print(time.strftime("%H:%M:%S"), k, "/", len(tasks), f"{STATS['bytes']/1e9:.2f} GB {STATS['files']} files miss {STATS['miss']}",
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
                    pd.DataFrame([dict(pair=p, sig_ms=s, status=v) for (p, s), v in done.items()]).to_csv(logf, index=False)
        pd.DataFrame([dict(pair=p, sig_ms=s, status=v) for (p, s), v in done.items()]).to_csv(logf, index=False)
    print("DONE", f"{STATS['bytes']/1e9:.2f} GB downloaded, {STATS['files']} files, {STATS['miss']} missing", flush=True)


if __name__ == "__main__":
    main()
