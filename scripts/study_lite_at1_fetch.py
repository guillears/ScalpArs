#!/usr/bin/env python3
"""LITE AT-+1 study — budgeted fetch of coin 1m klines WITH taker-buy volume for every cohort fill window
[min(te - 65 min, spike_at - 5 min), te + 725 min] + BTC/ETH/SOL 1m for the live window after the cache end. Public fapi, read-only.
Budget <= 1,200 weight/min: limit 1000 (weight 5), pause when X-MBX-USED-WEIGHT-1M > 900, stop on 418/429 and arm
reports/backtest_cache/.binance_cooldown_until. Output reports/lite_at1/k1m/<PAIR>.csv (never backtest_cache). Resumable.
Refuses to run without reports/LITE_AT1_PREREG.txt.
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import os, sys, time, json, hashlib
import numpy as np, pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "lite_at1", "k1m")
CD = os.path.join(REP, "backtest_cache", ".binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines"
LIMIT = 1000
MIN = 60_000
COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ig"]
STATE = {"w": 0, "calls": 0}
LIVE_CSV = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_21-18-32.csv")


def cooldown_check():
    if os.path.exists(CD):
        try:
            if _BC.active(CD):
                raise SystemExit("cooldown active — stop")
        except ValueError:
            pass


def get(params):
    cooldown_check()
    if STATE["w"] > 900:
        print(f"  used weight {STATE['w']} > 900 → sleep 61 s", flush=True)
        time.sleep(61); STATE["w"] = 0
    r = requests.get(URL, params=params, timeout=20)
    STATE["w"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
    STATE["calls"] += 1
    if r.status_code in (418, 429):
        ra = int(r.headers.get("Retry-After", "120") or 120)
        open(CD, "w").write(str(time.time() + max(ra, 120)))
        raise SystemExit(f"rate limited {r.status_code} — cooldown written, stop")
    if r.status_code == 400:
        return []
    r.raise_for_status()
    time.sleep(0.3)
    return r.json()


def fetch_range(sym, s, e):
    rows = []
    while s < e:
        k = get(dict(symbol=sym, interval="1m", startTime=int(s), endTime=int(e) - 1, limit=LIMIT))
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + MIN
        if len(k) < LIMIT:
            break
    return rows


def save(path, rows):
    df = pd.DataFrame([r[:12] for r in rows], columns=COLS).apply(pd.to_numeric)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df.to_csv(path, index=False)
    return df


def covered(df, a, b):
    if df is None or not len(df):
        return False
    need = np.arange(a, b, MIN)
    return bool(np.isin(need, df.open_time.values.astype("int64")).all())


def windows():
    M = pd.read_csv(os.path.join(REP, "lite_wlsweep", "matrix.csv"), low_memory=False)
    M = M[M.block.isin(["BT", "MASTER"])]
    W = []
    for r in M.itertuples():
        te = int(r.te)
        sp = int(pd.Timestamp(r.entry_frenzy_spike_at, tz="UTC").value // 1_000_000)
        W.append((r.pair, (min(te - 65 * MIN, sp - 5 * MIN) // MIN) * MIN, (te // MIN) * MIN + 725 * MIN))
    L = pd.read_csv(LIVE_CSV, low_memory=False)
    L = L[(L.entry_strategy == "FRENZY_LITE") & (L.status == "CLOSED")]
    for r in L.itertuples():
        te = int(pd.Timestamp(r.opened_at, tz="UTC").value // 1_000_000)
        sp = int(pd.Timestamp(r.entry_frenzy_spike_at, tz="UTC").value // 1_000_000) if isinstance(r.entry_frenzy_spike_at, str) else te
        W.append((r.pair, (min(te - 65 * MIN, sp - 5 * MIN) // MIN) * MIN, (te // MIN) * MIN + 725 * MIN))
        for b in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            W.append((b, max((te // MIN) * MIN - 300 * MIN, int(pd.Timestamp("2026-10-15 15:00", tz="UTC").value // 1_000_000)),
                      (te // MIN) * MIN + 725 * MIN))
    return W


def main():
    assert os.path.exists(os.path.join(REP, "LITE_AT1_PREREG.txt")), "pre-register first"
    os.makedirs(OUT, exist_ok=True)
    now = (int(time.time() * 1000) // MIN) * MIN
    W = windows()
    by = {}
    for p, a, b in W:
        by.setdefault(p, []).append([a, min(b, now)])
    t0 = time.time()
    empty = []
    for i, (p, ws) in enumerate(sorted(by.items())):
        ws.sort(); mg = []
        for a, b in ws:
            if mg and a <= mg[-1][1]:
                mg[-1][1] = max(mg[-1][1], b)
            else:
                mg.append([a, b])
        path = os.path.join(OUT, f"{p}.csv")
        df = pd.read_csv(path) if os.path.exists(path) else None
        for a, b in mg:
            if covered(df, a, b):
                continue
            rows = fetch_range(p, a, b)
            if rows:
                df = save(path, rows)
            else:
                empty.append((p, a, b))
        if i % 25 == 0:
            print(f"{i + 1}/{len(by)} {p} calls {STATE['calls']} used-weight {STATE['w']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    json.dump(empty, open(os.path.join(REP, "lite_at1", "fetch_empty.json"), "w"))
    with open(os.path.join(REP, "lite_at1", "fetch_manifest.txt"), "w") as f:
        f.write(f"fetched {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} calls {STATE['calls']} empty windows {len(empty)}\n")
        for fn in sorted(os.listdir(OUT)):
            f.write(f"{hashlib.sha256(open(os.path.join(OUT, fn), 'rb').read()).hexdigest()}  k1m/{fn}\n")
    print("done calls", STATE["calls"], "empty", len(empty))


if __name__ == "__main__":
    main()
