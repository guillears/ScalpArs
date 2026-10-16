#!/usr/bin/env python3
"""LITE FADE study — step 2: budgeted fetch of 5m klines WITH taker-buy volume (all 12 fields) for every cohort fill window, a BTC 1m tail
(master / live days) and the two live trades' 1m + 5m bars. Public fapi endpoint, read-only. Refuses to run without the PREREG.
Budget: ≤ 1,200 weight/min — reads X-MBX-USED-WEIGHT-1M, pauses above 700 (another agent was fetching concurrently), stops on 418/429 and writes / respects
reports/backtest_cache/.binance_cooldown_until. Output under reports/lite_fade_klines/ (outside backtest_cache). Resumable.
Usage: venv/bin/python scripts/study_lite_fade_fetch.py [--live-only]
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import argparse, os, sys, time
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_fade_common as F  # noqa: E402

CD = os.path.join(F.REP, "backtest_cache", ".binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines"
LIMIT = 499                                    # weight 2 (fapi: [100, 500) → 2)
LIVE = [("EPICUSDT", "2026-10-15 14:55:05"), ("DUSKUSDT", "2026-10-15 15:05:05")]
STATE = {"w": 0, "calls": 0}


def cooldown_check():
    if os.path.exists(CD):
        try:
            if _BC.active(CD):
                raise SystemExit("cooldown active — stop")
        except ValueError:
            pass


def get(params):
    cooldown_check()
    if STATE["w"] > 700:
        print(f"  used weight {STATE['w']} > 700 → sleep 61 s", flush=True)
        time.sleep(61)
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
    time.sleep(0.6)
    return r.json()


def fetch_range(sym, interval, s, e, step):
    rows = []
    while s < e:
        k = get(dict(symbol=sym, interval=interval, startTime=int(s), endTime=int(e) - 1, limit=LIMIT))
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + step
        if len(k) < LIMIT:
            break
    return rows


def save(path, rows):
    df = pd.DataFrame([r[:12] for r in rows], columns=F.K5COLS)
    for c in F.K5COLS:
        df[c] = pd.to_numeric(df[c])
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df.to_csv(path, index=False)
    return df


def segments(D_list):
    w = sorted((d - F.PRE_BARS * F.B5, d + F.POST_BARS * F.B5) for d in D_list)
    out = []
    for a, b in w:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(df, a, b):
    if df is None or not len(df):
        return False
    o = df.open_time.values.astype("int64")
    need = np.arange(a, b, F.B5)
    return bool(np.isin(need, o).all())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--live-only", action="store_true")
    a = ap.parse_args()
    assert os.path.exists(F.PREREG), "pre-register first"
    for d in (F.K5_DIR, F.B1_DIR, F.LIVE_DIR):
        os.makedirs(d, exist_ok=True)
    now = int(time.time() * 1000)
    # ── live: 1m + 5m for EPIC / DUSK ──
    for sym, ts in LIVE:
        te = int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)
        D = (te // F.B5) * F.B5
        s5, e5 = D - F.PRE_BARS * F.B5, min(now, D + F.POST_BARS * F.B5)
        save(os.path.join(F.LIVE_DIR, f"{sym}_5m.csv"), fetch_range(sym, "5m", s5, e5, F.B5))
        save(os.path.join(F.LIVE_DIR, f"{sym}_1m.csv"), fetch_range(sym, "1m", D - 120 * F.MIN, min(now, D + 720 * F.MIN), F.MIN))
        print("live", sym, "calls", STATE["calls"], "w", STATE["w"], flush=True)
    # ── BTC 1m tail (cache ends ~2026-10-12 04:16 UTC) ──
    b0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)
    bp = os.path.join(F.B1_DIR, "BTCUSDT.csv")
    have = pd.read_csv(bp) if os.path.exists(bp) else None
    if have is None or have.open_time.max() < now - 10 * F.MIN:
        s = b0 if have is None else int(have.open_time.max()) + F.MIN
        save(bp, fetch_range("BTCUSDT", "1m", s, now, F.MIN))
    print("btc tail done calls", STATE["calls"], flush=True)
    if a.live_only:
        return
    # ── cohort windows ──
    K = pd.read_csv(F.W.COHORT, low_memory=False)
    K = K[K.block.isin(["BT", "MASTER", "LIVE"])]
    K["D"] = (K.te.astype("int64") // F.B5) * F.B5
    pairs = sorted(K.pair.unique())
    t0 = time.time()
    for i, p in enumerate(pairs):
        path = os.path.join(F.K5_DIR, f"{p}.csv")
        df = pd.read_csv(path) if os.path.exists(path) else None
        for s, e in segments(K[K.pair == p].D.astype("int64").tolist()):
            e = min(e, now)
            if covered(df, s, e):
                continue
            rows = fetch_range(p, "5m", s, e, F.B5)
            if rows:
                df = save(path, rows)
        if i % 20 == 0:
            el = (time.time() - t0) / 60
            print(f"{i + 1}/{len(pairs)} {p} calls {STATE['calls']} used-weight {STATE['w']} ({el:.1f} min)", flush=True)
    print("done calls", STATE["calls"])


if __name__ == "__main__":
    main()
