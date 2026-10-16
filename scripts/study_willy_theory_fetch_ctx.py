#!/usr/bin/env python3
"""WILLY THEORY — master + live context (the local cache ends 2026-10-12 04:05): public Binance USDⓈ-M klines for the 19 master + 8 live
WILLY fills — the pair's 5m bars (spike − 8 h → entry + 95 min, with taker volume), BTC / ETH / SOL 1m (entry − 60 → + 35 min) and the
top-50 universe's 5m bars (entry − 60 → + 35 min; the last universe_daily_top50 day ≤ the fill day). Same budget / cooldown rules as
study_willy_theory_fetch_taker (shared helpers). Cache: reports/willy_theory/ctx/<kind>_<pair>_<startms>.csv.
"""
import os, sys, time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402
import study_willy_theory_fetch_taker as FT  # noqa: E402

CD = os.path.join(W.OUT, "ctx")
os.makedirs(CD, exist_ok=True)


def fills():
    out = []
    for f, blk in ((W.MASTER, "MASTER"), (W.LIVE, "LIVE")):
        d = pd.read_csv(f, low_memory=False)
        d = d[d.entry_strategy == "FRENZY_WILLY"]
        for r in d.itertuples():
            te = int(W.ms(pd.Series([str(r.opened_at)[:19]]))[0])
            sp = int(W.ms(pd.Series([str(r.entry_frenzy_spike_at)[:19]]))[0])
            out.append((blk, r.pair, te, sp))
    return out


def top50_for(day_ms):
    U = pd.concat([pd.read_csv(os.path.join(W.REP, f)) for f in ("backtest_cache_2025/universe_daily_top50.csv", "backtest_cache/universe_daily_top50.csv")])
    d = U[U.date_ms <= day_ms]
    last = d.date_ms.max()
    return sorted(d[d.date_ms == last].pair.unique())


def get(kind, pair, interval, a, lim, last):
    f = os.path.join(CD, f"{kind}_{pair}_{a}.csv")
    if os.path.exists(f):
        return last
    w = FT.PACE_S * (2 if lim >= 100 else 1) - (time.time() - last)
    if w > 0:
        time.sleep(w)
    last = time.time()
    r = requests.get(FT.URL, params=dict(symbol=pair, interval=interval, startTime=a, limit=lim), timeout=15)
    used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
    if r.status_code in (418, 429):
        FT.arm_cooldown(r.status_code, r.headers.get("Retry-After"))
        sys.exit(f"STOP {r.status_code} — cooldown armed")
    rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7], x[8], x[9], x[10]] for x in r.json()] if r.status_code == 200 else []
    pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol", "qvol", "n", "tbv", "tbq"]).to_csv(f, index=False)
    if used >= 900:
        time.sleep(30)
    return last


def main():
    if FT.cooldown_until() > time.time() * 1000:
        sys.exit("cooldown active — not fetching")
    last = 0.0
    F = fills()
    for k, (blk, pair, te, sp) in enumerate(F):
        a = sp - 8 * W.HOUR_MS
        last = get("p5", pair, "5m", a, int((te + 95 * W.MIN - a) // W.BAR_MS) + 1, last)
        T = (te // W.BAR_MS) * W.BAR_MS - W.BAR_MS
        for m in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            last = get("m1", m, "1m", T - 60 * W.MIN, 99, last)
        for p in top50_for((te // W.DAY_MS) * W.DAY_MS - W.DAY_MS):
            last = get("u5", p, "5m", T - 60 * W.MIN, 20, last)
        print(f"{k + 1}/{len(F)} {blk} {pair}", flush=True)


if __name__ == "__main__":
    main()
