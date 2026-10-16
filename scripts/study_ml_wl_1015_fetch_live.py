"""Minimal Binance top-up for block ① (live fills after the kline cache ends, 2026-10-14 ~16:00 UTC).
Fetches ONLY what the live MOMENTUM LONG fills need: BTC + fill pairs 1m (entry −2 h → close +20 min) and the
top-50 universe 5m (entry −26 h → close +20 min). Writes to reports/ml_wl_1015_fetch/{k1m,k5m_full}/ (never into
reports/backtest_cache). Budget rule: ≤ 1,200 weight/min, reads x-mbx-used-weight-1m, stops on 418/429, honours and
writes the shared cooldown file reports/backtest_cache/.binance_cooldown_until.

    venv/bin/python -W ignore scripts/study_ml_wl_1015_fetch_live.py
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_wl_1015_common import ROOT, TOPUP, load_live, load_universe, universe_for  # noqa: E402

COOL = os.path.join(ROOT, "reports/backtest_cache/.binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit=499"


def cooldown_ok():
    try:
        return not _BC.active(COOL)
    except Exception:
        return True


def get(sym, iv, a, b):
    if not cooldown_ok():
        sys.exit("shared Binance cooldown active — stopping")
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=a, b=b), headers={"User-Agent": "scalpars-study"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            w = int(r.headers.get("x-mbx-used-weight-1m", "0"))
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            open(COOL, "w").write(str(time.time() + 600))
            sys.exit(f"HTTP {e.code} — cooldown written, stopping")
        raise
    if w > 900:
        print(f"  used weight {w} — pausing 60 s"); time.sleep(60)
    return data, w


def save(sub, sym, rows):
    d = os.path.join(TOPUP, sub); os.makedirs(d, exist_ok=True)
    df = pd.DataFrame([[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    df.to_csv(os.path.join(d, sym + ".csv"), index=False)


L = load_live()
if not len(L):
    sys.exit("no live MOMENTUM LONG fills")
days, keys = load_universe()
need1, need5 = {}, {}
for r in L.itertuples():
    t = int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6)
    c = int(pd.Timestamp(str(r.closed_at)[:19]).value // 10**6)
    for s in ("BTCUSDT", r.pair):
        a, b = need1.get(s, (t, c))
        need1[s] = (min(a, t - 2 * 3600_000), max(b, c + 20 * 60_000))
    univ, _ = universe_for(days, keys, t)
    for s in univ + [r.pair, "BTCUSDT"]:
        a, b = need5.get(s, (t, c))
        need5[s] = (min(a, t - 26 * 3600_000), max(b, c + 20 * 60_000))
tot = 0
for s, (a, b) in need1.items():
    rows, w = get(s, "1m", a, b); save("k1m", s, rows); tot += 1
for s, (a, b) in need5.items():
    rows, w = get(s, "5m", a, b)
    if rows:
        save("k5m_full", s, rows)
    tot += 1
    time.sleep(0.15)
print(f"fetched {tot} series · last used weight {w}")
