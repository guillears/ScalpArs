#!/usr/bin/env python3
"""ML impulse study — fetch only what the caches lack (pre-registration reports/ML_IMPULSE_PREREG.txt).

  phase1 SYM   : XRPUSDT / ZECUSDT 1m klines with taker-buy, 2024-12-29 -> now     -> reports/ml_impulse_fetch/k1m_taker/SYM.csv
  live         : live-window top-up (the cache k1m ends 2026-10-12 00:00, k5m_taker 2026-10-13): the LAST universe list's top-10
                 1m (k1m_live) and top-50 5m with taker (k5m_taker), 2026-10-11 22:00 -> now.
Reuses the budget-rule client of scripts/study_ml_btcfwd_fetch.py (<= 1,200 weight/min, pause at used-weight 950 read from
x-mbx-used-weight-1m, stop + shared cooldown file on 418/429, refuse while the cooldown is active). Writes outside backtest_cache.

    venv/bin/python -W ignore scripts/study_ml_impulse_fetch.py phase1 XRPUSDT
    venv/bin/python -W ignore scripts/study_ml_impulse_fetch.py live
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_btcfwd_fetch as BF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BF.OUT = os.path.join(ROOT, "reports/ml_impulse_fetch")
EXCL = {"BTCUSDT", "USDCUSDT", "PAXGUSDT", "XAUTUSDT", "FDUSDUSDT"}
LIVE_FROM = int(pd.Timestamp("2026-10-11 22:00").value // 10**6)


def last_list():
    u = pd.read_csv(os.path.join(ROOT, "reports/backtest_cache/universe_daily_top50.csv"))
    u = u[u.date_ms == u.date_ms.max()].sort_values("rank")
    return [p for p in u.pair if p not in EXCL]


def main():
    if not BF.cooldown_ok():
        sys.exit("shared Binance cooldown active — refusing to start")
    os.makedirs(BF.OUT, exist_ok=True)
    now = int(time.time() * 1000)
    ph = sys.argv[1]
    if ph == "phase1":
        for s in sys.argv[2:]:
            print("phase1", s, flush=True)
            BF.fetch_span(s, "1m", BF.START, now, "k1m_taker")
    elif ph == "live":
        mem = last_list()
        for p in mem[:10]:
            BF.fetch_span(p, "1m", LIVE_FROM, now, "k1m_live")
        for p in mem:
            BF.fetch_span(p, "5m", LIVE_FROM, now, "k5m_taker")
        print("live members", len(mem), "top10", mem[:10], flush=True)
    print(f"DONE requests {BF.NREQ} · last used weight {BF.LASTW}", flush=True)


if __name__ == "__main__":
    main()
