#!/usr/bin/env python3
"""🎯 LITE C revert CF (DECISION_LOG 420): LITE / RETRY fills that really closed at the LITE C +2 target have a peak cut at +2, so the master stack cannot re-price them at
today's +3 from peak_pnl. This walks each one from its entry on 1m bars at +3 / −3 / 12 h (stop first inside a minute, against the revert) → reports/LITE_C_REVERT_CF.csv,
read by build_master_pool.py. Run it before archiving a batch that opened LITE fills before the revert. Public klines, cache-first.
Usage: venv/bin/python scripts/build_lite_c_revert_cf.py"""
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG                 # noqa: E402
import scout_long_stop4 as LS            # noqa: E402

ROOT = os.path.dirname(_D)
OUT = os.path.join(ROOT, "reports", "LITE_C_REVERT_CF.csv")
FAMILY = ("FRENZY_LITE", "FRENZY_LITE_RETRY")
COLS = ("opened_at", "pair", "entry_strategy", "status", "entry_price", "pnl_percentage", "frenzy_lite_tp_used", "close_reason")
CACHE = os.path.join(ROOT, "reports", "backtest_cache", "scout_regime_study")


def walk(bars, entry, tp=3.0, sl=3.0):
    """bars = klines (open time, o, h, l, c, …) → the +tp / −sl / last-close result, % from the entry; None without bars."""
    if not bars or not entry:
        return None
    for q in bars:
        hi, lo = (float(q[2]) / entry - 1) * 100, (float(q[3]) / entry - 1) * 100
        if lo <= -sl:
            return -sl
        if hi >= tp:
            return tp
    return (float(bars[-1][4]) / entry - 1) * 100


def capped(d):
    """rows that closed at the +2 target (stored TP 2, a profit near it)."""
    pct, tp = pd.to_numeric(d.pnl_percentage, errors="coerce"), pd.to_numeric(d.frenzy_lite_tp_used, errors="coerce")
    return d[d.entry_strategy.astype(str).isin(FAMILY) & (d.status.astype(str) == "CLOSED") & (tp == 2.0) & (pct >= 1.5)]


def main():
    fr = []
    for f in [os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")] + list(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"))):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS).reindex(columns=list(COLS)))
        except Exception:
            continue
    d = capped(pd.concat(fr, ignore_index=True))
    d = d.assign(okey=d.opened_at.astype(str).str.replace(" ", "T").str[:19]).drop_duplicates(["okey", "pair"], keep="last")
    now, bud, rows = int(time.time() * 1000), LS.Budget(), []
    for r in d.itertuples():
        t = pd.Timestamp(r.okey)
        v = walk(LS.fapi_klines(r.pair, "1m", int(t.value // 10**6) // 60000 * 60000, 720, bud, CACHE, now, fetch=True), float(r.entry_price))
        if v is not None:
            rows.append(dict(opened_at=r.okey, pair=r.pair, entry_strategy=r.entry_strategy, booked=round(float(r.pnl_percentage), 4), cf_pct=round(v, 4)))
    if len(rows) < len(d):      # budget / cooldown: never overwrite with a partial file (a dropped fill would stay at +2 silently)
        print(f"⚠ only {len(rows)} of {len(d)} fills priced — {OUT} not written; re-run.")
        return 1
    pd.DataFrame(rows).sort_values("opened_at").to_csv(OUT, index=False)
    print(f"{len(rows)} of {len(d)} fills → {OUT} · requests {bud.requests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
