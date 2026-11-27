#!/usr/bin/env python3
"""BATCH28 — the current batch re-read on TODAY's full stack (operator 2026-11-27: "show me current batch before and after with the actual
current filters and TP and SL, and the new EXH filters too"). Research only.
Per sleeve: AS TRADED (booked) → TODAY'S STACK (every filter armed now applied to fills taken before it; every exit re-priced to today's
TP / SL on the fill's own 1m path) → + the DL 650 EXH blocks. One size 0.61 × equity; DCR over every calendar day 11-18 → 11-27.
  EXH   re-priced at TP min(20×ATR, 7) / SL min(1.5×ATR, 6) / 24 h (reports/BATCH28_REVIEW_2026-11-27_data/exh_batch_repriced.csv,
        parity 23 / 23) · filters: no-spot, SSC_OIVOL · new: BEAR80, EARLY_SPIKE.
  DIST  exit unchanged since deploy (Exit B) and its RSI1h filter armed from the start → booked = today.
  FB    exit unchanged (TP −5 / stop +10 incl. funding / 72 h; the funding-stop DL 607 armed 11-21 — every batch stop already ≥ the price
        stop) · FUNDING_ABOVE_BASE (DL 601, 11-20) applied: entry funding > +0.005 % → out.
  HT    TP +5 → +6 net (DL 608, 11-21) re-walked on 1m: long, TP = entry × (1 + 6.063 %), stop = entry × (1 − 3.937 %) (net −4 after taker
        fees), 24 h, stop first on a bar touching both · filters armed later re-read from stamps: NEAR_HIGH dip < 1.4 (DL 594),
        BURST_REJECTION pre_vol > 1.60 ∧ cloc15 < 0.50 (DL 619), FAILED_SPIKE vs_vwap ≤ 0 (DL 509/531).
  WILLY unchanged (V2b exit, 2 fills).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, R)
from services.watch_filters import ht_burst_state  # noqa: E402
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-27_15-23-36.csv")
X, D0, D1 = 0.61, pd.Timestamp("2026-11-18"), pd.Timestamp("2026-11-27")
MIN = 60_000


def stats(net, closed):
    net = pd.Series(np.asarray(net, float)); c = pd.to_datetime(pd.Series(closed).values).normalize()
    if not len(net):
        return dict(N=0)
    days = pd.date_range(D0, D1); r = (net * X / 100).groupby(c).sum().reindex(days).fillna(0); eq = (1 + r).cumprod()
    return dict(N=len(net), WR=round((net > 0).mean() * 100), avg=round(net.mean(), 2), sum=round(net.sum(), 1),
                DCR=round((eq.iloc[-1] ** (1 / len(days)) - 1) * 100, 2), maxDD=round((eq / eq.cummax().clip(lower=1) - 1).min() * 100, 1),
                worst=round(r.min() * 100, 1))


def k1m(sym, t0, t1):
    rows, s = [], t0
    while s < t1:
        js = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=sym, interval="1m", startTime=s, endTime=t1, limit=1500),
                          headers={"User-Agent": "research-client/1.0"}, timeout=30).json()
        if not js:
            break
        rows += js; s = int(js[-1][0]) + MIN
        if len(js) < 1500:
            break
        time.sleep(0.3)
    return pd.DataFrame([[int(r[0]), float(r[2]), float(r[3]), float(r[4])] for r in rows], columns=["t", "h", "l", "c"])


def ht_walk(r):
    T = int(pd.Timestamp(r.opened_at).value // 10**6); E = float(r.entry_price)
    b = k1m(r.pair, T // MIN * MIN, T + 24 * 3_600_000)
    b = b[b.t >= T // MIN * MIN]
    tp, sl = E * 1.06063, E * (1 - 0.03937)
    for t, h, l in zip(b.t, b.h, b.l):
        if l <= sl:
            return -4.0, "SL", t + MIN
        if h >= tp * 1.0005:
            return 6.0, "TP", t + MIN
    c = float(b.c.iloc[-1]); return (c / E - 1) * 100 - 0.09, "TIME", int(b.t.iloc[-1]) + MIN


def main():
    o = pd.read_csv(EXPORT, low_memory=False); o = o[o.status == "CLOSED"]
    rows = []
    # EXH
    K = pd.read_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "exh_batch_repriced.csv"))
    ex = o[o.entry_strategy == "EXHAUSTION_SHORT"]
    rows.append(dict(sleeve="EXH", step="as traded", **stats(ex.pnl_percentage, ex.closed_at)))
    Kk = K[~K.blk_nospot & ~K.blk_ssc]; ct = pd.to_datetime(Kk.xm, unit="ms")
    rows.append(dict(sleeve="EXH", step="today's stack (TP 7, filters)", **stats(Kk.net, ct)))
    z = (Kk.entry_bear_pct > 80) | (Kk.hrs < 0.5)
    rows.append(dict(sleeve="EXH", step="+ DL 650 BEAR80 + EARLY_SPIKE", **stats(Kk.net[~z], ct[~z])))
    # DIST
    di = o[o.entry_strategy == "DISTRIBUTION_SHORT"]
    for st in ("as traded", "today's stack before DL 651 (= as traded)"):
        rows.append(dict(sleeve="DIST", step=st, **stats(di.pnl_percentage, di.closed_at)))
    # 📉🌡 (DL 651) DIST_CRASH_FUNDING: every batch day had BTC 23–30 % below its 30-day high → the funding cut decides
    dkeep = ~(pd.to_numeric(di.entry_funding_rate, errors="coerce") >= 0.00005 - 1e-12)
    rows.append(dict(sleeve="DIST", step=f"+ DL 651 crash-funding (drops {', '.join(di.pair[~dkeep].str.replace('USDT', ''))})",
                     **stats(di.pnl_percentage[dkeep], di.closed_at[dkeep])))
    # FB
    fb = o[o.entry_strategy == "FLATBASE_SHORT"]
    rows.append(dict(sleeve="FB", step="as traded", **stats(fb.pnl_percentage, fb.closed_at)))
    keep = ~(pd.to_numeric(fb.entry_funding_rate, errors="coerce") > 0.00005)
    rows.append(dict(sleeve="FB", step=f"today's stack (FUNDING_ABOVE_BASE drops {', '.join(fb.pair[~keep])})", **stats(fb.pnl_percentage[keep], fb.closed_at[keep])))
    # HT
    ht = o[o.entry_strategy == "FRENZY_WILLY_HT"].copy()
    rows.append(dict(sleeve="HT", step="as traded", **stats(ht.pnl_percentage, ht.closed_at)))
    w = [ht_walk(r) for r in ht.itertuples()]
    ht["n6"], ht["why6"], ht["x6"] = [a for a, _, _ in w], [b for _, b, _ in w], [pd.to_datetime(c, unit="ms") for _, _, c in w]
    pv, cl = pd.to_numeric(ht.entry_frenzy_pre_vol15_vs60, errors="coerce"), pd.to_numeric(ht.entry_frenzy_cloc15, errors="coerce")
    burst = [ht_burst_state(a, b) == "zone" for a, b in zip(pv, cl)]
    near = pd.to_numeric(ht.entry_frenzy_willy_dip_pct, errors="coerce") < 1.4
    failed = pd.to_numeric(ht.entry_frenzy_vs_vwap_pct, errors="coerce") <= 0
    blk = np.array(burst) | near.fillna(False).values | failed.fillna(False).values
    print(ht[["pair", "opened_at", "pnl_percentage", "n6", "why6"]].assign(burst=burst, near=near.values, failed=failed.values).to_string(index=False))
    rows.append(dict(sleeve="HT", step="TP 6 re-walk (no filter change)", **stats(ht.n6, ht.x6)))
    rows.append(dict(sleeve="HT", step=f"today's stack (TP 6 + filters drop {', '.join(ht.pair[blk].str.replace('USDT', '') + ' ' + ht.opened_at[blk].str[5:13])})",
                     **stats(ht.n6[~blk], ht.x6[~blk])))
    wl = o[o.entry_strategy == "FRENZY_WILLY"]
    rows.append(dict(sleeve="WILLY", step="as traded = today", **stats(wl.pnl_percentage, wl.closed_at)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "today_stack_batch.csv"), index=False)
    print(T.to_string(index=False))
    # whole book: as traded · today's stack (before DL 650 / 651) · + DL 650 + DL 651
    days = pd.date_range(D0, D1)
    def ser(net, c):
        return (pd.Series(np.asarray(net, float)) * X / 100).groupby(pd.to_datetime(pd.Series(c).values).normalize()).sum().reindex(days).fillna(0)
    def m(r):
        eq = (1 + r).cumprod()
        return dict(DCR=round((eq.iloc[-1] ** (1 / len(r)) - 1) * 100, 2), maxDD=round((eq / eq.cummax().clip(lower=1) - 1).min() * 100, 1),
                    worst=round(r.min() * 100, 1), sum=round(r.sum() * 100, 1))
    fbk = fb[keep]
    common = ser(wl.pnl_percentage, wl.closed_at) + ser(fbk.pnl_percentage, fbk.closed_at) + ser(ht.n6, ht.x6)
    before = ser(o.pnl_percentage, o.closed_at)
    stack = common + ser(Kk.net, ct) + ser(di.pnl_percentage, di.closed_at)
    new = common + ser(Kk.net[~z], ct[~z]) + ser(di.pnl_percentage[dkeep], di.closed_at[dkeep])
    print("\nWHOLE BOOK (one size 0.61, % of equity):")
    for lab, r in (("as traded", before), ("today's stack before DL 650 / 651", stack), ("+ DL 650 EXH + DL 651 DIST filters", new)):
        print(f"  {lab}: {m(r)}")


if __name__ == "__main__":
    main()
