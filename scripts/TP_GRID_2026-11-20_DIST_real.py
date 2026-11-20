#!/usr/bin/env python3
"""TP_GRID_2026-11-20_DIST — STEP 1: the 5 real DISTRIBUTION_SHORT fills of the current batch, re-walked on their 1m path from each REAL
fill price under fixed TP X in {4..10} % (TP = min(20 x ATR15, X); X = 10 = today), today's stop (the fill's own entry_dist_sl_pct) and 24 h.

Fill rules (= the DIST_EXIT engine): stop checked first on the bar HIGH (gap-aware: max(stop, bar open)); the maker TP fills only when the
bar LOW trades THROUGH it (two rules shown: >= 1 tick, as the engine; >= 0.05 %, operator's spec); both in one bar -> stop.
The walk starts on the entry-minute bar (the fill is a few seconds into it).
Costs: entry taker 0.05 % · TP maker 0.02 % · stop taker 0.05 % + slip (model 0.10 %, or the fill's REAL stop slippage vs its stop line)
· time exit taker 0.05 % + 0.10 % · real funding (Binance fundingRate history, cached) for every settlement the position spans.
Parity: X = 10 must reproduce each booked net (pnl / notional).
Master (MASTER_POOL_stacked.csv) holds 0 DISTRIBUTION fills -> nothing to re-price there. N = 5 -> ILLUSTRATIVE ONLY.
Output: reports/TP_GRID_2026-11-20_DIST_data/real_*.csv + printed tables.
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_DIST_real.py
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
K1M = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data", "k1m")
OUT = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data")
XS = [4, 5, 6, 7, 8, 9, 10]
SIZE = 0.61                      # one size: notional = 0.61 x equity per trade (DL 503)
MIN, H1 = 60_000, 3_600_000


def funding(sym, t0, t1):
    """Binance public fundingRate history (cached); weight 1 per call, used-weight header read."""
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"funding_{sym}.json")
    if not os.path.exists(f):
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={t0 - 9 * H1}&endTime={t1 + 25 * H1}&limit=1000"
        with urllib.request.urlopen(url, timeout=20) as r:
            if r.status in (418, 429):
                raise SystemExit(f"Binance {r.status} - stop")
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w and int(w) > 1000:
                raise SystemExit(f"used weight {w} - stop")
            json.dump(json.loads(r.read()), open(f, "w"))
        time.sleep(0.3)
    return [(int(x["fundingTime"]), float(x["fundingRate"])) for x in json.load(open(f))]


def walk(T, E0, sl, tp, t, h, l, c, through, slip_stop, fund):
    """returns kind, exit_ms, net % (short)."""
    stop = E0 * (1 + sl / 100); tpp = E0 * (1 - tp / 100)
    lim = tpp * (1 - through / 100) if through else tpp - 1e-12
    end = T + 24 * H1
    a = np.searchsorted(t, T - T % MIN)
    for i in range(a, len(t)):
        if t[i] >= end:
            X, k, xm = c[i - 1], "TIME", end
            break
        if h[i] >= stop:
            X, k, xm = stop, "STOP", t[i] + MIN
            break
        if l[i] <= lim:
            X, k, xm = tpp, "TP", t[i] + MIN
            break
    else:
        return "OPEN", np.nan, np.nan
    gross = -(X / E0 - 1) * 100
    fpct = sum(r for ft, r in fund if T < ft <= xm) * 100           # short receives positive funding
    cost = 0.05 + (0.02 if k == "TP" else 0.05 + (slip_stop if k == "STOP" else 0.10))
    return k, xm, gross - cost + fpct


def main():
    pd.set_option("display.width", 250)
    d = pd.read_csv(EXPORT, low_memory=False)
    B0 = pd.to_datetime(d.opened_at.min()).normalize()     # batch start: every calendar day of the batch counts
    d = d[(d.status == "CLOSED") & d.entry_dist_tp_pct.notna()].copy()
    if "is_probe" in d.columns:
        d = d[~d.is_probe.fillna(False).astype(bool)]
    d["T_ms"] = (pd.to_datetime(d.opened_at) - pd.Timestamp(0)) // pd.Timedelta("1ms")
    d["Tc_ms"] = (pd.to_datetime(d.closed_at) - pd.Timestamp(0)) // pd.Timedelta("1ms")
    d["notional"] = d.investment * d.leverage
    d["booked"] = d.pnl / d.notional * 100
    stopline = d.entry_price * (1 + d.entry_dist_sl_pct / 100)
    d["real_slip"] = np.where(d.close_reason == "STOP_LOSS", (d.exit_price / stopline - 1) * 100, np.nan)
    d["real_fee_pct"] = d.total_fee / d.notional * 100
    rows = []
    for _, r in d.sort_values("T_ms").iterrows():
        k = pd.read_csv(os.path.join(K1M, r.pair + ".csv"))
        t, h, l, c = k.open_time.values.astype(np.int64), k.h.values, k.l.values, k.c.values
        cov_h = (t[-1] + MIN - r.T_ms) / H1
        fund = funding(r.pair, int(r.T_ms), int(r.Tc_ms))
        for rule, thr in (("1tick", 0.0), ("0.05%", 0.05)):
            for slipname in ("model", "real"):
                ss = 0.10 if slipname == "model" else float(r.real_slip) if np.isfinite(r.real_slip) else 0.10
                for X in XS:
                    tp = min(20 * r.entry_dist_atr_pct, X)
                    kk, xm, net = walk(int(r.T_ms), r.entry_price, r.entry_dist_sl_pct, tp, t, h, l, c, thr, ss, fund)
                    rows.append(dict(pair=r.pair, opened=r.opened_at, booked=r.booked, booked_reason=r.close_reason, real_slip=r.real_slip,
                                     cov_h=cov_h, rule=rule, slip=slipname, X=X, kind=kk, exit=pd.to_datetime(xm, unit="ms") if np.isfinite(xm) else None,
                                     net=net))
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "real_rows.csv"), index=False)
    # parity: X = 10, engine rule, real slip
    P = R[(R.X == 10) & (R.rule == "1tick")].pivot_table(index=["pair", "booked_reason", "booked", "cov_h"], columns="slip", values="net").reset_index()
    P["kind10"] = R[(R.X == 10) & (R.rule == "1tick") & (R.slip == "real")].set_index("pair").kind.reindex(P.pair).values
    print("PARITY (X = 10 = today; booked = pnl / notional incl. real fees + funding)\n", P.round(3).to_string(index=False))
    # grid table per rule / slip
    S = []
    days = pd.date_range(B0, pd.Timestamp("2026-11-20"), freq="D")
    for (rule, slip), g in R.groupby(["rule", "slip"]):
        for X, gx in g.groupby("X"):
            v = gx.net.values
            dd = gx.assign(day=pd.to_datetime(gx.exit).dt.normalize()).groupby("day").net.sum().reindex(days, fill_value=0.0) * SIZE / 100
            eq = np.cumprod(1 + dd.values); dcr = eq[-1] ** (1 / len(days)) - 1
            mdd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
            S.append(dict(rule=rule, slip=slip, X=X, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(), dcr=dcr * 100,
                          maxDD=mdd * 100, worst_day=dd.min() * 100, kinds="/".join(gx.kind.str[0])))
    S = pd.DataFrame(S)
    S.to_csv(os.path.join(OUT, "real_grid.csv"), index=False)
    print("\nGRID (", len(days), "calendar days, one size 0.61 x equity )\n", S.round(2).to_string(index=False))
    W = R[(R.rule == "1tick") & (R.slip == "real")].pivot(index="pair", columns="X", values="net")
    print("\nper fill net % (1tick, real slip)\n", W.round(2).to_string())
    W.to_csv(os.path.join(OUT, "real_perfill.csv"))


if __name__ == "__main__":
    main()
