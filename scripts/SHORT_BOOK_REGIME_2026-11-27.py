#!/usr/bin/env python3
"""SHORT_BOOK_REGIME_2026-11-27 — is there a MARKET variable that marks the days the short sleeves lose together? (operator 11-27: "if it's
the market, there MUST be a market variable"). Research only.

Already refuted (not re-run): BTC 1 h / 4 h / 1 d / 24 h return, BTC daily regime, BTC 1h slope, breadth (bear %), Frenzy flags 24 h,
funding (ENTRY_WL 11-18 T4, EXH_BREADTH DL 643, BATCH28 W/L check). NEW theory: the squeeze regime is a property of the PUMPED-COIN
CROWD itself, not of BTC — when fades on OTHER pumped coins are failing (shorts getting squeezed, coins making new highs), the next fade
fails too (momentum in the alt-pump crowd, shorts trapped, liquidations feed each other). Measurable at entry from CLOSED trades only:
  R1 trailing fade win rate: share of OTHER-coin EXH signals that closed in the 48 h before this entry as winners
  R2 trailing fade P&L: mean net of those trades
  R3 trailing stop count 24 h (other coins)
on the 2-year EXH backtest at today's exit (3,358 fills; signals priced whether or not a slot was free — the whole crowd). Sign split and
terciles, DAY-block bootstrap, DESIGN / HOLDOUT, half-years; then the same variable on the real batch fills (EXH, DIST, FLATBASE).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
H, DAY = 3_600_000, 86_400_000
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)


def trailing(entry_ms, pair, P, win_h=48):
    """P: all priced signals (pair, entry_ms, xm, net). For each trade: other-coin signals with xm in [T − win, T)."""
    P = P.sort_values("xm").reset_index(drop=True)
    xm, nt, pr = P.xm.values, P.net.values, P.pair.values
    out = []
    for T, p in zip(entry_ms, pair):
        a, b = np.searchsorted(xm, T - win_h * H), np.searchsorted(xm, T)
        m = pr[a:b] != p
        v = nt[a:b][m]
        stops24 = int(((nt[a:b][m] < 0) & (xm[a:b][m] >= T - 24 * H)).sum())
        out.append((len(v), (v > 0).mean() if len(v) else np.nan, v.mean() if len(v) else np.nan, stops24))
    return pd.DataFrame(out, columns=["n48", "R1_wr48", "R2_avg48", "R3_stops24"])


def boot(K, g, B=3000, seed=11):
    rng = np.random.default_rng(seed); days = K.day.values; ud = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in ud}
    bs = []
    for _ in range(B):
        s = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))]); gg = g[s]; nn = K.net.values[s]
        bs.append(nn[gg].mean() - nn[~gg].mean())
    return np.percentile(bs, [5, 95])


def main():
    sp = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "EXH_SPIKE_AGE_2026-11-26.py"))
    SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
    D, _ = SA.load_bt()                       # every priced signal (the crowd)
    K = SA.kept(D).reset_index(drop=True)     # the fills the sleeve takes
    T = trailing(K.entry_ms.values, K.pair.values, D)
    K = pd.concat([K, T], axis=1)
    K = K[K.n48 >= 3].reset_index(drop=True)
    rows = []
    for col, good_high in (("R1_wr48", True), ("R2_avg48", True), ("R3_stops24", False)):
        v = K[col]
        for cut, thr in (("median", v.median()), ("tercile-lo", v.quantile(1 / 3))):
            bad = (v <= thr) if good_high else (v > (v.median() if cut == "median" else v.quantile(2 / 3)))
            g = ~bad.values
            lo, hi = boot(K, g)
            r = dict(var=col, cut=cut, thr=round(float(thr), 3), N_bad=int(bad.sum()), bad_avg=K.net[bad].mean(), bad_wr=(K.net[bad] > 0).mean() * 100,
                     N_good=int(g.sum()), good_avg=K.net[g].mean(), gap=K.net[g].mean() - K.net[bad].mean(), ci90_lo=lo, ci90_hi=hi)
            for blk, m in (("DESIGN", K.entry_ms < HOL), ("HOLDOUT", K.entry_ms >= HOL)):
                m = m.values
                r[blk] = K.net[m & g].mean() - K.net[m & bad.values].mean()
            hv = [K.net[(K.half == h) & g].mean() - K.net[(K.half == h) & bad.values].mean() for h in sorted(K.half.unique())]
            r["halves"] = f"{sum(x > 0 for x in hv)}/{len(hv)}"
            rows.append(r)
    Tb = pd.DataFrame(rows)
    os.makedirs(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data"), exist_ok=True)
    Tb.to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "short_book_regime_bt.csv"), index=False)
    print(Tb.round(2).to_string(index=False))
    # dose-response on R1
    K["R1b"] = pd.cut(K.R1_wr48, [-0.01, 0.3, 0.4, 0.5, 0.6, 1.0])
    print(K.groupby("R1b", observed=True).agg(N=("net", "size"), days=("day", "nunique"), wr=("net", lambda x: (x > 0).mean() * 100), avg=("net", "mean")).round(2))


if __name__ == "__main__":
    main()
