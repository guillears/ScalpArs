#!/usr/bin/env python3
"""📉 DAY-AFTER SHORT — operator (2026-10-02): "the pairs that ran yesterday are the ones falling today … shorting yesterday's
pairs would be huge". PRE-DECLARED before any result:
  signal   at each 00:00 UTC, every futures pair whose PREVIOUS UTC day closed ≥ +X % above its open ∧ traded ≥ $20M that day
  trade    SHORT at the 00:00 open, hold 24 h (close at the next 00:00), stop S % above entry (filled at the worse of the stop and
           the breaching 5m bar's open); cost 0.11 %; funding from Binance history (short receives + / pays −)
  grid     X ∈ {20, 30, 50} × S ∈ {10, 20, none} = 9 cells; year cache Jan–Sep 2026 (5m bars, pairs with ≥ 40 days of history)
  PASS     after funding: mean > 0 in both halves ∧ day-clustered 95 % interval above 0 ∧ still > 0 with the best 5 % removed
  context  the frenzy "switch-off" swing short (DECISION_LOG pending) failed exactly the last two conditions.
Usage: venv/bin/python scripts/day_after_short_test.py → reports/DAY_AFTER_SHORT_TEST_2026-10-02.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_swing_review as FS  # noqa: E402  (funding cache + t-quantile via H)
sys.argv = _a
H = FS.H; DAY = 86_400_000; COST = 0.11; OUT = os.path.join(H.ROOT, "reports", "DAY_AFTER_SHORT_TEST_2026-10-02.md")
XS, SS = (20, 30, 50), (10, 20, None)

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(H.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        t = d.open_time.values.astype("int64"); o, h, c, q = d.o.values, d.h.values, d.c.values, d.qvol.values
        day = t // DAY; starts = np.nonzero(np.diff(day, prepend=day[0] - 1))[0]          # first bar of each UTC day
        for a, b, e in zip(starts[:-2], starts[1:-1], starts[2:]):                        # day [a,b) is "yesterday", [b,e) is the trade
            if t[a] % DAY != 0 or b - a < 280 or e - b < 280:
                continue
            ret = (c[b - 1] / o[a] - 1) * 100; vol = q[a:b].sum()
            if ret < XS[0] or vol < 20e6:
                continue
            ent = o[b]; hh = h[b:e]; res = {}
            for S in SS:
                if S is None:
                    res["none"] = ((1 - c[e - 1] / ent) * 100, e - b)
                else:
                    k = np.nonzero(hh >= ent * (1 + S / 100))[0]
                    res[str(S)] = ((1 - max(ent * (1 + S / 100), o[b + k[0]]) / ent) * 100, int(k[0]) + 1) if len(k) else ((1 - c[e - 1] / ent) * 100, e - b)
            rows.append(dict(pair=pair, t=int(t[b]), ret=ret, vol=vol, worst=(hh.max() / ent - 1) * 100, **{f"r{k}": v[0] for k, v in res.items()}, **{f"b{k}": v[1] for k, v in res.items()}))
    T = pd.DataFrame(rows); T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d"); T["mon"] = T.day.str[:7]
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); fr = {}
    for p in T.pair.unique():
        try:
            fr[p] = FS.funding(p, lo, hi)
        except SystemExit:
            fr[p] = None
    for k in ("10", "20", "none"):
        T[f"f{k}"] = [np.nan if fr[r.pair] is None else fr[r.pair][(fr[r.pair].t > r.t) & (fr[r.pair].t <= r.t + getattr(r, f"b{k}") * 300_000)].rate.sum() * 100 for r in T.itertuples()]
        T[f"n{k}"] = T[f"r{k}"] - COST + T[f"f{k}"].fillna(0)
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "day_after_short_trades.csv"), index=False)
    L = ["# 📉 DAY-AFTER SHORT — short at 00:00 UTC every pair that gained ≥ X % the day before, hold 24 h", "",
         f"{len(T)} pair-days at ≥ +20 % on {T.pair.nunique()} pairs, {T.day.nunique()} days, Jan–Sep 2026. % of position at 1×, after 0.11 % costs and funding.", "",
         "| Yesterday ≥ | Stop | N | days | won | stopped | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | funding / trade | worst against (median) | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for X in XS:
        g = T[T.ret >= X]
        for k in ("10", "20", "none"):
            c_ = f"n{k}"; a1, a2 = g[g.day < H.SPLIT][c_].mean(), g[g.day >= H.SPLIT][c_].mean(); dm = g.groupby("day")[c_].mean(); n = len(dm)
            se = dm.std(ddof=1) / np.sqrt(n); q_ = H.tq(0.975, n); l_, h_ = dm.mean() - q_ * se, dm.mean() + q_ * se
            s_ = g[c_].sort_values(); tr = s_.iloc[:-max(1, int(len(s_) * 0.05))].mean(); stp = (g[f"b{k}"] < 280).mean() * 100 if k != "none" else 0
            L.append(f"| +{X} % | {k} | {len(g)} | {n} | {(g[c_] > 0).mean() * 100:.0f}% | {stp:.0f}% | {a1:+.2f} / {a2:+.2f} | {dm.mean():+.2f} [{l_:+.2f}, {h_:+.2f}] | {tr:+.2f} | {g[f'f{k}'].mean():+.2f} | {g.worst.median():+.1f}% | {'✅' if a1 > 0 and a2 > 0 and l_ > 0 and tr > 0 else '—'} |")
    g = T[T.ret >= 30]; L += ["", "By month (≥ +30 %, 20 % stop): " + " · ".join(f"{m} {len(x)}×{x.n20.mean():+.1f}" for m, x in g.groupby("mon")),
                              f"No-stop tail (≥ +30 %): worst {g.rnone.min():+.0f} % · moved ≥ 30 % against in {(g.worst >= 30).mean() * 100:.0f} % · ≥ 100 % in {(g.worst >= 100).mean() * 100:.1f} % of trades."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
