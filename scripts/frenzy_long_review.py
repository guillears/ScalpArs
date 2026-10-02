#!/usr/bin/env python3
"""🔥 FRENZY_LONG design review — the exact designed cell on the strict ruler, with the design's own limits and sizing.
  entry   scripts/staircase_long_tight_test.py (state turns ON after ≥ 1 h off, 24 h volume ≥ $100M) · one position per pair · at most 2 open at once
  exit    stop 3 % · trail arms +5 %, gives back 1.5 % · 12 h cap; neighbours shown, nothing re-fit
  ruler   1-minute bars, gap-aware fills, 0.10 % slippage on every stop / trail fill, 0.11 % costs, real funding (the long pays)
  checks  costs · clusters (day / pair) · leave-one-month-out · pair concentration · best trades removed · slot limit · account path by size
Usage: venv/bin/python scripts/frenzy_long_review.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
import run_case_study as C  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_LONG_REVIEW_2026-10-02.md")
MAIN = (3.0, 5.0, 1.5); CELLS = (MAIN, (3.0, 5.0, 2.0), (3.0, 3.0, 1.5), (2.0, 5.0, 1.5), (2.2, 5.0, 1.5))


def run(G, paths, S, A, T, slots=None, **kw):
    rows = []; free = {}; open_ends = []
    for r in G.itertuples():
        if r.t < free.get(r.pair, 0) or r.Index not in paths:
            continue
        if slots:
            open_ends = [e for e in open_ends if e > r.t]
            if len(open_ends) >= slots:
                continue
        res, mins, how = H.walk(*paths[r.Index], r.entry, S, A, T, side=1, **kw); end = r.t + mins * 60_000; free[r.pair] = end; open_ends.append(end)
        rows.append((r.Index, res, mins, how))
    return pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i")


if __name__ == "__main__":
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        t = d.open_time.values.astype("int64"); cq = np.concatenate([[0.0], np.cumsum(d.qvol.values)])
        pc = d.c.shift(1); trng = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs())); atr = (trng.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100).values
        for x in C.long_triggers(d):
            i = int(np.searchsorted(t, x["t"])); q24 = cq[i] - cq[max(i - 288, 0)]
            if q24 >= 100e6:
                tr.append(dict(pair=pair, t=x["t"], entry=x["entry"], stretch=(x["entry"] / x["vwap"] - 1) * 100, run=(x["entry"] / x["base"] - 1) * 100, q24=q24, atr=float(atr[i - 1])))
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["month"] = TR.day.str[:7]
    paths = {}
    for r in TR.itertuples():
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def net(G, S, A, T, slots=None, **kw):
        E = run(G, paths, S, A, T, slots, **kw); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        f = [0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
        X["net"] = E.r - B.COST + np.array(f); return X

    hv = lambda X: f"{X[X.day < B.SPLIT].net.mean():+.2f} / {X[X.day >= B.SPLIT].net.mean():+.2f}"
    cell = lambda c: f"stop {c[0]:g} · trail {c[1]:g}/{c[2]:g}"
    lrun = lambda X: max((len(s) for s in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
    L = ["# 🔥 FRENZY_LONG — review of the designed cell", "", f"{len(TR):,} entries on {TR.pair.nunique()} pairs, {TR.day.nunique()} days, Jan–Sep 2026. % of position at 1×, after 0.11 % and funding.", "",
         "## 1 · Costs", "", "| Exit | as tested | +0.10 % slippage | gap-aware + 0.10 % | gap-aware + 0.30 % | halves (gap-aware + 0.10 %) |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        Xg = net(TR, *c, slip=0.10, gap=True)
        L.append(f"| {cell(c)} | {net(TR, *c).net.mean():+.2f} | {net(TR, *c, slip=0.10).net.mean():+.2f} | {Xg.net.mean():+.2f} | {net(TR, *c, slip=0.30, gap=True).net.mean():+.2f} | {hv(Xg)} |")
    L += ["", "From here: gap-aware fills + 0.10 % slippage.", "", "## 2 · Who carries it", "",
          "| Exit · limit | trades | won | stopped | per trade | halves | 95 % by day | 95 % by pair | worst with a month removed | best 5 % removed | top 3 pairs' share | longest losing run |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    KEEP = {}
    for c in CELLS:
        for slots in (None, 2):
            X = net(TR, *c, slots, slip=0.10, gap=True); KEEP[(c, slots)] = X; bd, bp = H.boot(X, "day"), H.boot(X, "pair"); tot = X.net.sum()
            lom = min((X[X.month != m].net.mean(), m) for m in X.month.unique()); s = X.net.sort_values(); top = X.groupby("pair").net.sum().sort_values(ascending=False)
            L.append(f"| {cell(c)} · {'max 2 open' if slots else 'no slot limit'} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(X.how == 'stop').mean() * 100:.0f}% | {X.net.mean():+.2f} | {hv(X)} | [{bd[0]:+.2f}, {bd[1]:+.2f}] | [{bp[0]:+.2f}, {bp[1]:+.2f}] | "
                     f"{lom[0]:+.2f} (−{lom[1]}) | {s.iloc[:-max(1, int(len(s) * 0.05))].mean():+.2f} | {(f'{top.head(3).sum() / tot * 100:.0f}%' if tot > 0 else 'n/a')} | {lrun(X)} |")
    X = KEEP[(MAIN, 2)]
    X["stop_atr"] = 3.0 / X.atr
    L += ["", f"**By month — {cell(MAIN)}, max 2 open**", "", "| Month | trades | pairs | won | per trade | total |", "|---|---|---|---|---|---|"]
    L += [f"| {m} | {len(v)} | {v.pair.nunique()} | {(v.net > 0).mean() * 100:.0f}% | {v.net.mean():+.2f} | {v.net.sum():+.0f} |" for m, v in X.groupby("month")]
    L += ["", f"Trades per day when it trades: median {X.groupby('day').size().median():.0f}, max {X.groupby('day').size().max()}; {X.day.nunique()} trading days of {TR.day.nunique()}. Median hold {X.mins.median():.0f} min.", "",
          "## 3 · Entry cuts (after the fact — descriptive)", "", "| Cut | trades · per trade (halves) |", "|---|---|"]
    for name, col, edges in (("5m ATR % at entry", "atr", (0, 1.0, 1.5, 2.0, 3.0, 1e9)), ("Stop as a multiple of ATR (3 % ÷ ATR)", "stop_atr", (0, 1.0, 1.5, 2.0, 3.0, 1e9)), ("Entry above the average price", "stretch", (0, 3, 6, 10, 20, 1e9)), ("Run since the spike", "run", (-100, 15, 30, 60, 100, 1e9)), ("24 h volume $M", "q24", (100e6, 250e6, 500e6, 1e9, 1e13))):
        for a, b in zip(edges, edges[1:]):
            v = X[(X[col] >= a) & (X[col] < b)]; k = 1e6 if col == "q24" else 1
            L.append(f"| {name} {a / k:g} to {'+' if b >= 1e9 else f'{b / k:g}'} | " + (f"{len(v)} · {v.net.mean():+.2f} ({hv(v)}) |" if len(v) >= 20 else f"{len(v)} · – |"))
    L += ["", "## 4 · The account at each size (max 2 open, 4 slots → one slot = 25 % of the account, 20×, compounding, trades in time order)", "",
          "| Investment multiplier | one stop costs | account after the year | deepest drawdown | lowest point |", "|---|---|---|---|---|"]
    r = X.sort_values("t").net.values
    for m in (2.0, 1.0, 0.5, 0.25):
        k = 0.25 * 20 * m / 100; eq = 1.0; peak = 1.0; dd = 0.0; low = 1.0
        for v in r:
            eq *= max(0.0, 1 + v * k); peak = max(peak, eq); dd = max(dd, 1 - eq / peak); low = min(low, eq)
            if eq <= 0:
                break
        L.append(f"| {m:g}× | {3.21 * k * 100:.0f}% of the account | {'wiped out' if eq <= 0.01 else f'×{eq:.2f}'} | −{dd * 100:.0f}% | ×{low:.2f} |")
    L += ["", "(Sequential approximation: overlapping trades are applied one after another; exchange position limits and the liquidity cap are ignored — they shrink real size on small pairs.)", "",
          "## NOT tested", "", "- Delisted pairs; order-book slippage beyond 0.30 %; the live exchange backstop (2.5 %) inside the 3 % stop.",
          "- The engine's scan delay: the test enters at the next 5m open; live enters on the next scan after the close.",
          "- No random-entry null for the long (the short review had one)."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
