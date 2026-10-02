#!/usr/bin/env python3
"""🪜 STAIRCASE LONG with a tight stop + trailing exit — the long leg of the sleeve in the form the operator wants (20×, small stop, trailing
take-profit). The ride-it exit (sell on a close below the spike VWAP, 8 % stop) is the version with year evidence; this tests the tight version.
FROZEN before the run:
  entry    the staircase state turns ON after being off for the previous hour (≥ 2 h after the spike ∧ every 5m close of the last hour ≥ the
           spike-anchored VWAP ∧ last-hour volume ≥ 100× normal) ∧ 24 h volume ≥ $100M → LONG at the next open
  NEAR     the same, only when the entry is ≤ 5 % above the VWAP (the MOVR lesson: a stretched entry dips first)
  exits    stop S ∈ {1, 1.5, 2, 3} % × trailing (starts at A %, gives back T %) ∈ {(2, 1), (3, 1), (3, 1.5), (5, 1.5), (5, 2)} · 12 h cap · 1m bars
  limits   none · at most 2 trades per pair per UTC day · pause the pair for the day after 2 stops in a row
  cost     0.11 % + real funding (the long pays +rate / receives −rate)
  PASS     per cell: mean > 0 in both halves ∧ day-clustered 95 % interval above 0
Usage: venv/bin/python scripts/staircase_long_tight_test.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import run_case_study as C  # noqa: E402
sys.argv = _a
B, ST = C.B, C.ST; FS = B.FS; OUT = os.path.join(ST.ROOT, "reports", "STAIRCASE_LONG_TIGHT_TEST_2026-10-02.md"); COST = 0.11; SPLIT = "2026-05-01"
STOPS = (1.0, 1.5, 2.0, 3.0); TRAILS = ((2.0, 1.0), (3.0, 1.0), (3.0, 1.5), (5.0, 1.5), (5.0, 2.0))


def evaluate(G, paths, S, A, T, limit):
    rows = []; free = {}; cnt = {}; streak = {}; paused = {}
    for r in G.itertuples():
        day = r.t // 86_400_000
        if r.t < free.get(r.pair, 0) or r.Index not in paths or (limit == "max2" and cnt.get((r.pair, day), 0) >= 2) or (limit == "pause2" and paused.get(r.pair) == day):
            continue
        h, l, c = paths[r.Index]; res, mins, how = C.walk("LONG", h, l, c, r.entry, S, A, T)
        free[r.pair] = r.t + mins * 60_000; cnt[(r.pair, day)] = cnt.get((r.pair, day), 0) + 1
        streak[r.pair] = streak.get(r.pair, 0) + 1 if how == "stop" else 0
        if streak[r.pair] >= 2:
            paused[r.pair] = day; streak[r.pair] = 0
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
        for x in C.long_triggers(d):
            i = int(np.searchsorted(t, x["t"])); q24 = cq[i] - cq[max(i - 288, 0)]
            if q24 >= 100e6:
                tr.append(dict(pair=pair, t=x["t"], entry=x["entry"], stretch=(x["entry"] / x["vwap"] - 1) * 100, q24=q24))
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    print(len(TR), "long entries ·", int((TR.stretch <= 5).sum()), "near the line", flush=True)
    paths = {}
    for n_, r in enumerate(TR.itertuples()):
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
        if n_ % 300 == 0:
            print(n_, flush=True)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def cell(G, S, A, T, limit):
        E = evaluate(G, paths, S, A, T, limit)
        if len(E) < 20:
            return None
        X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        fund = [0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
        X["net"] = E.r - COST + np.array(fund); a1, a2 = X[X.day < SPLIT].net.mean(), X[X.day >= SPLIT].net.mean()
        dm = X.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); l_, h_ = dm.mean() - ST.tq(n) * se, dm.mean() + ST.tq(n) * se
        s = X.net.sort_values(); st = mx = 0
        for v in X.sort_values("t").net:
            st = st + 1 if v < 0 else 0; mx = max(mx, st)
        return dict(n=len(X), won=(X.net > 0).mean() * 100, stopped=(X.how == "stop").mean() * 100, win=X[X.net > 0].net.mean(), loss=X[X.net <= 0].net.mean(), a1=a1, a2=a2, dm=dm.mean(), lo=l_, hi=h_,
                    trim=s.iloc[:-max(1, int(len(s) * 0.05))].mean(), run=mx, ok=a1 > 0 and a2 > 0 and l_ > 0)
    L = ["# 🪜 STAIRCASE LONG — tight stop + trailing exit, year test (1-minute bars)", "",
         f"{len(TR):,} long entries on ≥ $100M pairs ({TR.pair.nunique()} pairs, {TR.day.nunique()} days), {int((TR.stretch <= 5).sum())} of them ≤ 5 % above the VWAP. % of position at 1× after 0.11 % and funding.", ""]
    for nm, G in (("ALL entries", TR), ("NEAR the line (≤ 5 % above the VWAP)", TR[TR.stretch <= 5])):
        for limit, lab in (("none", "no trade limit"), ("max2", "at most 2 trades per pair per day")):
            L += [f"## {nm} · {lab}", "", "| Stop | Trail (start / give-back) | trades | won | stopped | avg win / loss | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for S in STOPS:
                for A, T in TRAILS:
                    x = cell(G, S, A, T, limit)
                    if x:
                        L.append(f"| −{S:g} % | {A:g} / {T:g} | {x['n']} | {x['won']:.0f}% | {x['stopped']:.0f}% | {x['win']:+.1f} / {x['loss']:+.1f} | {x['a1']:+.2f} / {x['a2']:+.2f} | {x['dm']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] | {x['trim']:+.2f} | {x['run']} | {'✅' if x['ok'] else '—'} |")
            L.append("")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
