#!/usr/bin/env python3
"""📉 OFF-PEAK SHORT test — FROZEN 2026-10-02 before reading: the hostile review of the EMA50 > +200 % short found, after the fact, that entries
15–40 % below the run's peak carried the result (+0.55 / +0.63, both halves) and entries within 15 % of the peak lost (−0.59).
  rule       SHORT the first 5m close below the line when the entry is 15–40 % below the run's highest price so far (−40 ≤ off_peak < −15)
  derived on EMA50 triggers with run > +200 % (499) → that set is IN-SAMPLE and shown only for reference
  judged on  triggers the cut never saw: EMA50 with run +50–200 % (primary) · EMA200, all runs (secondary)
  ruler      1-minute bars, gap-aware stop, 0.10 % slippage on every stop / trail fill, 0.11 % costs, real funding
  exit       stop 2 % · trail from +5 % giving back 3 % (the proposal) + the three neighbours of the review; nothing re-fit
  PASS       on the primary set: per trade > 0 in both halves ∧ 95 % interval by day above 0. The band must also beat the rest (dose table).
Usage: venv/bin/python scripts/break_short_offpeak_test.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST
OUT = os.path.join(ST.ROOT, "reports", "BREAK_SHORT_OFFPEAK_TEST_2026-10-02.md"); BANDS = ((-100, -40), (-40, -25), (-25, -15), (-15, -8), (-8, 0.01))

if __name__ == "__main__":
    TR = pd.read_csv(os.path.join(B.BC, "break_short_triggers.csv")); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["month"] = TR.day.str[:7]
    TR["ep"] = TR.pair + "|" + ((TR.t - TR.hrs * 3600e3) // 3600e3).astype("int64").astype(str)
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

    def net(G, S, A, T):
        E = H.run(G, paths, S, A, T, slip=0.10, gap=True); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        f = [0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()]
        X["net"] = E.r - B.COST + np.array(f); return X

    hv = lambda X: f"{X[X.day < B.SPLIT].net.mean():+.2f} / {X[X.day >= B.SPLIT].net.mean():+.2f}"
    zone = lambda G: G[(G.off_peak >= -40) & (G.off_peak < -15)]
    SETS = (("PRIMARY · EMA50, run +50–200 % (never seen by the cut)", TR[(TR.line == 50) & (TR.gain < 200)]),
            ("SECONDARY · EMA200, all runs (never seen by the cut)", TR[TR.line == 200]),
            ("SECONDARY · EMA200, run > +200 %", TR[(TR.line == 200) & (TR.gain >= 200)]),
            ("REFERENCE (in-sample) · EMA50, run > +200 %", TR[(TR.line == 50) & (TR.gain >= 200)]))
    L = ["# 📉 OFF-PEAK SHORT — short the break only when the entry is 15–40 % below the run's peak (frozen before reading)", "",
         "% of position at 1×, 1-minute bars, gap-aware stop + 0.10 % slippage, after 0.11 % and funding. One position per pair.", ""]
    verdict = []
    for name, G in SETS:
        Z = zone(G)
        L += [f"## {name} — {len(Z)} of {len(G)} triggers are in the band ({Z.pair.nunique()} pairs, {Z.day.nunique()} days)", "",
              "| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | 95 % by pair | worst with one month removed | top 3 pairs' share | longest losing run | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in H.CELLS:
            X = net(Z, *c)
            if len(X) < 20:
                L.append(f"| stop {c[0]:g} · trail {c[1]:g}/{c[2]:g} | {len(X)} | – | | | | | | | | | |"); continue
            bd, bp = H.boot(X, "day"), H.boot(X, "pair"); a1, a2 = X[X.day < B.SPLIT].net.mean(), X[X.day >= B.SPLIT].net.mean(); ok = a1 > 0 and a2 > 0 and bd[0] > 0
            lom = min((X[X.month != m].net.mean(), m) for m in X.month.unique()); top = X.groupby("pair").net.sum().sort_values(ascending=False); tot = X.net.sum()
            run_ = max((len(s) for s in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
            L.append(f"| stop {c[0]:g} · trail {c[1]:g}/{c[2]:g} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(X.how == 'stop').mean() * 100:.0f}% | {X.net.mean():+.2f} | {a1:+.2f} / {a2:+.2f} | [{bd[0]:+.2f}, {bd[1]:+.2f}] | [{bp[0]:+.2f}, {bp[1]:+.2f}] | "
                     f"{lom[0]:+.2f} (−{lom[1]}) | {(f'{top.head(3).sum() / tot * 100:.0f}%' if tot > 0 else 'n/a (total ≤ 0)')} | {run_} | {'✅' if ok else '—'} |")
            if c == H.MAIN:
                verdict.append((name.split(" (")[0], len(X), X.net.mean(), a1, a2, bd, ok))
        L += ["", "**Dose — every band of distance below the peak, proposed exit (trades · per trade · halves)**", "", "| Entry vs peak | result |", "|---|---|"]
        for a, b in BANDS:
            X = net(G[(G.off_peak >= a) & (G.off_peak < b)], *H.MAIN)
            L.append(f"| {a:g} to {b:g} % | {len(X)} · {X.net.mean():+.2f} ({hv(X)}) · {X.pair.nunique()} pairs |" if len(X) >= 20 else f"| {a:g} to {b:g} % | {len(X)} · – |")
        L.append("")
    L += ["## Verdict (proposed exit: stop 2 %, trail from +5 % giving back 3 %)", ""]
    L += [f"- {n}: {k} trades · {m:+.2f} per trade ({a1:+.2f} / {a2:+.2f}) · by day [{bd[0]:+.2f}, {bd[1]:+.2f}] → {'PASS' if ok else 'no pass'}" for n, k, m, a1, a2, bd, ok in verdict]
    L += ["", "## NOT tested", "", "- No fresh time period exists: the year is fully used, so 'unseen' means other run sizes and the other line, not later dates.",
          "- Delisted pairs, exchange position limits on small pairs, slippage beyond 0.10 %.", "- Triggers on one pair in one episode are not independent; the by-pair interval is the stricter read."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
