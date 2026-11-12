#!/usr/bin/env python3
"""B25 on a COMPOUNDING account path (operator 11-12): every trade sized off the account value AT ITS OPEN (X × equity, today's sizes), P&L booked
at its close in time order, free balance = equity − margin of the trades open at that moment. Three sizings: FLAT (no Binance limits) · TODAY
(limits, same margin → lower leverage, smaller position) · OPTION (limits, keep the position with more margin, ≤ 1.6× margin, not ML).
Limits = today's bracket table — they bind HARDER as the account grows (bigger positions on the same small coins). DCR = (E_end/E_0)^(1/days) − 1."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import study_bracket_lev_step_1112 as B
from services.trading_engine import leverage_bracket_limit

K, k = B.K, B.k
X0 = {s: v[3] for s, v in K.SLEEVES.items()}
KEEP = {"LITE", "RETRY", "WILLY", "HT", "FLAT"}
G = k[k.era == "B25"].sort_values("o")
days = pd.DatetimeIndex(np.concatenate([pd.date_range(a.normalize(), b.normalize(), freq="D").values for e, a, b in B.per if e == "B25"]))


def keep(tiers, lev, target, max_m):
    best = None
    for cap, tl in sorted(tiers):
        le = float(int(min(lev, tl)))
        if le < 1:
            continue
        n = min(target, cap, max_m * le)
        if best is None or n > best[1] + 1e-9 or (abs(n - best[1]) <= 1e-9 and le > best[0]):
            best = (le, n)
    return best


def run(mode, E0=10_000.0):
    ev = [(r.o, 0, i) for i, r in G.iterrows()] + [(r.c, -1, i) for i, r in G.iterrows()]   # closes before opens at the same instant
    ev.sort(key=lambda x: (x[0], x[1]))
    E, open_, pos, daily = E0, {}, {}, {}
    sq = 0
    for t, kind, i in ev:
        r = G.loc[i]
        if kind == 0:
            s = r.sleeve
            lev = B.NEW[s]
            mg = X0[s] * E / B.OLD[s]
            want = mg * lev
            if mode == "FLAT":
                n, m = want, mg
            else:
                lim = leverage_bracket_limit(B.T.get(r.pair), lev, mg)
                le = float(int(lim[0])) if lim else lev
                n = min(mg, lim[1] / le) * le if lim else want
                m = n / le
                if mode == "OPTION" and s in KEEP and n < want - 1 and B.T.get(r.pair):
                    b = keep(B.T[r.pair], lev, want, mg * 1.6)
                    if b and b[1] > n:
                        n, m = b[1], b[1] / b[0]
            free = E - sum(open_.values())
            if m > free:
                if free <= 0:
                    continue
                n, m, sq = n * free / m, free, sq + 1
            open_[i], pos[i] = m, n
        elif i in pos:
            pnl = r.pct_sc * pos.pop(i) / 100
            open_.pop(i, None)
            E += pnl
            d = t.normalize()
            daily[d] = daily.get(d, 0) + pnl
    return E, sq


L = ["# B25 on a compounding account path (2026-11-12)", __doc__, "",
     "| Sizing | $10,000 → end | total return | %/day compounded | trades squeezed by free balance |", "|---|---|---|---|---|"]
for mode in ("FLAT", "TODAY", "OPTION"):
    E, sq = run(mode)
    L.append(f"| {mode} | ${E:,.0f} | {(E / 10_000 - 1) * 100:+.1f} % | {((E / 10_000) ** (1 / len(days)) - 1) * 100:+.2f} | {sq} |")
open(os.path.join(K.ROOT, "reports", "B25_COMPOUND_PATH_2026-11-12.txt"), "w").write("\n".join(L))
print("\n".join(L))
