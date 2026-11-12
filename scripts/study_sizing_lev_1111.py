#!/usr/bin/env python3
"""Leverage step proposal (operator 11-11): Momentum Long 10× → 15× · WILLY 5× → 8× · RETRY 7× → 10× · Flatbase 5× → 8× (HT, LITE unchanged) —
per sleeve before → after on B25 and master (today's stack, $10,000 account, money in unchanged → exposure × new/old leverage).
WR and avg % are the same trades; $, loss per stop, daily compound and worst day move. One row per step + all together."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_sizing_kelly_1111 as K
ACC = 10_000
STEP = {"ML": 15 / 10, "WILLY": 8 / 5, "HT": 8 / 5, "RETRY": 10 / 7, "FLAT": 8 / 5}
NAMES = {"ML": "Momentum Long 10→15×", "LITE": "LITE (8×, unchanged)", "RETRY": "LITE RETRY 7→10×", "WILLY": "WILLY 5→8×", "HT": "WILLY HT 5→8×", "FLAT": "Flatbase 5→8×"}
k, per = K.master()
X0 = {s: v[3] for s, v in K.SLEEVES.items()}

def days_of(eras):
    return pd.DatetimeIndex(np.concatenate([pd.date_range(t0.normalize(), t1.normalize(), freq="D").values for e, t0, t1 in per if e in eras]))

def daily(G, X, days, sleeves):
    r = pd.Series(0.0, index=days)
    for s in sleeves:
        g = G[G.sleeve == s]
        r = r.add((g.pct * X[s] / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0), fill_value=0.0)
    return r

def cmp(r):
    return ((1 + r).prod() ** (1 / len(r)) - 1) * 100

def table(scope, eras):
    days = days_of(eras); G = k[k.era.isin(eras)]
    X1 = {s: X0[s] * STEP.get(s, 1.0) for s in X0}
    L = [f"\n## {scope} — {len(days)} days · ${ACC:,} account",
         "| Sleeve | N | WR | Avg % | Loss per stop now → new | Net $ now → new | %/day now → new |", "|---|---|---|---|---|---|---|"]
    for s in K.SLEEVES:
        g = G[G.sleeve == s]
        if not len(g):
            continue
        st = K.SLEEVES[s][2]
        n0, n1 = (g.pct * X0[s] / 100 * ACC).sum(), (g.pct * X1[s] / 100 * ACC).sum()
        d0, d1 = cmp(daily(G, X0, days, [s])), cmp(daily(G, X1, days, [s]))
        L.append(f"| {NAMES[s]} | {len(g)} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.2f} | {X0[s]*st:.1f} → {X1[s]*st:.1f} % | "
                 f"{n0:+,.0f} → {n1:+,.0f} | {d0:+.2f} → {d1:+.2f} |")
    r0, r1 = daily(G, X0, days, K.SLEEVES), daily(G, X1, days, K.SLEEVES)
    L.append(f"| **TOTAL** | {len(G)} | {(G.pct > 0).mean():.0%} | {G.pct.mean():+.2f} | — | **{(G.pct * G.sleeve.map(X0) / 100 * ACC).sum():+,.0f} → "
             f"{(G.pct * G.sleeve.map(X1) / 100 * ACC).sum():+,.0f}** | **{cmp(r0):+.2f} → {cmp(r1):+.2f}** |")
    eq0, eq1 = np.cumprod(1 + r0.values), np.cumprod(1 + r1.values)
    L.append(f"Worst day {r0.min()*100:+.1f} % → {r1.min()*100:+.1f} % · max drawdown {(eq0/np.maximum.accumulate(eq0)-1).min()*100:+.1f} % → "
             f"{(eq1/np.maximum.accumulate(eq1)-1).min()*100:+.1f} % · days below −5 %: {(r0 < -0.05).sum()} → {(r1 < -0.05).sum()}")
    L.append("One step at a time (total %/day, the others at today's size):")
    for s in STEP:
        Xs = dict(X0); Xs[s] = X0[s] * STEP[s]
        rs = daily(G, Xs, days, K.SLEEVES)
        L.append(f"  {NAMES[s]}: {cmp(r0):+.2f} → {cmp(rs):+.2f} %/day · worst day {rs.min()*100:+.1f} %")
    return L

L = ["# LEVERAGE STEP — Momentum Long 15× · WILLY 8× · RETRY 10× · Flatbase 8× (2026-11-11)", __doc__]
L += table("B25", ["B25"]) + table("MASTER (all batches)", [e for e, _, _ in per])
open(os.path.join(K.ROOT, "reports", "SIZING_LEV_STEP_HT_2026-11-11.txt"), "w").write("\n".join(L))
print("\n".join(L))
