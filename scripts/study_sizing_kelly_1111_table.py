#!/usr/bin/env python3
"""Per-sleeve BEFORE → AFTER table for the sizing study (operator 11-11: "by sleeve WR, net $, avg %, daily compound, per sleeve and total",
master and B25). Account $10,000 (C5 was engine-verified at $10k). BEFORE = today's C5 X. AFTER-a = cautious ¼-Kelly fitted on the WHOLE
master (in-sample, includes B25 — the optimistic version) · AFTER-b = walk-forward: sizes from fills closed BEFORE B25 only (what the rule
would really have set going into B25). WR and avg % do not depend on size (same trades) — only $ and the daily compound move."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_sizing_kelly_1111 as K
ACC = 10_000
k, per = K.master()
b25_t0 = dict((e, t0) for e, t0, t1 in per)["B25"]
X_now = {s: v[3] for s, v in K.SLEEVES.items()}
X_is = {s: K.kelly_x(k[k.sleeve == s].pct.values, K.SLEEVES[s][2], 0.25, 8.0, 15) for s in K.SLEEVES}
X_wf = {s: K.kelly_x(k[(k.sleeve == s) & (k.c < b25_t0)].pct.values, K.SLEEVES[s][2], 0.25, 8.0, 15) for s in K.SLEEVES}

def days_of(eras):
    return pd.DatetimeIndex(np.concatenate([pd.date_range(t0.normalize(), t1.normalize(), freq="D").values for e, t0, t1 in per if e in eras]))

def dcr(g, X, days):
    r = (g.pct * X / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0)
    return ((1 + r).prod() ** (1 / len(days)) - 1) * 100

def table(scope, eras):
    days = days_of(eras)
    G = k[k.era.isin(eras)]
    L = [f"\n## {scope} — {len(days)} calendar days · account ${ACC:,}",
         "| Sleeve | N | WR | Avg % | Loss/stop now → in-sample → walk-fwd | Net $ now | in-sample | walk-fwd | %/day now | in-sample | walk-fwd |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {m: pd.Series(0.0, index=days) for m in ("now", "is", "wf")}
    nets = {m: 0.0 for m in tot}
    for s in K.SLEEVES:
        g = G[G.sleeve == s]
        if not len(g):
            continue
        st = K.SLEEVES[s][2]
        row = []
        for m, X in (("now", X_now), ("is", X_is), ("wf", X_wf)):
            net = (g.pct * X[s] / 100 * ACC).sum(); nets[m] += net
            tot[m] = tot[m].add((g.pct * X[s] / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0), fill_value=0.0)
            row.append((net, dcr(g, X[s], days)))
        L.append(f"| {s} | {len(g)} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.2f} | {X_now[s]*st:.1f} → {X_is[s]*st:.1f} → {X_wf[s]*st:.1f} % | "
                 f"{row[0][0]:+,.0f} | {row[1][0]:+,.0f} | {row[2][0]:+,.0f} | {row[0][1]:+.2f} | {row[1][1]:+.2f} | {row[2][1]:+.2f} |")
    g = G
    c = {m: ((1 + tot[m]).prod() ** (1 / len(days)) - 1) * 100 for m in tot}
    w = {m: tot[m].min() * 100 for m in tot}
    L.append(f"| **TOTAL** | {len(g)} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.2f} | — | **{nets['now']:+,.0f}** | **{nets['is']:+,.0f}** | **{nets['wf']:+,.0f}** | "
             f"**{c['now']:+.2f}** | **{c['is']:+.2f}** | **{c['wf']:+.2f}** |")
    L.append(f"Worst day (% of account): now {w['now']:+.1f} · in-sample {w['is']:+.1f} · walk-forward {w['wf']:+.1f}. "
             "Net $ = simple sum at a flat $10k (no compounding); %/day = compounded over every calendar day.")
    return L

L = ["# SIZING — before → after per sleeve (2026-11-11)", __doc__]
L += table("MASTER (all batches, today's stack)", [e for e, _, _ in per])
L += table("B25", ["B25"])
open(os.path.join(K.ROOT, "reports", "SIZING_KELLY_2026-11-11_table.txt"), "w").write("\n".join(L))
print("\n".join(L))
