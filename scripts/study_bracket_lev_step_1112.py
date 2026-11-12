#!/usr/bin/env python3
"""How much of the DL 533 leverage step survives Binance's per-pair position limits (operator 11-12, after H / ONE opened at 5× not 8×).
Tiers = the bot's own cached table (dashboard /api/manual/limits, 2026-11-12, already × 0.95 headroom — the order path's numbers) →
reports/BRACKETS_2026-11-12.txt. TODAY'S table applied to every fill (Binance changes limits now and then — caveat).
Per fill at today's one size ($10,000 account, K.SLEEVES): margin = X × account / old leverage; old = what the bracket allowed at the old
leverage, new = at the new leverage (services.trading_engine.leverage_bracket_limit, same margin, whole leverages). Gain kept = new / old notional.
Then the B25 / master daily compound with each fill's REAL bracket-limited notional instead of the flat step."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import study_sizing_kelly_1111 as K
import ml_sc_history as SCH
from services.trading_engine import leverage_bracket_limit

ACC = 10_000
OLD = {"ML": 10, "LITE": 8, "RETRY": 7, "WILLY": 5, "HT": 5, "FLAT": 5}
NEW = {"ML": 15, "LITE": 8, "RETRY": 10, "WILLY": 8, "HT": 8, "FLAT": 8}
NAMES = {"ML": "Momentum Long", "LITE": "LITE", "RETRY": "LITE RETRY", "WILLY": "WILLY", "HT": "WILLY HT", "FLAT": "Flatbase"}
T = {}
for part in open(os.path.join(K.ROOT, "reports", "BRACKETS_2026-11-12.txt")).read().strip().split("|"):
    p, s = part.split("=")
    T[p] = [(float(a), float(b)) for a, b in (x.split("@") for x in s.split(";"))]


def notional(pair, margin, lev):
    lim = leverage_bracket_limit(T.get(pair), lev, margin)
    if lim is None:
        return margin * lev, lev
    le = float(int(lim[0])) if lim[0] >= 1 else lim[0]
    return min(margin, lim[1] / le) * le, le


k, per = K.master()
k["pct_sc"] = k.pct
m = k.sleeve == "ML"
k.loc[m, "pct_sc"] = SCH.rebase(k.pct[m], k.opened_at[m], k.pair[m], k.entry_adx_delta[m], k.close_reason[m], 4_102_444_800_000)[0]
k["pct_sc"] = k.pct_sc.fillna(k.pct)
X0 = {s: v[3] for s, v in K.SLEEVES.items()}
rows = []
for i, r in k.iterrows():
    s = r.sleeve
    mg = X0[s] * ACC / OLD[s]
    n0, l0 = notional(r.pair, mg, OLD[s])
    n1, l1 = notional(r.pair, mg, NEW[s])
    rows.append((i, n0, l0, n1, l1, mg * OLD[s], mg * NEW[s]))
R = pd.DataFrame(rows, columns=["i", "n_old", "lev_old", "n_new", "lev_new", "want_old", "want_new"]).set_index("i")
k = k.join(R)


def cmp(r):
    return ((1 + r).prod() ** (1 / len(r)) - 1) * 100


def scope(eras, title):
    days = pd.DatetimeIndex(np.concatenate([pd.date_range(a.normalize(), b.normalize(), freq="D").values for e, a, b in per if e in eras]))
    G = k[k.era.isin(eras)].copy()
    L = [f"\n## {title} — {len(days)} days · ${ACC:,} account · today's sizes, today's bracket table",
         "| Sleeve | Fills | Step | Fills that get the FULL step | capped (lower lev / smaller) | avg notional new ÷ old | %/day: flat step (assumed) → with the limits |",
         "|---|---|---|---|---|---|---|"]
    tot = {}
    for lab, nk in (("flat", "flat"), ("real", "real"), ("old", "old")):
        r = pd.Series(0.0, index=days)
        for s in K.SLEEVES:
            g = G[G.sleeve == s]
            x = (g.want_new if lab == "flat" else g.n_new if lab == "real" else g.n_old) / ACC
            r = r.add((g.pct_sc * x / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0), fill_value=0.0)
        tot[lab] = r
    for s in K.SLEEVES:
        g = G[G.sleeve == s]
        if not len(g):
            continue
        full = (g.n_new >= g.want_new - 1).sum()
        dd = {}
        for lab, col in (("flat", "want_new"), ("real", "n_new")):
            dd[lab] = cmp((g.pct_sc * g[col] / ACC / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0))
        L.append(f"| {NAMES[s]} | {len(g)} | {OLD[s]}→{NEW[s]}× | {full} ({full / len(g):.0%}) | {len(g) - full} | {(g.n_new / g.n_old).mean():.2f}× "
                 f"(flat {NEW[s] / OLD[s]:.2f}×) | {dd['flat']:+.2f} → **{dd['real']:+.2f}** |")
    L.append(f"| **TOTAL** | {len(G)} | | | | | {cmp(tot['flat']):+.2f} → **{cmp(tot['real']):+.2f}** (before the step, with the limits: {cmp(tot['old']):+.2f}) |")
    w = G[G.sleeve.isin(["WILLY", "HT"])].sort_values("o")
    if len(w):
        L += ["", "WILLY / HT fills, one by one (want = today's size at 8×):", "| Opened | Pair | Sleeve | want at 8× | brackets allow | at | booked % |", "|---|---|---|---|---|---|---|"]
        for r in w.itertuples():
            L.append(f"| {r.o:%m-%d %H:%M} | {r.pair} | {NAMES[r.sleeve]} | ${r.want_new:,.0f} | ${r.n_new:,.0f} | {r.lev_new:g}× | {r.pct_sc:+.2f} |")
    return L


L = ["# LEVERAGE STEP vs BINANCE POSITION LIMITS (2026-11-12)", __doc__]
L += scope(["B25"], "B25") + scope([e for e, _, _ in per], "MASTER (all batches)")
open(os.path.join(K.ROOT, "reports", "BRACKET_LEV_STEP_2026-11-12.txt"), "w").write("\n".join(L))
print("\n".join(L))
