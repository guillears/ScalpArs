#!/usr/bin/env python3
"""B25 only (operator 11-12): the "keep the planned position, add margin at a lower leverage" option for bracket-capped fills.
Three sizings per fill, today's one size ($10,000 account, K.SLEEVES X), today's bracket table (reports/BRACKETS_2026-11-12.txt, × 0.95 headroom):
  TODAY   what the engine does now: same margin, the bracket lowers the leverage → smaller position (study_bracket_lev_step_1112.notional)
  OPTION  the planned position (X × new-step leverage ratio) at the HIGHEST leverage ≤ the planned one whose tier allows it; margin = position ÷ that
          leverage; if no tier allows it → the largest position any tier allows at that margin need ≤ … (the biggest the brackets allow)
  FLAT    the planned position everywhere (no brackets — the assumption the DL 533 projection made)
FREE BALANCE replayed on open / close times: a fill needs its margin ≤ $10,000 − Σ margin of the fills open at that moment (one-size account,
no compounding); if not → it opens at what is free (position = free × its leverage), 0 = skipped. Applied to TODAY and OPTION alike.
Second chance (DL 534 rule) on momentum longs; ML risk cap not replayed (it touched 1 B25 fill)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_bracket_lev_step_1112 as B   # loads k (master, re-based), T, notional, NAMES, OLD, NEW, ACC

K, k, ACC = B.K, B.k, B.ACC
X0 = {s: v[3] for s, v in K.SLEEVES.items()}


def option(pair, want, lev_max):
    tiers = B.T.get(pair)
    if not tiers:
        return want, want / lev_max, lev_max
    ok = [(l, c) for c, l in tiers if c >= want - 1e-6 and l >= 1]
    if ok:
        lev = float(int(min(max(l for l, _ in ok), lev_max)))
        return want, want / lev, lev
    c, l = max(tiers)                       # nothing allows the full position → the biggest tier
    lev = float(int(min(l, lev_max)))
    return c, c / lev, lev


G = k[k.era == "B25"].copy().sort_values("o")
rows = []
for i, r in G.iterrows():
    s = r.sleeve
    mg0 = X0[s] * ACC / B.OLD[s]
    want = mg0 * B.NEW[s]
    n_t, l_t = B.notional(r.pair, mg0, B.NEW[s])
    n_o, m_o, l_o = option(r.pair, want, B.NEW[s])
    rows.append((i, want, n_t, n_t / l_t, l_t, n_o, m_o, l_o))
R = pd.DataFrame(rows, columns=["i", "want", "n_today", "m_today", "l_today", "n_opt", "m_opt", "l_opt"]).set_index("i")
G = G.join(R)


def free_balance(G, ncol, mcol, lcol):
    """replay margin use → (position after the free-balance check, margin used, n squeezed, n skipped, max margin in use %)."""
    open_, out, sq, sk, peak = [], {}, 0, 0, 0.0
    for i, r in G.sort_values("o").iterrows():
        open_ = [(c, m) for c, m in open_ if c > r.o]
        free = ACC - sum(m for _, m in open_)
        m, n = r[mcol], r[ncol]
        if m > free + 1e-6:
            if free <= 0:
                out[i] = (0.0, 0.0); sk += 1; continue
            n, m = free * r[lcol], free; sq += 1
        out[i] = (n, m)
        open_.append((r.c, m))
        peak = max(peak, sum(mm for _, mm in open_) / ACC * 100)
    o = pd.DataFrame(out, index=["n", "m"]).T
    return o.n, o.m, sq, sk, peak


days = pd.DatetimeIndex(np.concatenate([pd.date_range(a.normalize(), b.normalize(), freq="D").values for e, a, b in B.per if e == "B25"]))


def dcr(g, n):
    r = (g.pct_sc * n.reindex(g.index).fillna(0) / ACC / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0)
    eq = np.cumprod(1 + r.values)
    return B.cmp(r), r.min() * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100


nT, mT, sqT, skT, pkT = free_balance(G, "n_today", "m_today", "l_today")
nO, mO, sqO, skO, pkO = free_balance(G, "n_opt", "m_opt", "l_opt")
G["nT"], G["nO"], G["mO"] = nT, nO, mO
L = ["# B25 — bracket-capped fills: keep the planned position with more margin (2026-11-12)", __doc__,
     f"\n## B25 — {len(days)} days · ${ACC:,} account", "",
     "| Sleeve | Fills | capped today | %/day TODAY (brackets) | %/day OPTION (more margin) | %/day FLAT (no limits) | avg margin per fill today → option |",
     "|---|---|---|---|---|---|---|"]
for s in K.SLEEVES:
    g = G[G.sleeve == s]
    if not len(g):
        continue
    cap = int((g.n_today < g.want - 1).sum())
    L.append(f"| {B.NAMES[s]} | {len(g)} | {cap} | {dcr(g, g.nT)[0]:+.2f} | **{dcr(g, g.nO)[0]:+.2f}** | {dcr(g, g.want)[0]:+.2f} | "
             f"${g.m_today.mean():,.0f} → ${g.mO.mean():,.0f} |")
t, o, f = dcr(G, G.nT), dcr(G, G.nO), dcr(G, G.want)
L.append(f"| **TOTAL** | {len(G)} | {int((G.n_today < G.want - 1).sum())} | {t[0]:+.2f} | **{o[0]:+.2f}** | {f[0]:+.2f} | |")
L += ["", f"Worst day: today {t[1]:+.1f} % · option {o[1]:+.1f} % · flat {f[1]:+.1f} %  |  max drawdown: today {t[2]:+.1f} % · option {o[2]:+.1f} % · flat {f[2]:+.1f} %",
      f"Free balance: most margin in use at once — today {pkT:.0f} % · option {pkO:.0f} % of the account; fills squeezed by the free balance today {sqT} / option {sqO}, skipped {skT} / {skO}."]
c = G[G.n_today < G.want - 1].sort_values("o")
L += ["", "Capped fills, one by one:", "| Opened | Pair | Sleeve | planned | today (margin · lev · position) | option (margin · lev · position) | booked % | $ today → option |",
      "|---|---|---|---|---|---|---|---|"]
for r in c.itertuples():
    L.append(f"| {r.o:%m-%d %H:%M} | {r.pair} | {B.NAMES[r.sleeve]} | ${r.want:,.0f} | ${r.m_today:,.0f} · {r.l_today:g}× · ${r.nT:,.0f} | "
             f"${r.mO:,.0f} · {r.l_opt:g}× · ${r.nO:,.0f} | {r.pct_sc:+.2f} | {r.pct_sc * r.nT / 100:+,.0f} → {r.pct_sc * r.nO / 100:+,.0f} |")
open(os.path.join(K.ROOT, "reports", "BRACKET_MARGIN_OPTION_B25_2026-11-12.txt"), "w").write("\n".join(L))
print("\n".join(L))
