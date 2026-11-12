#!/usr/bin/env python3
"""B25 RE-BASE on everything live in B26 (operator 11-12): today's filters (master stack_keep / stack_pct, STACK 2026-11-11a) · C5 sizes (DL 528)
· ML SECOND CHANCE (DL 532; DL 534 rule — scripts/ml_sc_history.py: booked + the frozen replica delta where the replica reproduced the exit) · leverage step (DL 533) · ML risk cap 10 %
(replayed on open/close times; an open ML counts at its worst stop unless it closed — CONSERVATIVE: arming / BE lock not on the path, so the cap
shrinks at least as much as live would). $10,000 account, one size per sleeve (DL 503), daily compound over every calendar day of the batch.
Columns: ① C5 today before DL 532/533 · ② + second chance · ③ + leverage step + ML cap (= what B26 runs)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_sizing_kelly_1111 as K
ACC, CAP = 10_000, 10.0
STEP = {"ML": 1.5, "WILLY": 1.6, "HT": 1.6, "RETRY": 10 / 7, "FLAT": 1.6}
NAMES = {"ML": "Momentum Long", "LITE": "LITE", "RETRY": "LITE RETRY", "WILLY": "WILLY", "HT": "WILLY HT", "FLAT": "Flatbase"}
k, per = K.master()
X0 = {s: v[3] for s, v in K.SLEEVES.items()}
X1 = {s: X0[s] * STEP.get(s, 1.0) for s in X0}
import ml_sc_history as SCH   # DL 534 rule: booked + frozen delta only where the replica reproduced the booked exit (else booked, no credit)
k["pct_sc"] = k.pct
_ml = k.sleeve == "ML"
_p, _n_ch, _n_un, _n_dis = SCH.rebase(k.pct[_ml], k.opened_at[_ml], k.pair[_ml], k.entry_adx_delta[_ml], k.close_reason[_ml], 4_102_444_800_000)
k.loc[_ml, "pct_sc"] = _p
miss = f"{_n_un} unpriced (kept at today's exit) · {_n_ch} credited · {_n_dis} replica-disagrees (no credit)"
k["pct_sc"] = k.pct_sc.fillna(k.pct)
k["atr"] = pd.to_numeric(k.entry_atr_pct, errors="coerce")


def ml_scale(G):
    """ML risk cap replay → size factor per ML fill (1 = full, <1 shrunk, 0 = skipped below the minimum)."""
    m = G[G.sleeve == "ML"].sort_values("o")
    f, open_ = {}, []
    for i, r in m.iterrows():
        open_ = [(c, n, w) for c, n, w in open_ if c > r.o]
        w = min(1.2, max(0.7, 1.5 * r.atr)) if r.atr == r.atr else 1.2
        risk = sum(n * X1["ML"] * ACC * ww / 100 for _, n, ww in open_) / ACC * 100
        room = (CAP - risk) / 100 * ACC
        full = X1["ML"] * ACC
        s = 1.0 if room >= full * w / 100 else max(0.0, room / (full * w / 100))
        s = 0.0 if s * full < 0.1 * full else s          # tiny remainder ≈ below the minimum margin → skip
        f[i] = s
        if s > 0:
            open_.append((r.c, s, w))
    return pd.Series(f)


def cmp(r):
    return ((1 + r).prod() ** (1 / len(r)) - 1) * 100


def scope(eras, title):
    days = pd.DatetimeIndex(np.concatenate([pd.date_range(a.normalize(), b.normalize(), freq="D").values for e, a, b in per if e in eras]))
    G = k[k.era.isin(eras)].copy()
    G["f"] = 1.0
    sh = ml_scale(G)
    G.loc[sh.index, "f"] = sh
    cols = [("C5 (before 532/533)", "pct", X0, False), ("+ second chance", "pct_sc", X0, False), ("+ leverage step + ML cap = B26", "pct_sc", X1, True)]
    out = {}
    for name, col, X, cap in cols:
        x = G.sleeve.map(X) * (G.f if cap else 1.0)
        G["d"] = G[col] * x / 100
        rows = {}
        for s in list(K.SLEEVES) + ["ALL"]:
            g = G if s == "ALL" else G[G.sleeve == s]
            if not len(g):
                continue
            kept = g[(g.f > 0) | (not cap)] if cap else g
            r = g.d.groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0)
            eq = np.cumprod(1 + r.values)
            rows[s] = dict(n=len(kept), wr=(kept[col] > 0).mean(), avg=kept[col].mean(), usd=g.d.sum() * ACC, dcr=cmp(r),
                           worst=r.min() * 100, dd=(eq / np.maximum.accumulate(eq) - 1).min() * 100)
        out[name] = rows
    L = [f"\n## {title} — {len(days)} calendar days · ${ACC:,} account (one size)",
         "| Sleeve | N | WR | Avg % | Net $ | %/day | → second chance: WR · Avg % · Net $ · %/day | → B26 (lev step + ML cap): N · Net $ · %/day |",
         "|---|---|---|---|---|---|---|---|"]
    a, b, c = (out[n] for n, *_ in cols)
    for s in list(K.SLEEVES) + ["ALL"]:
        if s not in a:
            continue
        nm = "**TOTAL**" if s == "ALL" else NAMES[s]
        L.append(f"| {nm} | {a[s]['n']} | {a[s]['wr']:.0%} | {a[s]['avg']:+.2f} | {a[s]['usd']:+,.0f} | {a[s]['dcr']:+.2f} | "
                 f"{b[s]['wr']:.0%} · {b[s]['avg']:+.2f} · {b[s]['usd']:+,.0f} · {b[s]['dcr']:+.2f} | {c[s]['n']} · {c[s]['usd']:+,.0f} · **{c[s]['dcr']:+.2f}** |")
    t = [o["ALL"] for o in (a, b, c)]
    L.append(f"Worst day {t[0]['worst']:+.1f} → {t[1]['worst']:+.1f} → {t[2]['worst']:+.1f} % · max drawdown {t[0]['dd']:+.1f} → {t[1]['dd']:+.1f} → {t[2]['dd']:+.1f} %")
    m = G[G.sleeve == "ML"]
    L.append(f"ML risk cap (conservative replay): {int((m.f < 1).sum())} of {len(m)} momentum longs touched — {int(((m.f < 1) & (m.f > 0)).sum())} shrunk "
             f"(avg to {m.f[(m.f < 1) & (m.f > 0)].mean():.0%}), {int((m.f == 0).sum())} skipped; their booked % "
             + (", ".join(f"{r.pair} {r.pct_sc:+.2f}→×{r.f:.2f}" for r in m[m.f < 1].itertuples()) or "—"))
    return L


L = ["# B25 RE-BASE on the B26 stack (2026-11-12)", __doc__, f"(ML second chance: {miss})"]
L += scope(["B25"], "B25") + scope([e for e, _, _ in per], "MASTER (all batches)")
open(os.path.join(K.ROOT, "reports", "B25_REBASE_B26_STACK_2026-11-12.txt"), "w").write("\n".join(L))
print("\n".join(L))
