#!/usr/bin/env python3
"""YR6 WL — FRENZY_LITE candidate gate: skip LITE when BTC's 1h EMA20 slope at entry ≤ 0 (entry_btc_1h_slope, the engine's
_current_btc_1h_slope = (EMA20_1h − EMA20_1h 3 bars ago) / … × 100). Screen: day-unit macro survivor in 2025 (p_adj 0.004), confirmed in
2026-H2 (P 0.975) — but the bear half 2026-H1 goes the other way. This script prices the gate two independent ways and tests ONE
pre-registered mechanism for the bear reversal (written before looking): "falling BTC hurts alt pumps only while BTC is in an uptrend
(BTC close > its 50-day SMA, ex-ante prior-day panel) — risk-off rotation; in a BTC downtrend alt frenzies are idiosyncratic".

  venv/bin/python -I scripts/study_yr6wl_frenzy_litegate.py   → reports/YR6_WL_FRENZY_litegate.md
Needs reports/YR6_WL_FRENZY_regime_daily_panel.csv (study_yr6wl_frenzy_regime.py).

Way 1: seed 1 fills. Way 2: seeds 2 + 3 pooled (independent scan cadence) — the headline must agree.
Per regime and pooled: blocked N · days · WR · avg · day-bootstrap CI and P(avg < 0) · kept avg · sleeve breakeven WR · concentration (top-2
pairs, top day share of the blocked loss) · top-50 subset · $ at today's size (usd_new, fixed $2,900 book) · Δ$ = −Σ blocked usd ·
haircut 30–50 %. Expectancy bar (CLAUDE.md): WR_blk < breakeven ∧ P(avg_blk < 0) ≥ 0.95 ∧ ≥ 8 days ∧ no pair / day ≥ 50 % ∧ N ≥ 15.
Not modelled: a skipped LITE stretch frees the pair / slot for a later LONG / WIDE / WILLY / LITE entry (second-order fills).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402

P = pd.read_csv(os.path.join(C.REP, "YR6_WL_FRENZY_regime_daily_panel.csv"), index_col=0)
A = C.load(all_seeds=True)
A = A[A.sleeve == "FRENZY_LITE"].copy()
A["sl"] = pd.to_numeric(A.entry_btc_1h_slope, errors="coerce")
A["up"] = A.day.map(P.btc_vs_sma50) > 0
out = []


def read(S, blk, label):
    y = S.y.values; b = blk & S.sl.notna().values; k = ~b
    if b.sum() < 3:
        return dict(rule=label, N_blk=int(b.sum()))
    yb = y[b]; lo, hi, pp = C.day_boot(yb, S.day.values[b])
    loss = -np.minimum(yb, 0)
    pc = pd.Series(loss, index=S.pair.values[b]).groupby(level=0).sum().sort_values(ascending=False)
    dc = pd.Series(loss, index=S.day.values[b]).groupby(level=0).sum().sort_values(ascending=False)
    be = C.breakeven_wr(y[k])
    r = dict(rule=label, N_blk=int(b.sum()), days=len(set(S.day.values[b])), WR_blk=100 * (yb > 0).mean(), avg_blk=yb.mean(), ci=f"[{lo:+.2f}, {hi:+.2f}]",
             P_lt0=1 - pp, N_kept=int(k.sum()), avg_kept=y[k].mean(), avg_all=y.mean(), be_wr=100 * be,
             top2pair=pc.head(2).sum() / max(loss.sum(), 1e-9), top1day=dc.head(1).sum() / max(loss.sum(), 1e-9),
             d_usd=-S.usd_new.values[b].sum())
    r["BAR"] = bool(r["WR_blk"] < r["be_wr"] and r["P_lt0"] >= 0.95 and r["days"] >= 8 and r["top2pair"] < 0.5 and r["top1day"] < 0.5 and r["N_blk"] >= 15)
    return r


RULES = {"G1  slope ≤ 0": lambda S: (S.sl <= 0).values,
         "G2  slope ≤ 0 ∧ BTC > SMA50 (pre-registered)": lambda S: ((S.sl <= 0) & S.up).values,
         "G2' slope ≤ 0 ∧ BTC ≤ SMA50 (complement)": lambda S: ((S.sl <= 0) & ~S.up).values}
for way, seeds in (("Way 1 — seed 1", [1]), ("Way 2 — seeds 2+3 pooled", [2, 3])):
    out.append(f"\n## {way}\n")
    rows = []
    for rule, f in RULES.items():
        for reg in C.REGIMES + ["ALL"]:
            for sub in ("all", "top50"):
                S = A[A.seed.isin(seeds) & ((A.regime == reg) if reg != "ALL" else True)]
                if sub == "top50":
                    S = S[S.top50]
                rows.append(dict(regime=reg, sub=sub, **read(S, f(S), rule)))
    T = pd.DataFrame(rows)
    out.append(C.md_table(T, 3))
    if way.startswith("Way 1"):
        T1 = T

# month view of G2
S = A[A.seed == 1]
out.append("\n## G2 by month (seed 1): blocked avg (N) vs kept avg\n")
rows = []
for m, g in S.groupby("month"):
    b = ((g.sl <= 0) & g.up).values
    rows.append(dict(month=m, BTC_up_days_share=100 * g.groupby("day").up.first().mean(), blk=f"{g.y[b].mean():+.2f} ({int(b.sum())})",
                     kept=f"{g.y[~b].mean():+.2f} ({int((~b).sum())})", G1_blk=f"{g.y[(g.sl <= 0).values].mean():+.2f}"))
out.append(C.md_table(pd.DataFrame(rows), 2))
open(os.path.join(C.REP, "YR6_WL_FRENZY_litegate.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
