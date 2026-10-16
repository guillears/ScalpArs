#!/usr/bin/env python3
"""ML_DOWNTREND_2D (2026-10-16) — deep read of the only walk-forward-confirmed theme (DT ∧ coin age > cut) + checks.
    venv/bin/python -W ignore scripts/study_ml_dt2d_deep.py    → reports/ML_DT2D_deep.txt
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import day_boot, day_boot_delta, breakeven_wr   # noqa: E402

ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NOTIONAL = 700.0          # today's momentum-long 1× notional ≈ $700 (CURRENT_STATE DL 289/298) — $ ruler only
D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_features.pkl"))
D["age"] = pd.to_numeric(D.entry_pair_age_days, errors="coerce")
D["month"] = D.ts.dt.strftime("%Y-%m")
L = []


def P(s=""):
    print(s); L.append(s)


def fmt(y, day):
    y = np.asarray(y, float)
    if not len(y):
        return "0"
    m, lo, hi, pp = day_boot(y, day, n=2000)
    return f"{len(y)}·{np.unique(day).size} d·{(y > 0).mean() * 100:.0f} %·{m:+.3f} [{lo:+.2f},{hi:+.2f}]"


def blocks():
    out = [(e, D[D.block == e], "y") for e in ERAS]
    m = D[D.block == "master"]
    out += [("master as-traded", m, "y"), ("master stack_keep", m[m.stack_keep.astype(bool)], "y_stack"),
            ("fresh line cohort", D[D.block == "fresh"], "y"), ("  of which LIVE (≥ 10-14 18:36)", D[(D.block == "fresh") & D.is_live.fillna(False).astype(bool)], "y")]
    return out


for CUT in (861.5, 546.0):
    P(f"\n# CELL  DT ∧ coin age > {CUT:g} d  (cut = 2025-zone {'median' if CUT > 800 else 'tercile 1'}, label-blind, frozen)")
    P("\n## A · three blocks: zone split and the 1D age effect OUTSIDE the zone (is it a 2D interaction or a plain age effect?)")
    P("| block | DT∧OLD | DT∧YOUNG | Δ in-zone [95 %] | REST∧OLD | REST∧YOUNG | Δ outside-zone | interaction Δin−Δout | age unscored |")
    P("|---|---|---|---|---|---|---|---|---|")
    for lab, d, yc in blocks():
        d = d.assign(yy=d[yc].astype(float))
        sc = d.age.notna() & d.DT.notna()
        z, r = d[sc & (d.DT == 1)], d[sc & (d.DT == 0)]
        zo, ro = (z.age > CUT).values, (r.age > CUT).values
        di = day_boot_delta(z.yy.values, zo, z.day.values, n=2000) if zo.any() and (~zo).any() else (np.nan,) * 4
        do = z.yy[zo].mean() - z.yy[~zo].mean() if False else (r.yy[ro].mean() - r.yy[~ro].mean() if ro.any() and (~ro).any() else np.nan)
        P(f"| {lab} | {fmt(z.yy[zo], z.day[zo])} | {fmt(z.yy[~zo], z.day[~zo])} | {di[0]:+.3f} [{di[1]:+.2f},{di[2]:+.2f}] | "
          f"{fmt(r.yy[ro], r.day[ro])} | {fmt(r.yy[~ro], r.day[~ro])} | {do:+.3f} | {di[0] - do:+.3f} | {int((~sc).sum())} |")
    P("\n## B · per seed (replay) inside-zone Δ (old − young)")
    for e in ERAS:
        d = D[(D.block == e) & (D.DT == 1) & D.age.notna()]
        P(f"- {e}: " + " · ".join(f"seed {s}: {g.y[g.age > CUT].mean() - g.y[g.age <= CUT].mean():+.3f} (n {int((g.age > CUT).sum())}/{int((g.age <= CUT).sum())})"
                                    for s, g in d.groupby("seed")))
    P("\n## C · leave-one-month-out (replay, all 3 eras pooled; inside-zone Δ old − young)")
    d = D[D.block.isin(ERAS) & (D.DT == 1) & D.age.notna()]
    full = d.y[d.age > CUT].mean() - d.y[d.age <= CUT].mean()
    lomo = []
    for mo in sorted(d.month.unique()):
        g = d[d.month != mo]
        lomo.append((mo, g.y[g.age > CUT].mean() - g.y[g.age <= CUT].mean()))
    v = np.array([x for _, x in lomo])
    P(f"- full {full:+.3f} · LOMO range [{v.min():+.3f}, {v.max():+.3f}] · negative in {int((v < 0).sum())}/{len(v)} folds")
    per = [(mo, g.y[g.age > CUT].mean() - g.y[g.age <= CUT].mean(), int((g.age > CUT).sum()), int((g.age <= CUT).sum()))
           for mo, g in d.groupby("month")]
    P("- per-month Δ (old−young): " + " · ".join(f"{mo} {x:+.2f}" for mo, x, a, b in per if a >= 5 and b >= 5))
    P(f"- months with Δ<0: {sum(1 for _, x, a, b in per if a >= 5 and b >= 5 and x < 0)} / {sum(1 for _, x, a, b in per if a >= 5 and b >= 5)}")

    P("\n## D · BLOCK candidate DT∧OLD on the real fills — expectancy bar (WR < BE · day-boot P(mean<0) ≥ 0.95 · ≥ 8 days · no day/pair ≥ 50 % of gross loss · N ≥ 15)")
    for lab, d, yc in blocks()[3:]:
        d = d.assign(yy=d[yc].astype(float))
        be = breakeven_wr(d.yy)
        c = d[(d.DT == 1) & (d.age > CUT)]
        if not len(c):
            P(f"- {lab}: 0"); continue
        m, lo, hi, pp = day_boot(c.yy.values, c.day.values, n=4000)
        loss = c[c.yy < 0]
        gl = -loss.yy.sum()
        tday = (-loss.groupby("day").yy.sum()).max() / gl if gl > 0 else np.nan
        tpair = (-loss.groupby("pair").yy.sum()).max() / gl if gl > 0 else np.nan
        ok = [(c.yy > 0).mean() < be, (1 - pp) >= 0.95, c.day.nunique() >= 8, max(tday, tpair) < 0.5, len(c) >= 15]
        P(f"- {lab}: N {len(c)} · {c.day.nunique()} d · WR {(c.yy > 0).mean() * 100:.0f} % vs BE {be * 100:.1f} % · mean {m:+.3f} "
          f"[{lo:+.2f},{hi:+.2f}] · P(mean<0) {1 - pp:.2f} · top day {tday * 100:.0f} % / top pair {tpair * 100:.0f} % of gross loss · "
          f"bar ①{'✓' if ok[0] else '✗'} ②{'✓' if ok[1] else '✗'} ③{'✓' if ok[2] else '✗'} ④{'✓' if ok[3] else '✗'} N{'✓' if ok[4] else '✗'}")

    P("\n## E · two-sided refusal (block DT∧OLD): before → after, % per fill and Σ, $ recorded and $ at the ~$700 1× notional")
    P("| block | before N·avg·Σ% | refused N (W/L) · Σ% refused | after N·avg·Σ% | Δ avg/fill | Σ$ recorded before→after | Σ$ @ $700 before→after |")
    P("|---|---|---|---|---|---|---|")
    for lab, d, yc in blocks():
        d = d.assign(yy=d[yc].astype(float)).dropna(subset=["yy"])
        ref = ((d.DT == 1) & (d.age > CUT)).values
        a = d[~ref]
        usd = pd.to_numeric(d.get("pnl"), errors="coerce") if lab.startswith(("master", "fresh", "  of")) else pd.Series(np.nan, index=d.index)
        if lab.startswith(("fresh", "  of")):
            usd = pd.to_numeric(d.usd, errors="coerce")
        P(f"| {lab} | {len(d)}·{d.yy.mean():+.3f}·{d.yy.sum():+.1f} | {int(ref.sum())} ({int((d.yy[ref] > 0).sum())}/{int((d.yy[ref] <= 0).sum())}) · "
          f"{d.yy[ref].sum():+.2f} | {len(a)}·{a.yy.mean():+.3f}·{a.yy.sum():+.1f} | {a.yy.mean() - d.yy.mean():+.3f} | "
          f"{usd.sum():+.0f} → {usd[~ref].sum():+.0f} | {d.yy.sum() * NOTIONAL / 100:+.0f} → {a.yy.sum() * NOTIONAL / 100:+.0f} |")
    m = D[D.block == "master"].copy()
    P("\n### master per batch (as-traded): before → after refusing DT∧OLD (age-unscored fills kept)")
    P("| batch | N | before avg·Σ% | refused (W/L) | after avg·Σ% | Σ$ recorded before→after |")
    P("|---|---|---|---|---|---|")
    for e, g in m.groupby("era", sort=False):
        ref = ((g.DT == 1) & (g.age > CUT)).values
        a = g[~ref]
        P(f"| {e} | {len(g)} | {g.y.mean():+.3f}·{g.y.sum():+.2f} | {int(ref.sum())} ({int((g.y[ref] > 0).sum())}/{int((g.y[ref] <= 0).sum())}) | "
          f"{(a.y.mean() if len(a) else np.nan):+.3f}·{a.y.sum():+.2f} | {g.pnl.sum():+.0f} → {a.pnl.sum():+.0f} |")
    if CUT > 800:
        P("\n### per trade — the refused fills (DT∧OLD) on master + fresh")
        P("| block | batch | opened | pair | g1h % | age d | P&L % | $ recorded |")
        P("|---|---|---|---|---|---|---|---|")
        for lab, d in (("master", m), ("fresh", D[D.block == "fresh"])):
            c = d[(d.DT == 1) & (d.age > CUT)].sort_values("ts")
            for r in c.itertuples():
                usd = r.pnl if lab == "master" else r.usd
                bt = r.era if lab == "master" else ("LIVE" if r.is_live else "paper")
                P(f"| {lab} | {bt} | {r.ts:%m-%d %H:%M} | {r.pair} | {r.g1h_z:+.2f} | {r.age:.0f} | {r.y:+.3f} | {float(usd):+.1f} |")
        P("\n### per trade — the fresh line cohort, DT zone, kept vs refused")
        f = D[(D.block == "fresh") & (D.DT == 1)].sort_values("ts")
        P("| opened | pair | live | g1h % | age d | P&L % | $ | DT∧OLD refused |")
        P("|---|---|---|---|---|---|---|---|")
        for r in f.itertuples():
            P(f"| {r.ts:%m-%d %H:%M} | {r.pair} | {'Y' if r.is_live else ''} | {r.g1h_z:+.2f} | {r.age:.0f} | {r.y:+.3f} | {r.usd:+.1f} | {'REFUSE' if r.age > CUT else 'keep'} |")

# z-formula validation (screen) vs a day bootstrap on one cell
S = pd.read_csv(os.path.join(ROOT, "reports/ML_DT2D_screen.csv"))
d = D[(D.block == ERAS[0]) & (D.DT == 1) & D.age.notna()]
o = (d.age > 861.5).values
dl, lo, hi, _ = day_boot_delta(d.y.values, o, d.day.values, n=4000)
row = S[S.cell.str.startswith("entry_pair_age_days >med")].iloc[0]
P(f"\nz validation: screen Δ {row['2025 BU_Din']:+.4f} z {row['2025 BU_Zin']:+.2f} · bootstrap Δ {dl:+.4f} sd≈{(hi - lo) / 3.92:.4f} → z≈{dl / ((hi - lo) / 3.92):+.2f}")
with open(os.path.join(ROOT, "reports/ML_DT2D_deep.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
