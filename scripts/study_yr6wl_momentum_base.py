"""YR6 WL MOMENTUM — step 1: per-sleeve x regime table (two independent computations), breakeven WR,
coverage table, uniformity (cohort) read, per-pair / per-day concentration, top-50 fidelity split.

    venv/bin/python scripts/study_yr6wl_momentum_base.py  > reports/YR6_WL_MOMENTUM_base.txt
"""
import csv
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from study_yr6wl_momentum_common import (FILLS, REGIMES, SLEEVES, breakeven_wr, day_boot, fmt, load_replay,
                                         signal_columns, is_macro)

d = load_replay()
print(f"# YR6 WL MOMENTUM base · {len(d)} fills · seeds {sorted(d.seed.unique())}")

# ---------- (A) sleeve x regime, way 1 (pandas + day bootstrap) ----------
print("\n## A. Sleeve x regime (pooled seeds, day-clustered 95% CI, outcome pct_new)")
print(f"{'sleeve':<10}{'regime':<14}{'N':>6}{'N/seed':>8}{'days':>6}{'WR':>7}{'avg%':>8}{'CI95':>18}{'P>0':>6}{'BE-WR':>7}{'avgW':>7}{'avgL':>7}")
way1 = {}
for sv in SLEEVES:
    for rg in REGIMES + ["ALL"]:
        g = d[(d.sleeve == sv) & ((d.regime == rg) if rg != "ALL" else True)]
        if not len(g):
            continue
        m, lo, hi, p = day_boot(g.y, g.day)
        be = breakeven_wr(g.y)
        way1[(sv, rg)] = (len(g), g.win.mean(), m)
        print(f"{sv:<10}{rg:<14}{len(g):>6}{len(g)/g.seed.nunique():>8.0f}{g.day.nunique():>6}{g.win.mean()*100:>6.1f}%"
              f"{m:>+8.3f}  [{lo:+.3f},{hi:+.3f}]{p:>6.2f}{be*100:>6.1f}%{g.y[g.y>0].mean():>+7.2f}{g.y[g.y<=0].mean():>+7.2f}")

# ---------- (A') way 2: raw csv module, no pandas, per seed then pooled ----------
acc = defaultdict(lambda: [0, 0, 0.0])
with open(FILLS, newline="") as f:
    for r in csv.DictReader(f):
        if r["sleeve"] not in SLEEVES or r["status"] != "CLOSED" or r.get("is_probe", "False") == "True":
            continue
        y = float(r["pct_new"])
        for key in ((r["sleeve"], r["regime"]), (r["sleeve"], "ALL")):
            a = acc[key]; a[0] += 1; a[1] += y > 0; a[2] += y
bad = 0
for k, (n, w, s) in acc.items():
    n1, w1, m1 = way1[k]
    if n != n1 or abs(w / n - w1) > 1e-9 or abs(s / n - m1) > 1e-9:
        bad += 1; print("MISMATCH", k, n, n1, w / n, w1, s / n, m1)
print(f"\nway-2 cross-check (csv module, independent of pandas): {len(acc)} cells, mismatches = {bad}")

# per-seed spread
print("\n## A2. Per seed avg% (stability)")
for sv in SLEEVES:
    for rg in REGIMES:
        g = d[(d.sleeve == sv) & (d.regime == rg)]
        print(f"  {sv:<10}{rg:<14}" + "  ".join(f"s{s}: {len(x):4d}·{x.win.mean()*100:3.0f}%·{x.y.mean():+.3f}" for s, x in g.groupby("seed")))

# ---------- (B) coverage ----------
print("\n## B. Coverage (non-null share) of every entry_* signal column, per sleeve x regime")
cols = signal_columns(d)
cov = d.groupby(["sleeve", "regime"])[cols].apply(lambda g: g.notna().mean()).T
cov.columns = [f"{a[:5]}|{b[:7]}" for a, b in cov.columns]
full = cov[(cov >= 0.999).all(axis=1)].index.tolist()
part = cov[~(cov >= 0.999).all(axis=1)]
print(f"  {len(full)} columns at 100 % in every cell (macro: {sum(map(is_macro, full))}, pair: {len(full) - sum(map(is_macro, full))})")
print("  columns below 100 % somewhere:")
print(part.round(2).to_string())
allc = [c for c in d.columns if c.startswith("entry_")]
dead = [c for c in allc if d[c].notna().mean() == 0]
print(f"  never stamped in the replay (0 %, cannot test): {len(dead)} → " + ", ".join(dead))

# ---------- (C) uniformity: cohort table ----------
print("\n## C. Uniformity read — avg% (N) by cohort, MOM-long / MOM-short, per regime (pooled seeds)")
d["rank_b"] = pd.cut(d.entry_pair_rank, [0, 20, 50, 100, 1e9], labels=["r≤20", "r21-50", "r51-100", "r>100"])
d["hour_b"] = pd.cut(d.hour, [-1, 7, 15, 23], labels=["00-07", "08-15", "16-23"])
d["cellsrc"] = d.cell_multiplier_source.fillna("none").str.split("+").str[0]
d["mon"] = d.ts.dt.strftime("%Y-%m")
d["c1"] = np.where(d.entry_pattern_c_any_match.astype(float) > 0, "C-any", "no-C")
d["wp"] = np.where(d.entry_pattern_w_any_match.astype(float) > 0, "W-any", "no-W")
d["t50"] = np.where(d.top50, "top50", "non-top50")
d["qs"] = d.entry_quality_score.astype("Int64").astype(str)
d["conf"] = d.confidence
d["regime_btc"] = d.entry_btc_regime
for sv in SLEEVES:
    print(f"\n### {sv}")
    for coh in ["t50", "rank_b", "hour_b", "cellsrc", "c1", "wp", "conf", "qs", "regime_btc", "entry_order_type"]:
        t = d[d.sleeve == sv].groupby([coh, "regime"], observed=True).y.agg(["mean", "size"]).unstack("regime")
        line = []
        for idx, r in t.iterrows():
            line.append(f"{str(idx):<14}" + " | ".join(
                f"{rg[:7]} {r[('mean', rg)]:+.3f} ({int(r[('size', rg)]) if np.isfinite(r[('size', rg)]) else 0:4d})" if ("mean", rg) in r.index and np.isfinite(r[("mean", rg)]) else f"{rg[:7]}    —        " for rg in REGIMES))
        print(f"  [{coh}]"); [print("    " + x) for x in line]
    # share of cohorts negative per regime
    print("  negative-cohort share per regime (cohorts with N>=60):")
    for rg in REGIMES:
        neg = tot = 0
        for coh in ["t50", "rank_b", "hour_b", "cellsrc", "c1", "wp", "conf", "qs", "regime_btc", "entry_order_type"]:
            for _, g in d[(d.sleeve == sv) & (d.regime == rg)].groupby(coh, observed=True):
                if len(g) >= 60:
                    tot += 1; neg += g.y.mean() < 0
        print(f"    {rg:<14} {neg}/{tot} cohorts negative")
    print("  month avg% (N/seed):  " + " · ".join(f"{k} {g.y.mean():+.2f}({len(g)/3:.0f})" for k, g in d[d.sleeve == sv].groupby("mon")))

# ---------- (D) concentration ----------
print("\n## D. Concentration — share of the regime's NET LOSS (sum of y over losing pairs/days) from top-2 pairs / top day")
for sv in SLEEVES:
    for rg in REGIMES:
        g = d[(d.sleeve == sv) & (d.regime == rg)]
        tot = g.y.sum()
        pp = g.groupby("pair").y.sum().sort_values(); dd = g.groupby("day").y.sum().sort_values()
        npairs = g.pair.nunique()
        print(f"  {sv:<10}{rg:<14} Σy {tot:+8.1f} pts · pairs {npairs} · top-2 losing pairs {pp.index[0]} {pp.iloc[0]:+.1f}, {pp.index[1]} {pp.iloc[1]:+.1f} "
              f"= {pp.iloc[:2].sum() / pp[pp < 0].sum() * 100:.0f}% of gross pair-loss · worst day {dd.index[0].date()} {dd.iloc[0]:+.1f} "
              f"= {dd.iloc[0] / dd[dd < 0].sum() * 100:.0f}% of gross day-loss · losing pairs {(pp < 0).mean() * 100:.0f}% · losing days {(dd < 0).mean() * 100:.0f}%")
        # sleeve ex top-5 pairs
        ex = g[~g.pair.isin(pp.index[:5])]
        print(f"      ex worst-5 pairs: N {len(ex)} avg {ex.y.mean():+.3f}  |  ex best-5 pairs: avg {g[~g.pair.isin(pp.index[-5:])].y.mean():+.3f}")

# ---------- (E) exit decomposition ----------
print("\n## E. Close-reason mix (share · avg%) per regime — MOM-long")
for sv in SLEEVES:
    print(f"### {sv}")
    for rg in REGIMES:
        g = d[(d.sleeve == sv) & (d.regime == rg)]
        cr = g.close_reason.str.replace(r" L\d", "", regex=True)
        t = g.groupby(cr).y.agg(["size", "mean"]).sort_values("size", ascending=False)
        print(f"  {rg:<14}" + " · ".join(f"{k} {v['size']/len(g)*100:.0f}% {v['mean']:+.2f}" for k, v in t.head(7).iterrows()))
        print(f"  {'':<14}peak≥+0.20 share {(g.peak_pnl.astype(float) >= 0.2).mean()*100:.0f}% · losers that peaked ≥+0.20: {((g.y<=0)&(g.peak_pnl.astype(float)>=0.2)).sum()/(g.y<=0).sum()*100:.0f}% · avg peak {g.peak_pnl.astype(float).mean():+.2f} · avg trough {g.trough_pnl.astype(float).mean():+.2f}")
