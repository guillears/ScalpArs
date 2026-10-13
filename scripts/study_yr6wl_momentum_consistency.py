"""YR6 WL MOMENTUM — step 5: all-regime consistency of every 1D split (thresholds frozen in 2025 BULL), the locked
EXPECTANCY bar on the would-be-blocked zone, and the master real-fill check. Independent re-computation of the
screen's 1D deltas (plain pandas groupby, no matrix code) — the "second way" for the headline separator numbers.

    venv/bin/python scripts/study_yr6wl_momentum_consistency.py ML|MS
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import (REGIMES, breakeven_wr, day_boot, day_boot_delta, is_macro, load_master_mom,
                                         load_replay, signal_columns)

SL = sys.argv[1] if len(sys.argv) > 1 else "ML"
SLV = {"ML": "MOM-long", "MS": "MOM-short"}[SL]
d = load_replay([SLV])
M = load_master_mom(); M = M[M.sleeve == SLV]
Mk = M[M.stack_keep == True].assign(y=lambda x: x.y_stack)
D0 = d[d.regime == "2025 BULL"]
cols = [c for c in signal_columns(d) if pd.to_numeric(D0[c], errors="coerce").notna().mean() >= 0.8]


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def splits(c):
    x = num(D0, c).dropna()
    if x.nunique() == 2:
        return [(f"{c}==hi", lambda s, v=x.max(): s >= v)]
    out = []
    if (x > 0).any() and (x <= 0).any():
        out.append((f"{c}>0", lambda s: s > 0))
    for nm, q, op in [("med", .5, "gt"), ("q67", 2 / 3, "gt"), ("q33", 1 / 3, "le")]:
        v = x.quantile(q)
        out.append((f"{c}{'>' if op == 'gt' else '<='}{nm}({v:.4g})", (lambda s, v=v: s > v) if op == "gt" else (lambda s, v=v: s <= v)))
    return out


def delta(df, c, f, macro):
    s = num(df, c); ok = s.notna(); g = df[ok]; inz = f(s[ok]).values
    if inz.sum() < 10 or (~inz).sum() < 10:
        return np.nan, np.nan, int(inz.sum()), np.nan, np.nan
    if macro:   # day units: each day's in-mean and out-mean are single observations
        a = g[inz].groupby("day").y.mean(); b = g[~inz].groupby("day").y.mean()
        dl = a.mean() - b.mean()
    else:
        dl = g.y[inz].mean() - g.y[~inz].mean()
    _, _, _, p = day_boot_delta(g.y.values, inz, g.day.values, n=1000)
    return dl, p, int(inz.sum()), g.y[inz].mean(), g.y[~inz].mean()


rows = []
for c in cols:
    mac = is_macro(c)
    for nm, f in splits(c):
        r = dict(test=nm, macro=mac)
        for rg in REGIMES:
            dl, p, n, ai, ao = delta(d[d.regime == rg], c, f, mac)
            r.update({f"{rg[:7]}_d": dl, f"{rg[:7]}_P+": p, f"{rg[:7]}_n": n, f"{rg[:7]}_in": ai, f"{rg[:7]}_out": ao})
        for lab, df in [("Mall", M), ("Mkeep", Mk)]:
            if c in df:
                dl, p, n, ai, ao = delta(df, c, f, mac)
                r.update({f"{lab}_d": dl, f"{lab}_n": n, f"{lab}_cov": num(df, c).notna().mean()})
        rows.append(r)
R = pd.DataFrame(rows)
sg = np.sign(R["2025 BU_d"])
for rg in ["2025 BU", "2026-H2", "2026-H1"]:
    R[f"{rg}_Pdir"] = np.where(sg > 0, R[f"{rg}_P+"], 1 - R[f"{rg}_P+"])
R["same_sign_3"] = (np.sign(R["2025 BU_d"]) == np.sign(R["2026-H2_d"])) & (np.sign(R["2025 BU_d"]) == np.sign(R["2026-H1_d"]))
R["bull_both_P90"] = (R["2025 BU_Pdir"] >= 0.90) & (R["2026-H2_Pdir"] >= 0.90)
R["score"] = R[["2025 BU_Pdir", "2026-H2_Pdir", "2026-H1_Pdir"]].min(axis=1)
R.to_csv(f"reports/YR6_WL_MOMENTUM_consistency_{SL}.csv", index=False)
pd.set_option("display.width", 260)
print(f"# {SLV}: {len(R)} 1D splits (thresholds frozen in 2025 BULL) · same sign in all 3 regimes: {int(R.same_sign_3.sum())} "
      f"(coin-flip expectation {len(R)/4:.0f}) · P>=0.90 in BOTH bulls: {int(R.bull_both_P90.sum())} · "
      f"P>=0.90 in all three: {int((R.score >= 0.90).sum())}")
show = ["test", "macro", "2025 BU_d", "2025 BU_Pdir", "2025 BU_in", "2026-H2_d", "2026-H2_Pdir", "2026-H2_in", "2026-H1_d",
        "2026-H1_Pdir", "2026-H1_in", "Mall_d", "Mall_n", "Mkeep_d", "Mall_cov"]
print(R.sort_values("score", ascending=False)[show].head(20).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ---------- expectancy bar on the top candidates' blocked zone ----------
print("\n## Locked EXPECTANCY bar on the blocked (worse) side of the top-5 consistent splits — per regime and master (as traded)")
be = {rg: breakeven_wr(d[d.regime == rg].y) for rg in REGIMES}
print("   sleeve breakeven WR: " + " · ".join(f"{rg} {v*100:.1f}%" for rg, v in be.items()) + f" · master kept {breakeven_wr(Mk.y)*100:.1f}%")
for _, r in R.sort_values("score", ascending=False).head(5).iterrows():
    c = max((cc for cc in cols if r.test.startswith(cc)), key=len)
    f = dict(splits(c))[r.test]
    worse_in = r["2025 BU_d"] < 0
    print(f"  ▶ {r.test}  — blocked side = {'IN' if worse_in else 'OUT'}")
    for lab, df, bew in [(rg, d[d.regime == rg], be[rg]) for rg in REGIMES] + [("master all", M, breakeven_wr(M.y))]:
        s = num(df, c); ok = s.notna(); g = df[ok]; inz = f(s[ok]).values
        z = g[inz] if worse_in else g[~inz]
        if len(z) < 5:
            print(f"     {lab:<14} N {len(z)}"); continue
        m, lo, hi, p = day_boot(z.y, z.day, n=2000)
        loss = z[z.y < 0]
        top_day = loss.groupby("day").y.sum().min() / loss.y.sum() if len(loss) else np.nan
        top_pair = loss.groupby("pair").y.sum().min() / loss.y.sum() if len(loss) else np.nan
        kept = g[~inz] if worse_in else g[inz]
        ok_bar = (z.win.mean() if "win" in z else (z.y > 0).mean()) < bew and hi < 0 and z.day.nunique() >= 8 and max(top_day, top_pair) < 0.5 and len(z) >= 15
        print(f"     {lab:<14} N {len(z):5d} days {z.day.nunique():3d} WR {(z.y>0).mean()*100:4.1f}% (BE {bew*100:.1f}%) avg {m:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"top-day {top_day*100:3.0f}% top-pair {top_pair*100:3.0f}% of zone loss → bar {'PASS' if ok_bar else 'fail'} · kept side avg {kept.y.mean():+.3f} (N {len(kept)})")
