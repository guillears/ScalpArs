#!/usr/bin/env python3
"""YR6 WL — FRENZY ① per sleeve × regime table (two ways), ② top-50 fidelity split, ③ monthly, ④ entry_* coverage table.

  venv/bin/python -I scripts/study_yr6wl_frenzy_summary.py   → reports/YR6_WL_FRENZY_summary.md + YR6_WL_FRENZY_coverage.csv

Way 1 = seed 1 (FRENZY analysis cohort), day-bootstrap CI. Way 2 = seeds 2+3 pooled (independent re-computation, different cadence).
Cross-check: regime avg vs reports/ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv (3-seed pooled, built by yr6_reprice_today.py).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_yr6wl_frenzy_common as C  # noqa: E402

out = []
A = C.load(all_seeds=True)
S1 = A[A.seed == 1]
REF = pd.read_csv(os.path.join(C.REP, "ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv"), header=None)

out.append("## ① Per sleeve × regime (y = pct_new, today's exit)\n")
rows = []
for s in C.SLEEVES:
    for r in C.REGIMES:
        a = S1[(S1.sleeve == s) & (S1.regime == r)]
        b = A[(A.sleeve == s) & (A.regime == r) & A.seed.isin([2, 3])]
        f = C.fmt_row(a.y.values, a.day.values)
        ref = REF[(REF[0] == s) & (REF[1] == r)]
        rows.append(dict(sleeve=s, regime=r, **f, avg_seeds23=round(b.y.mean(), 3), WR_seeds23=round(100 * b.win.mean(), 1),
                         ref3seed_avg=round(float(ref[4].iloc[0]), 3) if len(ref) else np.nan,
                         usd_seed1=round(a.usd_new.sum(), 0)))
    a = S1[S1.sleeve == s]
    rows.append(dict(sleeve=s, regime="ALL", **C.fmt_row(a.y.values, a.day.values), avg_seeds23=round(A[(A.sleeve == s) & A.seed.isin([2, 3])].y.mean(), 3),
                     usd_seed1=round(a.usd_new.sum(), 0)))
T1 = pd.DataFrame(rows)
out.append(C.md_table(T1))

out.append("\n\n## ② Top-50 fidelity split (seed 1) — 2025 findings must hold inside top50\n")
rows = []
for s in C.SLEEVES:
    for r in C.REGIMES:
        for t in (True, False):
            a = S1[(S1.sleeve == s) & (S1.regime == r) & (S1.top50 == t)]
            if len(a) == 0:
                continue
            f = C.fmt_row(a.y.values, a.day.values)
            rows.append(dict(sleeve=s, regime=r, top50=t, N=f["N"], days=f["days"], WR=f["WR"], avg=f["avg"], ci=f["ci"]))
out.append(C.md_table(pd.DataFrame(rows)))

out.append("\n\n## ③ Monthly avg % (seed 1; N in brackets; top-50-only avg after the slash)\n")
rows = []
for s in C.SLEEVES:
    a = S1[S1.sleeve == s]
    g = a.groupby("month").agg(N=("y", "size"), avg=("y", "mean"), wr=("win", "mean"))
    gt = a[a.top50].groupby("month").y.mean()
    rows.append({"sleeve": s, **{m: f"{g.loc[m, 'avg']:+.2f} ({int(g.loc[m, 'N'])}) / {gt.get(m, np.nan):+.2f}" for m in g.index}})
M = pd.DataFrame(rows).set_index("sleeve").T.reset_index().rename(columns={"index": "month"}).fillna("")
out.append(C.md_table(M))

out.append("\n\n## ④ Coverage — non-null share of every entry_* column per sleeve × regime (seed 1)\n")
cols = C.entry_cols(S1)
cov = []
for c in cols:
    x = C.num(S1, c)
    row = dict(col=c, kind="MACRO" if C.is_macro(c) else "pair")
    for s in C.SLEEVES:
        for r in C.REGIMES:
            m = (S1.sleeve == s) & (S1.regime == r)
            row[f"{s.split('_')[1]}·{C.RSHORT[r]}"] = round(float(x[m].notna().mean()), 2)
    row["nuniq"] = int(x.nunique())
    cov.append(row)
COV = pd.DataFrame(cov)
COV.to_csv(os.path.join(C.REP, "YR6_WL_FRENZY_coverage.csv"), index=False)
low = COV[(COV.iloc[:, 2:-1] < 0.8).any(axis=1) & (COV.iloc[:, 2:-1] > 0).any(axis=1)]
full = COV[(COV.iloc[:, 2:-1] >= 0.8).all(axis=1)]
none = COV[(COV.iloc[:, 2:-1] == 0).all(axis=1)]
out.append(f"{len(cols)} entry_* columns with any value. ≥ 80 % in every sleeve × regime: {len(full)}. "
           f"Below 80 % somewhere (own read only, never zone-vs-rest): {len(low)}. Full table: reports/YR6_WL_FRENZY_coverage.csv\n")
out.append("Columns < 80 % somewhere:\n")
out.append(C.md_table(low.drop(columns=["nuniq"]), floatfmt=2))

open(os.path.join(C.REP, "YR6_WL_FRENZY_summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ── ⑤ own read of the < 80 %-coverage pair columns (scored-only terciles; unscored shown apart, never pooled into "rest")
out2 = ["\n\n## ⑤ Own read — mcap / turnover / CMC rank (coverage < 80 % in some 2025 cells): scored-only terciles vs UNSCORED (seed 1)\n"]
rows = []
for c in ("entry_vol_mcap_ratio", "entry_mcap_usd", "entry_cmc_rank"):
    for s in C.SLEEVES:
        for r in C.REGIMES:
            a = S1[(S1.sleeve == s) & (S1.regime == r)]
            x = pd.to_numeric(a[c], errors="coerce")
            sc = a[x.notna()]; xs = x.dropna()
            if len(sc) < 30:
                continue
            q = xs.quantile([1 / 3, 2 / 3]).values
            t = np.where(xs <= q[0], 0, np.where(xs <= q[1], 1, 2))
            rows.append(dict(col=c, sleeve=s, regime=r, scored=len(sc), T1_low=f"{sc.y[t == 0].mean():+.2f}", T2=f"{sc.y[t == 1].mean():+.2f}",
                             T3_high=f"{sc.y[t == 2].mean():+.2f}", unscored=f"{a.y[x.isna()].mean():+.2f} ({int(x.isna().sum())})"))
out2.append(C.md_table(pd.DataFrame(rows), 2))
open(os.path.join(C.REP, "YR6_WL_FRENZY_summary.md"), "a").write("\n".join(out2) + "\n")
print("\n".join(out2))
