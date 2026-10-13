"""YR6 WL MOMENTUM — step 3: why does the replay (ML −0.19 / −0.09 / −0.02 %/fill) disagree with the real master
fills (136 kept · +0.16 % · +$1,564)?  Overlap months only (master spans 2026-06-18 → 2026-10-12).

Decomposition:
  ① outcome level: replay over the same calendar window, and over the live-active era spans only
  ② selection: real as-traded (all non-probe) vs today's-stack kept (in-sample filters) — the 52 blocked fills
  ③ trade-level reproduction: each real fill matched to replay fills (same pair + direction, open within ±15 min,
     any seed) — matched outcome real vs replay (exit fidelity on the SAME trade) and unmatched real / replay-only
  ④ cohort: every entry_* column, replay (window) vs real kept — median, standardised mean diff, KS distance
     (columns scored on < 80 % of real fills are flagged; their read is on scored rows only)
  ⑤ washed-out window (Jun-18 → Jul-2, BTC 15-24 % below 30d high) shown with AND without.

    venv/bin/python scripts/study_yr6wl_momentum_replay_vs_real.py > reports/YR6_WL_MOMENTUM_replay_vs_real.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import SLEEVES, day_boot, is_macro, load_master_mom, load_replay, signal_columns

R = load_replay()
M = load_master_mom()
W0, W1 = M.ts.min().floor("D"), M.ts.max().ceil("D")
WASH = (pd.Timestamp("2026-06-18"), pd.Timestamp("2026-07-03"))
spans = M.groupby("era").ts.agg(["min", "max"])
Rw = R[(R.ts >= W0) & (R.ts <= W1)].copy()
Rw["live_span"] = False
for _, s in spans.iterrows():
    Rw.loc[(Rw.ts >= s["min"] - pd.Timedelta(minutes=15)) & (Rw.ts <= s["max"] + pd.Timedelta(minutes=15)), "live_span"] = True


def st(y, day, lab, nseed=1):
    y = pd.Series(y).astype(float); day = pd.Series(day)
    ok = y.notna(); y, day = y[ok], day[ok]
    if not len(y):
        return f"  {lab:<58}  —"
    m, lo, hi, p = day_boot(y.values, day.values, n=2000)
    return (f"  {lab:<58}N {len(y):5d} ({len(y)/nseed:6.1f}/seed) · days {day.nunique():3d} · WR {(y>0).mean()*100:4.1f}% · "
            f"avg {m:+.3f} [{lo:+.3f},{hi:+.3f}] P>0 {p:.2f}")


print(f"# Replay vs real — overlap {W0.date()} → {W1.date()} · master eras {', '.join(spans.sort_values('min').index)}")
for sv in SLEEVES:
    m = M[M.sleeve == sv]; mk = m[m.stack_keep == True]; r = Rw[Rw.sleeve == sv]
    print(f"\n## {sv}")
    print("### ① outcome level")
    print(st(m.y, m.day, "REAL as-traded (all non-probe), pnl %"))
    print(st(mk.y_stack, mk.day, "REAL stack-kept (today's filters), stack %"))
    print(st(m[m.stack_keep != True].y, m[m.stack_keep != True].day, "REAL blocked by today's stack, as traded %"))
    print(st(r.y, r.day, "REPLAY same calendar window (3 seeds)", 3))
    print(st(r[r.live_span].y, r[r.live_span].day, "REPLAY inside live-active era spans only", 3))
    nw = lambda df, c="ts": df[~((df[c] >= WASH[0]) & (df[c] < WASH[1]))]
    print("  — ex washed-out window Jun-18→Jul-2:")
    print(st(nw(m).y, nw(m).day, "REAL as-traded ex-wash"))
    print(st(nw(mk).y_stack, nw(mk).day, "REAL kept ex-wash"))
    print(st(nw(r).y, nw(r).day, "REPLAY window ex-wash", 3))
    print(st(nw(r[r.live_span]).y, nw(r[r.live_span]).day, "REPLAY live-spans ex-wash", 3))
    print("  per era (real kept · real all · replay live-span/seed):")
    for era, s in spans.sort_values("min").iterrows():
        a = m[m.era == era]; k = mk[mk.era == era]
        rr = r[(r.ts >= s["min"] - pd.Timedelta(minutes=15)) & (r.ts <= s["max"] + pd.Timedelta(minutes=15))]
        print(f"    {era:<5} {s['min']:%m-%d}→{s['max']:%m-%d}  kept {len(k):3d} {k.y_stack.mean() if len(k) else np.nan:+.3f} · all {len(a):3d} "
              f"{a.y.mean() if len(a) else np.nan:+.3f} · replay {len(rr)/3:5.1f}/seed {rr.y.mean() if len(rr) else np.nan:+.3f}")

    # ③ reproduction
    print("### ③ trade-level reproduction (same pair+dir, open ±15 min, any seed)")
    rr = r.sort_values("ts")
    match_y, match_ry, matched = [], [], np.zeros(len(m), bool)
    used = set()
    for i, (_, f) in enumerate(m.iterrows()):
        c = rr[(rr.pair == f.pair) & ((rr.ts - f.ts).abs() <= pd.Timedelta(minutes=15))]
        if len(c):
            matched[i] = True
            match_y.append(f.y); match_ry.append(c.y.mean()); used.update(c.index)
    m = m.assign(matched=matched)
    print(f"  real fills reproduced by the replay: {matched.sum()}/{len(m)} ({matched.mean()*100:.0f}%) · "
          f"kept {m[m.stack_keep == True].matched.mean()*100:.0f}% · blocked {m[m.stack_keep != True].matched.mean()*100:.0f}% · "
          f"real winners {m[m.y > 0].matched.mean()*100:.0f}% vs real losers {m[m.y <= 0].matched.mean()*100:.0f}%")
    if match_y:
        my, mry = np.array(match_y), np.array(match_ry)
        print(f"  SAME trades (matched {len(my)}): real avg {my.mean():+.3f} · replay avg {mry.mean():+.3f} · "
              f"gap real−replay {np.mean(my - mry):+.3f} (median {np.median(my - mry):+.3f}) · sign agreement {(np.sign(my) == np.sign(mry)).mean()*100:.0f}%")
    um = m[~m.matched]
    print(f"  real fills the replay did NOT take: N {len(um)} avg {um.y.mean():+.3f} (kept {um[um.stack_keep == True].y_stack.mean():+.3f}, N {int((um.stack_keep == True).sum())})")
    ro = r[~r.index.isin(used)]
    print(f"  replay fills live did NOT take: N {len(ro)} ({len(ro)/3:.0f}/seed) avg {ro.y.mean():+.3f} · inside live spans: "
          f"N {ro.live_span.sum()} avg {ro[ro.live_span].y.mean():+.3f} · outside (bot down / reset gaps): {ro[~ro.live_span].y.mean():+.3f}")

    # ④ cohort distributions
    print("### ④ entry_* distributions — REPLAY window (pooled seeds) vs REAL kept (scored rows); sorted by |SMD|")
    cols = [c for c in signal_columns(r) if c in mk.columns]
    out = []
    for c in cols:
        a = pd.to_numeric(r[c], errors="coerce").dropna().values
        b = pd.to_numeric(mk[c], errors="coerce").dropna().values
        cov = pd.to_numeric(mk[c], errors="coerce").notna().mean()
        if len(a) < 10 or len(b) < 8:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2) or np.nan
        grid = np.unique(np.concatenate([a, b]))
        ks = np.max(np.abs(np.searchsorted(np.sort(a), grid, "right") / len(a) - np.searchsorted(np.sort(b), grid, "right") / len(b)))
        # KS critical value at alpha 0.05 (effective n of replay = fills/3 seeds)
        na = len(a) / 3; crit = 1.36 * np.sqrt((na + len(b)) / (na * len(b)))
        out.append(dict(col=c, macro=is_macro(c), real_cov=cov, n_real=len(b), med_replay=np.median(a), med_real=np.median(b),
                        smd=(b.mean() - a.mean()) / sd, ks=ks, ks_crit=crit, sig=ks > crit))
    o = pd.DataFrame(out).sort_values("smd", key=np.abs, ascending=False)
    pd.set_option("display.width", 220)
    print(o.head(25).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"  columns with KS > 5 % critical value: {int(o.sig.sum())}/{len(o)} · |SMD| ≥ 0.3: {int((o.smd.abs() >= 0.3).sum())}")
    o.to_csv(f"reports/YR6_WL_MOMENTUM_replay_vs_real_{sv}.csv", index=False)

    # matched-vs-unmatched replay fills — are the live-taken trades a better cohort inside the replay?
    print("### ⑤ inside the replay window: fills that live ALSO took vs fills live did not take")
    r2 = r.assign(live_took=r.index.isin(used))
    for k, g in r2.groupby("live_took"):
        print(st(g.y, g.day, f"replay fills, live also took = {k}", 3))
