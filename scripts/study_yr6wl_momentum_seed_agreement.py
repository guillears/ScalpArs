"""YR6 WL MOMENTUM — step 4: signal-persistence cohort. The replay's 3 seeds differ only in scan cadence. A fill that
all 3 seeds take (same pair + direction, open within ±W min) is a signal that held across scan phases; a fill only
1 seed takes is a flickering / phase-dependent signal. Hypothesis from step 3: live∩replay matched trades won
(+0.23) while either side's unmatched trades lost (≈ −0.09). Read per sleeve × regime, day-clustered.

    venv/bin/python scripts/study_yr6wl_momentum_seed_agreement.py [window_min=15]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import REGIMES, SLEEVES, day_boot, load_replay

W = pd.Timedelta(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 15)
d = load_replay()
agree = np.zeros(len(d), int)
first = np.zeros(len(d), bool)     # earliest-opening seed of its cluster
lag_min = np.full(len(d), np.nan)
for (sv, pair), g in d.groupby(["sleeve", "pair"]):
    ts = g.ts.values; sd = g.seed.values; idx = g.index.values
    for i in range(len(g)):
        near = np.abs(ts - ts[i]) <= W.to_timedelta64()
        agree[idx[i]] = len(set(sd[near]))
        others = ts[near]
        lag_min[idx[i]] = (ts[i] - others.min()) / np.timedelta64(1, "m")
d["agree"] = agree; d["lag_min"] = lag_min
print(f"# seed-agreement cohort (±{W}) — share · N · WR · avg % [day CI] · P>0")
for sv in SLEEVES:
    print(f"\n## {sv}")
    for rg in REGIMES:
        g0 = d[(d.sleeve == sv) & (d.regime == rg)]
        for a in (1, 2, 3):
            g = g0[g0.agree == a]
            if len(g) < 5: print(f"  {rg:<14} agree {a}/3: N {len(g)}"); continue
            m, lo, hi, p = day_boot(g.y, g.day, n=2000)
            print(f"  {rg:<14} agree {a}/3: {len(g)/len(g0)*100:4.0f}% · N {len(g):5d} · days {g.day.nunique():3d} · WR {g.win.mean()*100:4.1f}% · "
                  f"avg {m:+.3f} [{lo:+.3f},{hi:+.3f}] P>0 {p:.2f}")
        g = g0[g0.agree == 3]
        late = g[g.lag_min > 1]
        print(f"  {rg:<14}   within agree 3/3: first-to-open (lag ≤1 min) avg {g[g.lag_min <= 1].y.mean():+.3f} (N {(g.lag_min<=1).sum()}) · later seeds avg {late.y.mean():+.3f} (N {len(late)})")
d[["sleeve", "regime", "seed", "pair", "t", "y", "agree", "lag_min"]].to_csv("reports/YR6_WL_MOMENTUM_seed_agreement.csv", index=False)
