"""YR6 WL MOMENTUM — step 6: the two ALREADY-REGISTERED observe-only ML candidates (CURRENT_STATE, DECISION_LOG 273)
read on the 2-year replay, frozen definitions, no re-fit. Proxies (flagged): BTC24H_DOWN uses entry_btc_1d_ret_pct <= 0
(the registered zone uses the last CLOSED 5m close vs 288 bars earlier — not this stamp); ADX_RISING uses the stamped
forming-bar entry_adx_delta (registered zone = closed-bar ADX delta, rebuilt from klines). Day-clustered.

    venv/bin/python scripts/study_yr6wl_momentum_registered.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import REGIMES, day_boot, load_master_mom, load_replay

d = load_replay(["MOM-long"])
M = load_master_mom(); M = M[M.sleeve == "MOM-long"]
def line(g, lab):
    if len(g) < 5: return f"    {lab:<34} N {len(g)}"
    m, lo, hi, p = day_boot(g.y, g.day, n=2000)
    return f"    {lab:<34} N {len(g):5d} days {g.day.nunique():3d} WR {(g.y>0).mean()*100:4.1f}% avg {m:+.3f} [{lo:+.3f},{hi:+.3f}] P>0 {p:.2f}"
for lab, df in [(rg, d[d.regime == rg]) for rg in REGIMES] + [("master all (as traded)", M), ("master ex-BASE", M[M.era != "BASE"])]:
    r1 = pd.to_numeric(df.entry_btc_1d_ret_pct, errors="coerce"); ad = pd.to_numeric(df.entry_adx_delta, errors="coerce")
    print(f"## {lab}  (btc_1d_ret scored {r1.notna().mean()*100:.0f}%)")
    print(line(df, "sleeve"))
    print(line(df[r1 <= 0], "① BTC24H_DOWN proxy zone (blocked)"))
    k1 = df[r1 > 0]
    print(line(k1, "① kept"))
    print(line(k1[ad[r1 > 0] <= 0], "② ADXΔ<=0 zone within ① kept"))
    print(line(k1[ad[r1 > 0] > 0], "①+② kept"))
    print(line(df[ad <= 0], "ADXΔ<=0 alone (zone)"))
    print(line(df[ad > 0], "ADXΔ>0 alone (kept)"))
