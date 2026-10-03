"""Builds reports/backtest_cache/gvr_year.pkl — the year global volume ratio (engine definition, CLOSED 5m bars, day top-50 universe, BASE volume = gvr;
 quote-volume twin gvr_q). Input to scripts/frenzy_global_volume_test.py / frenzy_global_volume_exits.py (DECISION_LOG 194)."""
# Year global-volume ratio, engine definition on CLOSED 5m bars: Σ base vol (signal bar) / Σ 48-bar mean base vol, over the day's top-50 universe.
import os, numpy as np, pandas as pd
C = "reports/backtest_cache"
u = pd.read_csv(f"{C}/universe_daily_top50.csv")
pairs = sorted(u.pair.unique()); vol, avg, qv, qa = {}, {}, {}, {}
for p in pairs:
    f = f"{C}/k5m_full/{p}.csv"
    if not os.path.exists(f): continue
    d = pd.read_csv(f, usecols=["open_time", "vol", "qvol"]).drop_duplicates("open_time").set_index("open_time").sort_index()
    vol[p] = d.vol; avg[p] = d.vol.rolling(48).mean(); qv[p] = d.qvol; qa[p] = d.qvol.rolling(48).mean()
V, A, QV, QA = (pd.DataFrame(x) for x in (vol, avg, qv, qa))
day = (V.index // 86_400_000) * 86_400_000
mem = pd.DataFrame(False, index=V.index, columns=V.columns)
for dms, g in u.groupby("date_ms"):
    cols = [c for c in g.pair if c in mem.columns]; mem.loc[day == dms, cols] = True
ok = mem & V.notna() & A.notna()
g = pd.DataFrame({"gvr": V.where(ok).sum(1) / A.where(ok).sum(1), "gvr_q": QV.where(ok).sum(1) / QA.where(ok).sum(1), "n": ok.sum(1)})
g = g[g.n >= 30]
g.to_pickle("reports/backtest_cache/gvr_year.pkl"); print(g.describe().round(3)); print("pairs loaded", len(vol), "of", len(pairs))
