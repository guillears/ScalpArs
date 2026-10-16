#!/usr/bin/env python3
"""LITE MACRO-REGIME study — POST-HOC diagnostic (declared in the report, not a pre-registered test): is there a slow regime at all?
(a) lag-1/2/4 autocorrelation of LITE's weekly avg %/fill within each year vs a week-shuffle null; (b) month eta^2 vs a month-label
shuffle null; (c) year shift inside BTC-grid cells (grid_cells*.csv). Usage: venv/bin/python scripts/study_lite_regime_persist.py [--tick]
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "lite_regime")
TICK = "--tick" in sys.argv
RNG = np.random.default_rng(7)
P = pd.read_csv(os.path.join(ROOT, "reports", "LITE_PROTECT_rows.csv"), usecols=["block", "era", "day", "src", "rm_base_pct"])
P = P[P.block == "BT"]
if TICK:
    P = P[P.src.astype(str).str.startswith("ticks")]
P["yr"] = P.era.str[:4]; P["dt"] = pd.to_datetime(P.day)
P["week"] = P.dt - pd.to_timedelta(P.dt.dt.weekday, unit="D"); P["month"] = P.day.str[:7]
lines = [f"# persistence diagnostic{' (tick-priced only)' if TICK else ''}"]
for yr, d in P.groupby("yr"):
    w = d.groupby("week").rm_base_pct.mean()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="7D"))
    for lag in (1, 2, 4):
        a, b = w.values[:-lag], w.values[lag:]
        ok = ~np.isnan(a) & ~np.isnan(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        nv = []
        for _ in range(2000):
            s = w.values.copy(); m = ~np.isnan(s); s[m] = RNG.permutation(s[m])
            a2, b2 = s[:-lag], s[lag:]; o = ~np.isnan(a2) & ~np.isnan(b2)
            nv.append(np.corrcoef(a2[o], b2[o])[0, 1])
        lines.append(f"{yr} weekly avg lag-{lag} autocorr r = {r:+.3f} (weeks {ok.sum()}), shuffle p(r >= obs) = {(np.array(nv) >= r).mean():.3f}")
    y = d.rm_base_pct.values; mo = d.month.values
    def eta(lab):
        g = pd.Series(y).groupby(lab)
        return ((g.mean() - y.mean()) ** 2 * g.size()).sum() / ((y - y.mean()) ** 2).sum()
    e = eta(mo)
    # null: shuffle whole DAYS between months (keeps same-day clustering)
    days = d.day.values; ud = np.unique(days)
    nv = []
    for _ in range(1000):
        mp = dict(zip(ud, RNG.permutation(ud)))
        nv.append(eta(pd.Series([mp[x][:7] for x in days]).values))
    lines.append(f"{yr} month eta^2 = {e:.4f}, day-shuffle null mean {np.mean(nv):.4f}, p = {(np.array(nv) >= e).mean():.3f}")
# year effect
y25, y26 = P[P.yr == "2025"].rm_base_pct, P[P.yr == "2026"].rm_base_pct
lines.append(f"year level: 2025 {y25.mean():+.3f} (N {len(y25)}) vs 2026 {y26.mean():+.3f} (N {len(y26)})")
G = pd.read_csv(os.path.join(OUT, "grid_cells_tick.csv" if TICK else "grid_cells.csv"))
g = G[(G.wk25 >= 8) & (G.wk26 >= 8)]
lines.append(f"BTC-grid cells with >= 8 weeks in both years: {len(g)}; cells where 2026 avg > 2025 avg: {(g.avg26 > g.avg25).sum()} "
             f"(mean shift {(g.avg26 - g.avg25).mean():+.3f} %/fill)")
txt = "\n".join(lines)
open(os.path.join(OUT, "persist_tick.txt" if TICK else "persist.txt"), "w").write(txt + "\n")
print(txt)
