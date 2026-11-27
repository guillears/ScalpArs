#!/usr/bin/env python3
"""DIST_ZONE_2D_2026-11-27 — DIST crash zone (BTC ≥ 20 % below its 30-day high) × coin-level short crowding (operator: "yes run the 2D test").
Research only. Prior: DL 627 squeeze fuel (funding / OI, all markets) not supported; DIST_WL 11-18 2D screen had no BTC-drawdown variable.
PRE-REGISTERED (frozen before any result was read): backtest kept DIST fills (1,536, Exit B, today's costs). Z = BTC prior-day close ≤ −20 %
below its 30-day high. Crowding variables, theory direction "more shorts crowded" beyond the median of ALL fills: prem_last LOW · glob_ls LOW ·
top_pos_ls LOW · oi_chg4h HIGH · fund_last LOW. Cells per variable: Z ∧ crowded · Z ∧ not crowded · crowded in ALL markets (no Z). Plus Z
alone. Each cell = the cohort a block would remove: N · days · drawdown episodes · WR · avg · DESIGN / HOLDOUT · P(mean < 0) by day-block
bootstrap · the sleeve before → after (one size 0.61: DCR / max DD / worst day). LUCK: pick-the-best null — labels shuffled across trades 1,000×,
the most negative cell mean (N ≥ 15) each time → family-wise p of the observed best cell. Breakeven WR 35.6 %.
Usage: venv/bin/python -I scripts/DIST_ZONE_2D_2026-11-27.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
sp = importlib.util.spec_from_file_location("dd", os.path.join(HERE, "DIST_DEEP_DIVE_2026-11-27.py"))
DD = importlib.util.module_from_spec(sp); sys.argv = sys.argv[:1]; sp.loader.exec_module(DD)
OUT = os.path.join(R, "reports", "DIST_ZONE_2D_2026-11-27")
HOL = DD.HOL
VARS = [("prem_last", -1, "perp premium low"), ("glob_ls", -1, "global long/short low"), ("top_pos_ls", -1, "top-trader position L/S low"),
        ("oi_chg4h", +1, "open interest 4 h growth high"), ("fund_last", -1, "funding low")]


def boot_p_neg(net, day, B=3000, seed=11):
    rng = np.random.default_rng(seed); ud = np.unique(day); idx = {d: np.flatnonzero(day == d) for d in ud}
    m = [np.concatenate([idx[d] for d in rng.choice(ud, len(ud))]) for _ in range(B)]
    return float(np.mean([net[s].mean() < 0 for s in m]))


def main():
    os.makedirs(OUT + "_data", exist_ok=True)
    t = pd.read_csv(os.path.join(R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), low_memory=False)
    K = t[t.rsi1h >= 74.07].sort_values("entry_ms").reset_index(drop=True)
    K["net"] = K.net_LIVE
    dd = DD.btc_dd()
    K["dd30"] = pd.to_datetime(K.entry_ms, unit="ms").dt.floor("D").map(dd)
    Z = (K.dd30 <= -20).fillna(False).values
    zd = sorted(set(pd.to_datetime(K.entry_ms[Z], unit="ms").dt.floor("D")))
    epi, k = {}, 0
    for i, d in enumerate(zd):
        if i and (d - zd[i - 1]).days > 3:
            k += 1
        epi[d] = k
    K["ep"] = pd.to_datetime(K.entry_ms, unit="ms").dt.floor("D").map(epi)
    cells = {"Z alone (BTC ≤ −20 % below 30 d high)": Z}
    cov = {}
    for c, sgn, lab in VARS:
        v = pd.to_numeric(K[c], errors="coerce"); cov[c] = v.notna().mean() * 100
        med = v.median()
        cr = ((v < med) if sgn < 0 else (v > med)).fillna(False).values & v.notna().values
        cells[f"Z ∧ {lab}"] = Z & cr
        cells[f"Z ∧ NOT {lab}"] = Z & ~cr & v.notna().values
        cells[f"ALL markets ∧ {lab}"] = cr
    print("coverage %:", {k: round(v, 1) for k, v in cov.items()})
    base = DD.path(K.net, K.day)
    rows = []
    for name, m in cells.items():
        C = K[m]
        if len(C) < 5:
            rows.append(dict(cell=name, N=len(C))); continue
        A = K[~m]
        pa = DD.path(A.net, A.day)
        rows.append(dict(cell=name, N=len(C), days=C.day.nunique(), episodes=int(C.ep.nunique()) if C.ep.notna().any() else 0,
                         WR=(C.net > 0).mean() * 100, avg=C.net.mean(), DESIGN=C.net[C.entry_ms < HOL].mean(), HOLDOUT=C.net[C.entry_ms >= HOL].mean(),
                         P_neg=boot_p_neg(C.net.values, C.day.values), DCR_after=pa[0], DD_after=pa[1], worst_after=pa[2]))
    T = pd.DataFrame(rows)
    print(f"sleeve today: N {len(K)} · avg {K.net.mean():+.2f} · DCR {base[0]:.2f} %/day · max DD {base[1]:.1f} % · worst day {base[2]:.1f} %")
    print(T.round(2).to_string(index=False))
    # pick-the-best null over every cell with N ≥ 15
    rng = np.random.default_rng(650)
    masks = [m for m in cells.values() if m.sum() >= 15]
    obs = min(K.net.values[m].mean() for m in masks)
    nb = []
    for _ in range(1000):
        y = rng.permutation(K.net.values)
        nb.append(min(y[m].mean() for m in masks))
    print(f"\nluck: observed most negative cell mean {obs:+.2f} · shuffled best: median {np.median(nb):+.2f}, 5th pct {np.percentile(nb, 5):+.2f} → "
          f"family-wise p {np.mean(np.array(nb) <= obs):.3f} ({len(masks)} cells)")
    T.to_csv(OUT + "_data/cells.csv", index=False)


if __name__ == "__main__":
    main()
