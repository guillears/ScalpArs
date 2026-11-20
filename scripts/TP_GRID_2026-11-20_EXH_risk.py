"""TP_GRID EXH (2026-11-20) — risk-matched read (same method as TP_GRID_2026-11-20_DIST_risk.py): each TP is sized so its backtest
max drawdown is −30 % (and −20 %), then its daily compound return at that size is compared with today's exit (REF);
10-day-block bootstrap P(better), paired by day. Cohort = the TP_GRID EXH pool (FRENZY + WIDE, today's stack), today's stop + 24 h."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("ev", os.path.join(HERE, "TP_GRID_2026-11-20_EXH_eval.py")); EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
X, C = EV.build(["FRENZY", "WIDE"])
CELLS = ["REF", "F9", "F8", "F7", "F6", "F5"]
J = {n: EV.NAMES.index(n) for n in CELLS}

def daily(j, m, d0, d1):
    xd = C["xm"][m, j] // EV.DAY
    return pd.Series(C["NET"][m, j]).groupby(xd).sum().reindex(np.arange(d0, d1 + 1), fill_value=0.0).values

def stats(r, s):
    g = np.maximum(1 + r * s / 100, 1e-9); eq = np.cumprod(g)
    return (eq[-1] ** (1 / len(g)) - 1) * 100, (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100, (g.min() - 1) * 100

def size_for_dd(r, tgt):
    lo, hi = 0.0, 50.0   # wide bracket: a resample without the cluster days may need a large size to reach the target
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if stats(r, mid)[1] > tgt else (lo, mid)
    return lo

rng = np.random.default_rng(17); rows = []
for blk in ("ALL", "HOLDOUT"):
    m = np.ones(len(C["ed"]), bool) if blk == "ALL" else C["ed"] >= EV.DH
    d0 = int(C["ed"][m].min()); d1 = int(max((C["xm"][m, J[c]] // EV.DAY).max() for c in CELLS))
    R = {c: daily(J[c], m, d0, d1) for c in CELLS}
    for c, r in R.items():
        a = stats(r, 0.61); row = dict(block=blk, cell=c, N=int(m.sum()), dcr_061=a[0], dd_061=a[1], worst_061=a[2])
        for tgt in (-20, -30):
            s = size_for_dd(r, tgt); b = stats(r, s)
            row[f"size_dd{-tgt}"] = s; row[f"dcr_dd{-tgt}"] = b[0]; row[f"worst_dd{-tgt}"] = b[2]
        rows.append(row)
    n = len(R["REF"]); nb = n // 10; res = {c: [] for c in CELLS}
    for _ in range(300):
        st = rng.integers(0, n - 9, nb); idx = (st[:, None] + np.arange(10)).ravel()
        for c, r in R.items():
            rr = r[idx]; res[c].append(stats(rr, size_for_dd(rr, -30))[0])
    for row in rows:
        if row["block"] == blk:
            dl = np.array(res[row["cell"]]) - np.array(res["REF"])
            row.update(p_better_eqdd30=(dl > 0).mean(), gain_med=np.median(dl), gain_lo90=np.quantile(dl, .05), gain_hi90=np.quantile(dl, .95))
O = pd.DataFrame(rows).round(3)
O.to_csv(os.path.join(os.path.dirname(HERE), "reports", "TP_GRID_2026-11-20_EXH_risk_matched.csv"), index=False)
pd.set_option("display.width", 250); print(O.to_string(index=False))
