#!/usr/bin/env python3
"""EXH_THEORY_EXIT — structure-stop vs fixed-stop CONTROL (post-hoc, labelled) with risk: avg win / loss, worst day, max drawdown,
one-size DCR (0.61 × equity per fill, compounding by exit day). Research only."""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sp = importlib.util.spec_from_file_location("EV", os.path.join(R, "scripts", "EXH_THEORY_EXIT_2026-11-21_eval.py")); EV = importlib.util.module_from_spec(sp); sp.loader.exec_module(EV)
D = EV.D
X = pd.read_pickle(os.path.join(R, "reports", "EXH_EXIT2_data", "rows_EXH.pkl"))
Z = dict(np.load(os.path.join(D, "sim_BT_ctrl.npz")))
ok = Z["ok"]; X = X[ok].reset_index(drop=True); Z = {k: (v[ok] if v.ndim and len(v) == len(ok) else v) for k, v in Z.items()}
names = list(Z["names"]); N = EV.nets(X.slip.values, Z["r"].astype(float), Z["k"], Z["f"].astype(float))
N2 = EV.nets(X.slip.values, Z["r"].astype(float), Z["k"], Z["f"].astype(float), flat=0.20)
per = np.where(X.entry_ms.values < EV.SPLIT, "DESIGN", "HOLDOUT")
dist = (Z["peak"] / (X.entry_ms.values * 0 + 1)).astype(float)
def path(net, xm):
    s = pd.Series(net * EV.X_EQ / 100).groupby(pd.to_datetime(xm, unit="ms").normalize()).sum()
    days = pd.date_range(s.index.min(), s.index.max(), freq="D"); r = s.reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r); dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return r.min() * 100, dd, ((eq[-1]) ** (1 / len(r)) - 1) * 100
rows = []
for nm in ("REF", "C_PK0", "C_PK1", "W_PK0", "C_PK0_NOCAP", "K_SL7", "K_SL8", "K_SL10", "K_SL12", "K_SL15"):
    j = names.index(nm)
    for p in ("DESIGN", "HOLDOUT"):
        m = per == p; n_ = N[m, j]; d = n_ - N[m, 0]
        lo, hi = EV.boot(d, X.entry_ms.values[m] // 86_400_000)
        hp = pd.Series(d).groupby(X.half.values[m]).sum()
        wd, dd, dc = path(n_, Z["xm"][m, j])
        rows.append(dict(cell=nm, period=p, N=int(m.sum()), WR=(n_ > 0).mean() * 100, avg_win=n_[n_ > 0].mean(), avg_loss=n_[n_ <= 0].mean(),
                         worst_trade=n_.min(), avg=n_.mean(), d=d.mean(), ci=f"[{lo:+.2f}, {hi:+.2f}]", halves=f"{int((hp > 0).sum())}/{len(hp)}",
                         d020=(N2[m, j] - N2[m, 0]).mean(), sl_pct=(Z["k"][m, j] == 1).mean() * 100, worst_day=wd, maxDD=dd, DCR=dc))
pd.set_option("display.width", 250)
T = pd.DataFrame(rows); print(T.round(2).to_string(index=False))
T.to_csv(os.path.join(D, "risk_ctrl.csv"), index=False)
pk = Z["peak"]; E = None
print("\nC_PK0 stop distance above entry (% , anchored fills) — needs entry price; using REF stop vs peak ratio from paths:")
