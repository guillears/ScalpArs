#!/usr/bin/env python3
"""EXH — fixed take profit 5 / 7 / 8 / 9 % vs today's (min(20 × ATR, 10 %)), same stop / 24 h. Backtest DESIGN / HOLDOUT / E 2023-H1,
with duration (hours) before / after, risk and one-size DCR. Descriptive (TP_GRID DL 603 is the decision study). Research only."""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import exh_exit2_eval as E2E   # noqa: E402
sp = importlib.util.spec_from_file_location("EV", os.path.join(R, "scripts", "EXH_THEORY_EXIT_2026-11-21_eval.py")); EV = importlib.util.module_from_spec(sp); sp.loader.exec_module(EV)
def block(tag):
    X = pd.read_pickle(os.path.join(R, "reports", "EXH_EXIT2_data", f"rows_{tag}.pkl"))
    Z = dict(np.load(os.path.join(EV.D, "sim_BT_tpf.npz" if tag == "EXH" else f"sim_BT_tpf_{tag}.npz")))
    ok = Z["ok"].copy()
    if tag == "E":
        _, blk = E2E.e_block(X); ii = np.nonzero(ok & ~blk)[0]
        ii = ii[E2E.one_pos(X.pair.values[ii], X.entry_ms.values[ii].astype(np.int64), Z["xm"][ii, 0])]
        sel = {"E 2023-H1": ii}
    else:
        per = X.entry_ms.values < EV.SPLIT
        sel = {"DESIGN": np.nonzero(ok & per)[0], "HOLDOUT": np.nonzero(ok & ~per)[0]}
    return X, Z, sel
rows = []
for tag in ("EXH", "E"):
    X, Z, sel = block(tag); names = list(Z["names"])
    N = EV.nets(X.slip.values.astype(float), Z["r"].astype(float), Z["k"], Z["f"].astype(float))
    for bn, ii in sel.items():
        for nm in ("REF", "TPF5", "TPF7", "TPF8", "TPF9"):
            j = names.index(nm); n_ = N[ii, j]; d = n_ - N[ii, 0]
            lo, hi = EV.boot(d, X.entry_ms.values[ii] // 86_400_000)
            dur = (Z["xm"][ii, j] - X.entry_ms.values[ii]) / 3.6e6
            s = pd.Series(n_ * 0.61 / 100).groupby(pd.to_datetime(Z["xm"][ii, j], unit="ms").normalize()).sum()
            days = pd.date_range(s.index.min(), s.index.max(), freq="D"); r = s.reindex(days, fill_value=0).values; eq = np.cumprod(1 + r)
            rows.append(dict(block=bn, exit=nm, N=len(ii), WR=(n_ > 0).mean() * 100, avg=n_.mean(), d=d.mean(), ci=f"[{lo:+.2f},{hi:+.2f}]",
                             avg_win=n_[n_ > 0].mean(), avg_loss=n_[n_ <= 0].mean(), tp=(Z["k"][ii, j] == 0).mean() * 100, sl=(Z["k"][ii, j] == 1).mean() * 100,
                             dur_avg_h=dur.mean(), dur_med_h=np.median(dur), worst_day=r.min() * 100, maxDD=((eq / np.maximum.accumulate(eq)) - 1).min() * 100,
                             DCR=(eq[-1] ** (1 / len(r)) - 1) * 100))
T = pd.DataFrame(rows); pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
T.to_csv(os.path.join(EV.D, "tpfix_bt.csv"), index=False)
