#!/usr/bin/env python3
"""EXH_THEORY_EXIT — confirmation on the untouched 2023-H1 E block (DL 582 block + one position per pair under REF, as EXH_EXIT2)."""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import exh_exit2_eval as E2E   # noqa: E402
sp = importlib.util.spec_from_file_location("EV", os.path.join(R, "scripts", "EXH_THEORY_EXIT_2026-11-21_eval.py")); EV = importlib.util.module_from_spec(sp); sp.loader.exec_module(EV)
XE = pd.read_pickle(os.path.join(R, "reports", "EXH_EXIT2_data", "rows_E.pkl"))
Z = dict(np.load(os.path.join(EV.D, "sim_BT_ctrl_E.npz")))
assert (Z["xi"] == XE.xi.values).all()
oiv, blk = E2E.E_block(XE) if hasattr(E2E, "E_block") else E2E.e_block(XE)
keep = ~blk & Z["ok"]
ii = np.nonzero(keep)[0]
op = E2E.one_pos(XE.pair.values[ii], XE.entry_ms.values[ii].astype(np.int64), Z["xm"][ii, 0])
ii = ii[op]
names = list(Z["names"]); N = EV.nets(XE.slip.values[ii], Z["r"][ii].astype(float), Z["k"][ii], Z["f"][ii].astype(float))
print("E block fills", len(ii))
for nm in ("REF", "K_SL7", "K_SL8", "K_SL10", "K_SL12"):
    j = names.index(nm); n_ = N[:, j]; d = n_ - N[:, 0]
    lo, hi = EV.boot(d, XE.entry_ms.values[ii] // 86_400_000)
    s = pd.Series(n_ * 0.61 / 100).groupby(pd.to_datetime(Z["xm"][ii, j], unit="ms").normalize()).sum()
    days = pd.date_range(s.index.min(), s.index.max(), freq="D"); r = s.reindex(days, fill_value=0).values; eq = np.cumprod(1 + r)
    print(f"{nm:7s} WR {(n_ > 0).mean() * 100:5.1f} avg {n_.mean():+.2f} Δ {d.mean():+.3f} [{lo:+.2f},{hi:+.2f}] worst day {r.min() * 100:+.1f}% maxDD {((eq / np.maximum.accumulate(eq)) - 1).min() * 100:+.1f}% DCR {(eq[-1] ** (1 / len(r)) - 1) * 100:+.3f}")
