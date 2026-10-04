#!/usr/bin/env python3
"""🎯 Oct-4 (operator: "we should know this") — the FRENZY / WIDE exit table re-run on REAL TICKS (Binance aggTrades), not 1-minute bars.

Trades: the quiet-market FRENZY first candles of the year (global volume < 1.0; scripts/frenzy_global_volume_test.py → frenzy_gvr_trades.csv),
entry = the same price as the 1-minute study (close of the first minute after the signal close, LAG 1). From that moment every trade print
for 12 h: stop −3 % (exit at the print that crossed), trail armed once the running high reaches +arm %, exit at the first print ≤ running
high × (1 − give %); fixed TP filled at the target. Costs 0.09 % fees + 0.02 % slip on every exit (same as the 1-minute table).
Usage: S=<dir with frenzy_gvr_trades.csv> venv/bin/python scripts/frenzy_exit_ticks.py"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _a, sys.argv = sys.argv, ["x"]
import frenzy_scalp_pattern_search_v2 as P2, frenzy_scalp_followup as FU  # noqa: E402
sys.argv = _a; H = P2.H
S = os.environ["S"]; C = "reports/backtest_cache"; SPLIT = "2026-05-01"; COST = 0.11; HOLD = 12 * 3600_000
Y = pd.read_csv(f"{S}/frenzy_gvr_trades.csv"); Y = Y[Y.gvr < 1.0].reset_index(drop=True)
idx = FU.episode_index()
EX = [("Today: arm +5, give back 1.5", "trail", 5, 1.5)] + [(f"Arm +{a:g}, give back {g:g}", "trail", a, g)
      for a, g in ((2, 0.5), (2, 1.0), (3, 0.5), (3, 1.0), (3, 1.5), (4, 0.5), (4, 1.0), (5, 1.0))] + \
     [(f"Fixed +{t:g} / −3", "tp", t, None) for t in (2, 3, 4)] + \
     [("Bull-Run TP + −3 (BE +1 → +0.2 · 2×ATR trail · ladder)", "bullrun", 2.0, None), ("Bull-Run with 1×ATR trail + −3", "bullrun", 1.0, None),
      ("Bull-Run NO break-even lock: 2×ATR trail · ladder · −3", "bullrun", 2.0, "nobe"), ("Bull-Run NO break-even lock: 1×ATR trail · ladder · −3", "bullrun", 1.0, "nobe")]
LADDER = [(4.0, 3.5), (5.0, 4.5), (6.0, 5.5), (8.0, 7.0), (10.0, 9.0), (12.0, 11.0), (15.0, 13.5), (20.0, 18.0), (25.0, 22.5), (30.0, 27.0)]


def ticks(pair, t0, t1):
    out_t, out_p = [], []
    for d in pd.date_range(pd.Timestamp(t0, unit="ms").normalize(), pd.Timestamp(t1, unit="ms").normalize()):
        for base in ("ticks_q", "ticks"):
            f = f"{C}/{base}/{pair}/{d:%Y-%m-%d}.npz"
            if os.path.exists(f):
                z = np.load(f); out_t.append(z["t"]); out_p.append(z["p"].astype(np.float64)); break
        else:
            return None
    t = np.concatenate(out_t); p = np.concatenate(out_p); o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= t0) & (t <= t1)
    return t[m], p[m]


rows = []
for r in Y.itertuples():
    pth = FU.path(idx, r.pair, int(r.t))
    if pth is None:
        continue
    tt, hh, ll, cc = pth; e = float(cc[0]); te = int(tt[0]) + 60_000
    tk = ticks(r.pair, te, te + HOLD)
    if tk is None or len(tk[1]) < 10:
        rows.append(dict(pair=r.pair, day=r.day, fz=r.fz, ok=False)); continue
    _, p = tk; hi = np.maximum.accumulate(p); res = dict(pair=r.pair, day=r.day, fz=r.fz, ok=True)
    ist = np.flatnonzero(p <= e * 0.97); i_stop = ist[0] if len(ist) else len(p)
    for name, kind, a, g in EX:
        if kind == "bullrun":                              # scripts/frenzy_exit_bullrun_test.line_for on the running high of PRIOR prints
            pk = np.concatenate([[0.0], (hi[:-1] / e - 1) * 100])
            line = np.full(len(p), -3.0)
            on = pk >= 1.0
            line[on] = np.maximum(line[on], (pk[on] - a * float(r.atr)) if g == "nobe" else np.maximum(0.2, pk[on] - a * float(r.atr)))
            for thr, fl in LADDER:
                line[pk >= thr] = np.maximum(line[pk >= thr], fl)
            hit = np.flatnonzero(p <= e * (1 + line / 100)); i = hit[0] if len(hit) else len(p)
            res[name] = ((p[i] / e - 1) * 100 if i < len(p) else (p[-1] / e - 1) * 100) - COST
            continue
        if kind == "tp":
            itp = np.flatnonzero(p >= e * (1 + a / 100)); i_tp = itp[0] if len(itp) else len(p)
            if i_tp < i_stop:
                res[name] = a - COST
            elif i_stop < len(p):
                res[name] = (p[i_stop] / e - 1) * 100 - COST
            else:
                res[name] = (p[-1] / e - 1) * 100 - COST
            continue
        arm = np.flatnonzero(hi >= e * (1 + a / 100)); i_arm = arm[0] if len(arm) else len(p)
        if i_arm < len(p):
            hit = np.flatnonzero(p[i_arm:] <= hi[i_arm:] * (1 - g / 100)); i_tr = i_arm + hit[0] if len(hit) else len(p)
        else:
            i_tr = len(p)
        i = min(i_stop, i_tr)
        res[name] = ((p[i] / e - 1) * 100 if i < len(p) else (p[-1] / e - 1) * 100) - COST
    rows.append(res)
T = pd.DataFrame(rows); T.to_csv(f"{S}/frenzy_exit_ticks.csv", index=False)
K = T[T.ok].copy(); h1 = K.day < SPLIT; base = EX[0][0]
L = [f"# 🎯 FRENZY / WIDE exits on REAL TICKS — quiet-market first candles, full year", "",
     f"{len(K)} of {len(T)} trades priced on ticks ({len(T) - len(K)} without a tick archive). Same entries and costs as the 1-minute table.", "",
     "| Exit | Won | Avg per trade | Jan–Apr / May–Sep | vs today [95 % by day] | Months better | FRENZY part | WIDE part |", "|---|---|---|---|---|---|---|---|"]
for name, *_ in EX:
    x = K[name]; d = x - K[base]
    if name == base:
        ci, mb = "–", "–"
    else:
        bd = H.boot(K.assign(net=d), "day", 1500); ci = f"{d.mean():+.3f} [{bd[0]:+.3f}, {bd[1]:+.3f}]"
        mm = d.groupby(K.day.str[:7]).mean(); mb = f"{(mm > 0).sum()} of {len(mm)}"
    L.append(f"| {name} | {(x > 0).mean() * 100:.0f}% | **{x.mean():+.3f}** | {x[h1].mean():+.3f} / {x[~h1].mean():+.3f} | {ci} | {mb} | "
             f"{K[K.fz][name].mean():+.3f} | {K[~K.fz][name].mean():+.3f} |")
open("reports/FRENZY_EXIT_TICKS_2026-10-04.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
