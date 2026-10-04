#!/usr/bin/env python3
"""🎯 Oct-4 (operator: "you are computing profit and losses not as in our bot") — FRENZY / WIDE exits on REAL TICKS with the BOT's accounting:
levels on NET P&L % (gross − 0.09 % round-trip taker fees, exactly as frenzy_exit_for receives it), filled at the trade print that crosses the
line (the paper bot closes at the live price), no extra slippage. Stop fires when net ≤ −SL, TP when net ≥ +TP; 12 h cap at the last print.
Trades = the quiet-market FRENZY first candles of the year (frenzy_gvr_trades.csv), entry = close of the first minute after the signal.
Usage: S=<dir with frenzy_gvr_trades.csv> venv/bin/python scripts/frenzy_exit_ticks_botexact.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _a, sys.argv = sys.argv, ["x"]
import frenzy_scalp_pattern_search_v2 as P2, frenzy_scalp_followup as FU  # noqa: E402
sys.argv = _a; H = P2.H
S = os.environ["S"]; C = "reports/backtest_cache"; SPLIT = "2026-05-01"; FEES = 0.09; HOLD = 12 * 3600_000
Y = pd.read_csv(f"{S}/frenzy_gvr_trades.csv"); Y = Y[Y.gvr < 1.0].reset_index(drop=True)
idx = FU.episode_index()
FIXED = [(tp, sl) for sl in (2, 3, 4) for tp in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6)]
TRAILS = [(a, g) for a in (2, 3, 4, 5) for g in (0.5, 1.0, 1.5)]
LADDER = [(4.0, 3.5), (5.0, 4.5), (6.0, 5.5), (8.0, 7.0), (10.0, 9.0), (12.0, 11.0), (15.0, 13.5), (20.0, 18.0), (25.0, 22.5), (30.0, 27.0)]
BULL = [("Bull-Run TP (BE lock +0.2 · 2×ATR · ladder) −3", 2.0, True), ("Bull-Run TP (BE lock · 1×ATR · ladder) −3", 1.0, True),
        ("Bull-Run NO lock · 2×ATR · ladder −3", 2.0, False), ("Bull-Run NO lock · 1×ATR · ladder −3", 1.0, False)]
NAMES = [f"fixed +{tp:g}/−{sl:g}" for tp, sl in FIXED] + [f"trail +{a:g}/{g:g} (−3)" for a, g in TRAILS] + [b[0] for b in BULL]


def ticks(pair, t0, t1):
    ts, ps = [], []
    for d in pd.date_range(pd.Timestamp(t0, unit="ms").normalize(), pd.Timestamp(t1, unit="ms").normalize()):
        for base in ("ticks_q", "ticks"):
            f = f"{C}/{base}/{pair}/{d:%Y-%m-%d}.npz"
            if os.path.exists(f):
                z = np.load(f); ts.append(z["t"]); ps.append(z["p"].astype(np.float64)); break
        else:
            return None
    t = np.concatenate(ts); p = np.concatenate(ps); o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= t0) & (t <= t1)
    return p[m]


rows = []
for r in Y.itertuples():
    pth = FU.path(idx, r.pair, int(r.t))
    if pth is None:
        continue
    tt, hh, ll, cc = pth; e = float(cc[0]); te = int(tt[0]) + 60_000
    p = ticks(r.pair, te, te + HOLD)
    if p is None or len(p) < 10:
        continue
    net = (p / e - 1) * 100 - FEES                              # the bot's P&L at every print
    res = dict(pair=r.pair, day=r.day, fz=r.fz)
    pk = np.maximum.accumulate(net)                             # running best NET P&L (the bot's peak_pnl)
    pkp = np.concatenate([[net[0]], pk[:-1]])                   # peak of PRIOR prints (a line set by the print itself can't close it)
    i_s3 = (lambda a: a[0] if len(a) else len(p))(np.flatnonzero(net <= -3))
    for tp, sl in FIXED:
        ist = np.flatnonzero(net <= -sl); i_s = ist[0] if len(ist) else len(p)
        itp = np.flatnonzero(net >= tp); i_t = itp[0] if len(itp) else len(p)
        i = min(i_s, i_t)
        res[f"fixed +{tp:g}/−{sl:g}"] = net[i] if i < len(p) else net[-1]
    for a, g in TRAILS:                                         # services.frenzy.frenzy_exit_for: line = peak − give·(1 + peak/100) once peak ≥ arm
        line = np.where(pkp >= a, pkp - g * (1 + pkp / 100.0), -np.inf)
        itr = np.flatnonzero(net <= line); i_t = itr[0] if len(itr) else len(p)
        i = min(i_s3, i_t)
        res[f"trail +{a:g}/{g:g} (−3)"] = net[i] if i < len(p) else net[-1]
    for nm, k, be in BULL:                                      # Bull-Run shape on NET P&L: −3 stop; peak ≥ +1 → (BE lock +0.2) and peak − k·ATR; ladder floors
        line = np.full(len(p), -3.0); on = pkp >= 1.0
        line[on] = np.maximum(line[on], pkp[on] - k * float(r.atr))
        if be:
            line[on] = np.maximum(line[on], 0.2)
        for thr, fl in LADDER:
            line[pkp >= thr] = np.maximum(line[pkp >= thr], fl)
        hit = np.flatnonzero(net <= line); i = hit[0] if len(hit) else len(p)
        res[nm] = net[i] if i < len(p) else net[-1]
    rows.append(res)
T = pd.DataFrame(rows); T.to_csv(f"{S}/frenzy_exit_ticks_botexact.csv", index=False); h1 = T.day < SPLIT; base = "fixed +4/−3"
L = ["# 🎯 FRENZY / WIDE exits on REAL TICKS — the BOT's accounting (levels on net P&L, 0.09 % fees, fill at the crossing print)", "",
     f"{len(T)} trades.", "",
     "| Exit (−3 stop unless stated) | Won | Avg win | Avg loss | Breakeven WR | Avg % | Jan–Apr / May–Sep | vs +4/−3 [95 % by day] | Months better | FRENZY · WIDE | Sum % / yr |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
ORDER = sorted(NAMES, key=lambda c: -T[c].mean())
for c in ORDER:
    x = T[c]; d = x - T[base]; w = x[x > 0].mean(); lo = -x[x <= 0].mean()
    if c == base:
        ci, mb = "–", "–"
    else:
        bd = H.boot(T.assign(net=d), "day", 1500); ci = f"{d.mean():+.3f} [{bd[0]:+.3f}, {bd[1]:+.3f}]"
        mm = d.groupby(T.day.str[:7]).mean(); mb = f"{(mm > 0).sum()} of {len(mm)}"
    L.append(f"| {c} | {(x > 0).mean() * 100:.0f}% | {w:+.2f} | {-lo:+.2f} | {lo / (w + lo) * 100:.0f}% | **{x.mean():+.3f}** | {x[h1].mean():+.3f} / {x[~h1].mean():+.3f} | "
             f"{ci} | {mb} | {T[T.fz][c].mean():+.3f} · {T[~T.fz][c].mean():+.3f} | {x.sum():+.0f} |")
open("reports/FRENZY_EXIT_TICKS_BOTEXACT_2026-10-04.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
