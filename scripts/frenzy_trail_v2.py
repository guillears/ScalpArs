#!/usr/bin/env python3
"""🎯 Oct-5 (operator: "re-evaluate deeply the exit for FRENZY and FRENZY-WIDE — what trailing mechanism, and its impact"; trigger: RLC
FRENZY_WIDE took the fixed +3 % at minute 3 and ran to +17.9 %).

Same trades, ticks and accounting as scripts/frenzy_exit_ticks_botexact.py (the Oct-4 study that chose fixed +3/−3): the year's quiet-market
FRENZY first candles (frenzy_gvr_trades.csv, gvr < 1), entry = close of the first minute after the signal, levels on NET P&L (gross − 0.09 %
fees), filled at the print that crosses the line, a line set by a print never closes on that same print, 12 h cap at the last print.

NEW families (pre-declared here, before any result was read) — the Oct-4 grid only had "arm at A, give back g" trails, which lost to the
fixed TP because they gave back on the many small moves. These keep the fixed TP's banked profit and add a runner on top:
  LOCK   peak ≥ A → stop jumps to +L (locked), then trails peak − g (percentage points): A ∈ {3, 4} · L ∈ {2, 2.5, 2.9} · g ∈ {1, 1.5, 2, 3, 4}
  FRAC   peak ≥ 3 → line = max(+L, peak × (1 − f)): keep (1 − f) of the peak · L ∈ {2.5, 2.9} · f ∈ {0.2, 0.3, 0.4, 0.5}
  ATR    peak ≥ 3 → line = max(+2.5, peak − k × entry ATR %) · k ∈ {1, 1.5, 2, 3}
  STEP   peak ≥ 3 → floor = floor(peak) − s (a ladder that rises 1 % per whole % of peak) · s ∈ {0.5, 1.0}
  CAP    LOCK +2.5 / g = 3 but closed by the time cap T after entry · T ∈ {60, 120, 240} min
Baseline = fixed +3/−3 (live). Robustness: Δ with the 5 / 10 biggest-gain trades of each variant removed (a runner rule lives on its tail).
Usage: S=<dir with frenzy_gvr_trades.csv> venv/bin/python scripts/frenzy_trail_v2.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _a, sys.argv = sys.argv, ["x"]
import frenzy_scalp_pattern_search_v2 as P2, frenzy_scalp_followup as FU  # noqa: E402
sys.argv = _a; H = P2.H
S = os.environ["S"]; C = "reports/backtest_cache"; SPLIT = "2026-05-01"; FEES = 0.09; HOLD = 12 * 3600_000
Y = pd.read_csv(f"{S}/frenzy_gvr_trades.csv"); Y = Y[Y.gvr < 1.0].reset_index(drop=True)
idx = FU.episode_index()
V = {"fixed +3/−3 (LIVE)": ("fixed", 3.0)}
for A in (3, 4):
    for L in (2, 2.5, 2.9):
        for g in (1, 1.5, 2, 3, 4):
            V[f"LOCK arm +{A} → +{L:g}, trail {g:g}"] = ("lock", A, L, g)
for L in (2.5, 2.9):
    for f in (0.2, 0.3, 0.4, 0.5):
        V[f"FRAC +3 → +{L:g}, keep {100 - f * 100:.0f}% of peak"] = ("frac", L, f)
for k in (1, 1.5, 2, 3):
    V[f"ATR +3 → +2.5, trail {k:g}×ATR"] = ("atr", k)
for s_ in (0.5, 1.0):
    V[f"STEP +3 → floor(peak) − {s_:g}"] = ("step", s_)
for T in (60, 120, 240):
    V[f"CAP +3 → +2.5 trail 3, max {T} min"] = ("cap", T)
V["fixed +4/−3"] = ("fixed", 4.0); V["fixed +6/−3"] = ("fixed", 6.0)
if os.environ.get("V2B"):   # Oct-5 follow-up (operator: "why 2 and not 1.5 or 1? ATR-based?") — dose-response around the leader
    V = {"fixed +3/−3 (LIVE)": ("fixed", 3.0)}
    for L in (1.5, 2.0, 2.25, 2.5):
        for g in (1.5, 1.75, 2.0, 2.25, 2.5):
            V[f"LOCK arm +3 → +{L:g}, trail {g:g}"] = ("lock", 3, L, g)
    for k in (0.5, 0.75, 1.0, 1.25, 1.5):
        V[f"ATR +3 → +2, trail {k:g}×ATR"] = ("atr2", k)


def ticks(pair, t0, t1):
    ts, ps, tt = [], [], []
    for d in pd.date_range(pd.Timestamp(t0, unit="ms").normalize(), pd.Timestamp(t1, unit="ms").normalize()):
        for base in ("ticks_q", "ticks"):
            f = f"{C}/{base}/{pair}/{d:%Y-%m-%d}.npz"
            if os.path.exists(f):
                z = np.load(f); ts.append(z["t"]); ps.append(z["p"].astype(np.float64)); break
        else:
            return None, None
    t = np.concatenate(ts); p = np.concatenate(ps); o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= t0) & (t <= t1)
    return t[m], p[m]


def run_line(net, line, i_s3, n):
    hit = np.flatnonzero(net <= line); i = hit[0] if len(hit) else n
    i = min(i, i_s3)
    return net[i] if i < n else net[-1], i


rows = []
for r in Y.itertuples():
    pth = FU.path(idx, r.pair, int(r.t))
    if pth is None:
        continue
    tt, hh, ll, cc = pth; e = float(cc[0]); te = int(tt[0]) + 60_000
    t, p = ticks(r.pair, te, te + HOLD)
    if p is None or len(p) < 10:
        continue
    n = len(p); net = (p / e - 1) * 100 - FEES
    pk = np.maximum.accumulate(net); pkp = np.concatenate([[net[0]], pk[:-1]])
    i_s3 = (lambda a: a[0] if len(a) else n)(np.flatnonzero(net <= -3))
    atr = float(r.atr) if np.isfinite(r.atr) and r.atr > 0 else 2.0
    res = dict(pair=r.pair, day=r.day, fz=r.fz, peak=float(pk[-1]))
    for nm, spec in V.items():
        kind = spec[0]
        if kind == "fixed":
            itp = np.flatnonzero(net >= spec[1]); i_t = itp[0] if len(itp) else n
            i = min(i_s3, i_t); res[nm] = net[i] if i < n else net[-1]; continue
        line = np.full(n, -np.inf)
        if kind == "lock":
            _, A, L, g = spec; on = pkp >= A; line[on] = np.maximum(L, pkp[on] - g)
        elif kind == "frac":
            _, L, f = spec; on = pkp >= 3; line[on] = np.maximum(L, pkp[on] * (1 - f))
        elif kind == "atr2":
            k = spec[1]; on = pkp >= 3; line[on] = np.maximum(2.0, pkp[on] - k * atr)
        elif kind == "atr":
            k = spec[1]; on = pkp >= 3; line[on] = np.maximum(2.5, pkp[on] - k * atr)
        elif kind == "step":
            s_ = spec[1]; on = pkp >= 3; line[on] = np.maximum(2.5, np.floor(pkp[on]) - s_)
        elif kind == "cap":
            T = spec[1]; on = pkp >= 3; line[on] = np.maximum(2.5, pkp[on] - 3.0)
            late = np.flatnonzero(t > te + T * 60_000)
            if len(late):
                j = late[0]
                v, i = run_line(net, line, i_s3, n)
                res[nm] = v if i < j else net[j]; continue
        res[nm] = run_line(net, line, i_s3, n)[0]
    rows.append(res)

T = pd.DataFrame(rows); T.to_csv(f"{S}/frenzy_trail_v2{'b' if os.environ.get('V2B') else ''}.csv", index=False); h1 = T.day < SPLIT; base = "fixed +3/−3 (LIVE)"
L = ["# 🎯 FRENZY / WIDE trailing exits v2 — lock-then-run families on REAL TICKS (bot accounting) — 2026-10-05", "",
     f"{len(T)} trades (the Oct-4 study's set). Baseline = fixed +3/−3 (live). Δ CI = day-block bootstrap. 'drop top 5/10' = Δ after removing "
     "each variant's 5 / 10 biggest winning trades (a runner rule that only works on 5 trades is a lottery, not a rule).", "",
     "| Exit (−3 stop) | Won | Avg win | Avg % | Jan–Apr / May–Oct | Δ vs +3/−3 [95 % by day] | months better | FRENZY · WIDE | Δ drop top 5 / top 10 | trades > +6 % |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for c in sorted(V, key=lambda c: -T[c].mean()):
    x = T[c]; d = x - T[base]; w = x[x > 0].mean()
    if c == base:
        ci, mb, rob = "–", "–", "–"
    else:
        bd = H.boot(T.assign(net=d), "day", 1500); ci = f"{d.mean():+.3f} [{bd[0]:+.3f}, {bd[1]:+.3f}]"
        mm = d.groupby(T.day.str[:7]).mean(); mb = f"{(mm > 0).sum()} of {len(mm)}"
        top = x.sort_values(ascending=False).index
        rob = f"{d.drop(top[:5]).mean():+.3f} / {d.drop(top[:10]).mean():+.3f}"
    L.append(f"| {c} | {(x > 0).mean() * 100:.0f}% | {w:+.2f} | **{x.mean():+.3f}** | {x[h1].mean():+.3f} / {x[~h1].mean():+.3f} | {ci} | {mb} | "
             f"{T[T.fz][c].mean():+.3f} · {T[~T.fz][c].mean():+.3f} | {rob} | {(x > 6).sum()} |")
L += ["", f"Peak distribution (best NET P&L inside 12 h): ≥ +3 % {(T.peak >= 3).mean() * 100:.0f}% · ≥ +6 % {(T.peak >= 6).mean() * 100:.0f}% · "
      f"≥ +10 % {(T.peak >= 10).mean() * 100:.0f}% · ≥ +15 % {(T.peak >= 15).mean() * 100:.0f}% of trades."]
open(f"reports/FRENZY_TRAIL_V2{'B' if os.environ.get('V2B') else ''}_2026-10-05.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
