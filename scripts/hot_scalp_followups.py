#!/usr/bin/env python3
"""🔥 Follow-ups on the hot-state 1-second cache (reports/backtest_cache/k1s_hot, built by hot_scalp_backtest.py) — the operator's
two ideas from his 40 manual MOVR trades (2026-10-01), tested on every cached episode:

  P  PAUSE, DON'T CHASE (his winners: side = the 5-minute direction, entry on a seconds-long pause; his losers: entries on a burst)
     every 5 s inside an episode window while the pair's 5m ATR ≥ 2 %: side = sign of the 5-minute move when |move| ≥ 1.5 %;
     bucket by the last-15-s move IN that direction: pullback ≤ −0.3 % · pause (−0.3, 0] · drift (0, +0.3) · burst ≥ +0.3 %
     exits (gross): X1 +0.59 / −1.11 · X2 +0.59 / −1.11 out at market after 45 s · X3 +0.59 / −0.80 out after 60 s
  D  WAIT FOR THE DIP ("identify the trade, wait the dip, make the buy")
     at HOT seconds 30 s apart: a resting BUY at signal price − d % (d = 0.5 / 1.0 / 1.5 / 2.0), valid 5 min; from the fill:
     +0.59 % target, stop −1.2 % or −2.0 %, out after 30 min
  costs  market entry: 0.09 % fees + 0.02 % measured slippage (master batch + tape replay); resting entry: 0.065 % fees + 0.01 %;
         a stress column adds 0.10 %
  units  per-episode means → UTC-day means → t-interval (EPISODE-weighted; the target-first column is signal-weighted; the first 5
         minutes of a window are not scored); halves Jan–Apr / May–Sep. Windows only cover LONG-side hot episodes, so
         the SHORT side of P is what happens inside those windows after a pump turns (stated in the report).
Usage: venv/bin/python scripts/hot_scalp_followups.py → reports/HOT_SCALP_FOLLOWUPS_2026-10-01.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import hot_scalp_backtest as H  # noqa: E402
sys.argv = _a
OUT = os.path.join(H.ROOT, "reports", "HOT_SCALP_FOLLOWUPS_2026-10-01.md")
C_MKT, C_LIM, STRESS = 0.11, 0.075, 0.10
BUCKETS = ["pullback ≤ −0.3 %", "pause (−0.3, 0]", "drift (0, +0.3)", "burst ≥ +0.3 %"]


def trade(h, l, c, i, sg, tp, sl, hold, e=None):
    e = c[i] if e is None else e; Hh, Ll = h[i + 1:i + 1 + hold], l[i + 1:i + 1 + hold]
    if not len(Hh):
        return 0.0, 0
    if sg > 0:
        hs = np.nonzero(Ll <= e * (1 - sl / 100))[0]; ht = np.nonzero(Hh >= e * (1 + tp / 100))[0]
    else:
        hs = np.nonzero(Hh >= e * (1 + sl / 100))[0]; ht = np.nonzero(Ll <= e * (1 - tp / 100))[0]
    a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
    if a == b == 10**9:
        return sg * (c[min(i + hold, len(c) - 1)] / e - 1) * 100, 0
    return (-sl, -1) if a <= b else (tp, 1)


def episode(pair, d5, f):
    z = np.load(f); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]
    st = d5.reindex(ts // H.BAR * H.BAR); atr, rsi, ema5, q24 = st.atr.values, st.rsi.values, st.ema5.values, st.q24.values
    hot = (atr >= 2) & (rsi >= 70) & (c >= ema5 * 1.03) & (q24 >= 20e6)
    P, D = [], []
    for i in range(305, len(c) - 120, 5):
        if not atr[i] >= 2:
            continue
        r300 = (c[i] / c[i - 300] - 1) * 100
        if abs(r300) < 1.5:
            continue
        sg = 1 if r300 > 0 else -1; r15 = sg * (c[i] / c[i - 15] - 1) * 100
        b = 0 if r15 <= -0.3 else 1 if r15 <= 0 else 2 if r15 < 0.3 else 3
        x1, w1 = trade(h, l, c, i, sg, 0.59, 1.11, 1800); x2, _ = trade(h, l, c, i, sg, 0.59, 1.11, 45); x3, _ = trade(h, l, c, i, sg, 0.59, 0.80, 60)
        P.append((sg, b, x1, w1 == 1, x2, x3))
    hi = np.nonzero(hot)[0]; last = -10**9
    for i in hi:
        if i - last < 30 or i > len(c) - 400:
            continue
        last = i
        for d in (0.5, 1.0, 1.5, 2.0):
            lim = c[i] * (1 - d / 100); k = np.nonzero(l[i + 1:i + 301] <= lim)[0]
            if not len(k):
                D.append((d, False, 0.0, False, 0.0)); continue
            j = i + 1 + int(k[0])
            a, wa = trade(h, l, c, j, 1, 0.59, 1.2, 1800, e=lim); b_, _ = trade(h, l, c, j, 1, 0.59, 2.0, 1800, e=lim)
            D.append((d, True, a, wa == 1, b_))
    return P, D


def dayci(x):
    """x: Series of episode values indexed by day → (mean of day means, lo, hi, n days)."""
    dm = x.groupby(level=0).mean(); n = len(dm)
    if n < 3:
        return dm.mean() if n else np.nan, np.nan, np.nan, n
    se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n)
    return dm.mean(), dm.mean() - q * se, dm.mean() + q * se, n


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(H.RAW, "*.npz"))); cache = {}; PR, DR = [], []
    for n_, f in enumerate(files):
        pair, t0 = os.path.basename(f)[:-4].rsplit("_", 1)
        if pair not in cache:
            cache = {pair: H.frame5(pair)[0]}
        P, D = episode(pair, cache[pair], f); day = pd.Timestamp(int(t0), unit="ms").strftime("%Y-%m-%d")
        if P:
            p = pd.DataFrame(P, columns=["sg", "b", "x1", "w1", "x2", "x3"])
            for (sg, b), g in p.groupby(["sg", "b"]):
                PR.append((pair, day, sg, b, len(g), g.x1.mean(), g.w1.mean(), g.x2.mean(), g.x3.mean()))
        if D:
            dd = pd.DataFrame(D, columns=["d", "fill", "a", "wa", "b"])
            for d, g in dd.groupby("d"):
                fl = g[g.fill]
                DR.append((pair, day, d, len(g), len(fl), fl.a.mean() if len(fl) else np.nan, fl.wa.mean() if len(fl) else np.nan, fl.b.mean() if len(fl) else np.nan))
        if n_ % 200 == 0:
            print(f"{n_}/{len(files)}", flush=True)
    P = pd.DataFrame(PR, columns=["pair", "day", "sg", "b", "n", "x1", "w1", "x2", "x3"]); D = pd.DataFrame(DR, columns=["pair", "day", "d", "n", "fills", "a", "wa", "b"])
    L = ["# 🔥 Hot-state follow-ups — pause-don't-chase and wait-for-the-dip, all cached episodes", "",
         f"{len(files):,} episodes with 1-second data. Net = after {C_MKT:.2f} % (market entry: fees + measured slippage) or {C_LIM:.3f} % (resting entry); "
         "the stress column subtracts a further 0.10 %. Ranges = 95 % interval on UTC-day means of episode averages.", "",
         "## P. Trade with the 5-minute move — does the last 15 seconds matter?", "",
         "| Side | Last 15 s (in the trade's direction) | episodes | days | signals | target first | net % / trade, +0.59 / −1.11 [95 % by day] | stress | Jan–Apr / May–Sep | out after 45 s | +0.59 / −0.80, 60 s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sg, sn in ((1, "LONG"), (-1, "SHORT")):
        for b, bn in enumerate(BUCKETS):
            g = P[(P.sg == sg) & (P.b == b)]
            if len(g) < 20:
                continue
            s = g.set_index("day"); m, lo, hi, nd = dayci(s.x1 - C_MKT)
            h1 = (g[g.day < H.SPLIT].groupby("day").x1.mean() - C_MKT).mean(); h2 = (g[g.day >= H.SPLIT].groupby("day").x1.mean() - C_MKT).mean()
            L.append(f"| {sn} | {bn} | {len(g):,} | {nd} | {int(g.n.sum()):,} | {np.average(g.w1, weights=g.n) * 100:.0f}% | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {m - STRESS:+.3f} | "
                     f"{h1:+.3f} / {h2:+.3f} | {dayci(s.x2 - C_MKT)[0]:+.3f} | {dayci(s.x3 - C_MKT)[0]:+.3f} |")
    L += ["", "## D. Wait for the dip — a resting buy below the hot signal", "",
          "| Dip | signals | filled | episodes with fills | days | target first (stop −1.2) | net % / trade, stop −1.2 [95 % by day] | stress | Jan–Apr / May–Sep | stop −2.0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for d in (0.5, 1.0, 1.5, 2.0):
        g = D[(D.d == d)]; f = g[g.fills > 0]
        if len(f) < 20:
            continue
        s = f.set_index("day"); m, lo, hi, nd = dayci(s.a - C_LIM)
        h1 = (f[f.day < H.SPLIT].groupby("day").a.mean() - C_LIM).mean(); h2 = (f[f.day >= H.SPLIT].groupby("day").a.mean() - C_LIM).mean()
        L.append(f"| −{d:.1f} % | {int(g.n.sum()):,} | {g.fills.sum() / g.n.sum() * 100:.0f}% | {len(f):,} | {nd} | {np.average(f.wa, weights=f.fills) * 100:.0f}% | "
                 f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {m - STRESS:+.3f} | {h1:+.3f} / {h2:+.3f} | {dayci(s.b - C_LIM)[0]:+.3f} |")
    L += ["", "Breakeven target-first rate at these costs: +0.59 / −1.11 → 72 % (market entry) · +0.59 / −1.2 → 71 % (resting entry)."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
