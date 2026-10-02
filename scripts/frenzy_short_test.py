#!/usr/bin/env python3
"""🌋 FRENZY SHORTS — the short side of the volume frenzies (operator, 2026-10-01: "test the shorts on these episodes").

PRE-DECLARED before any result. Same frozen frenzy as frenzy_dip_test.py (24 h volume ≥ 100× normal ∧ +30 % / 24 h ∧ 5m ATR ≥ 2 %
∧ ≥ $20M), same 241 spot 1-second episodes (cache only, no fetch), same costs (0.09 % fees + 0.02 % slippage), same PASS bar:
trade-weighted mean > 0 in BOTH halves ∧ day-mean 95 % interval above 0 ∧ first-trade-of-episode mean > 0.
  scalps (1 s path, 30-min limit, stop first inside a second, one position, 1 min pause), SHORT at a minute close inside a frenzy bar:
    ANY minute            → 3.09 down / 1.51 up · 1.09 / 1.11 · 0.59 / 1.11
    BREAKDOWN (≥ 5 % below the 4 h high)  → 3.09 / 1.51
    TOP (within 2 % of the 4 h high)      → 3.09 / 1.51
  swing (5m bars, ALL futures pairs incl. futures-only): SHORT at the first bar the frenzy flag switches OFF, hold 24 h,
    with a 10 % stop and with no stop; one position per pair. Funding NOT included.
Usage: venv/bin/python scripts/frenzy_short_test.py → reports/FRENZY_SHORT_TEST_2026-10-01.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_dip_test as F  # noqa: E402
sys.argv = _a
H = F.H; COST, HOLD = 0.11, 1800; OUT = os.path.join(H.ROOT, "reports", "FRENZY_SHORT_TEST_2026-10-01.md")
SC = {"ANY minute → 3.09 / 1.51": ("any", 3.09, 1.51), "ANY minute → 1.09 / 1.11": ("any", 1.09, 1.11), "ANY minute → 0.59 / 1.11": ("any", 0.59, 1.11),
      "BREAKDOWN (≥5 % below 4 h high) → 3.09 / 1.51": ("dip", 3.09, 1.51), "TOP (within 2 % of 4 h high) → 3.09 / 1.51": ("top", 3.09, 1.51)}


def scalps(pair, d5, t0):
    f = os.path.join(F.RAW, f"{pair}_{t0}.npz")
    if not os.path.exists(f):
        return []
    z = np.load(f); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]
    st = d5.reindex(ts // H.BAR * H.BAR); fz = st.frenzy.values.astype(bool); hi4 = st.hi4.values; out = []
    for vn, (mode, tp, sl) in SC.items():
        free = 0; first = True
        for i in range(59, len(c) - 60, 60):
            if i < free or not fz[i]:
                continue
            top = max(hi4[i], h[max(0, i - 14400):i + 1].max()); below = (1 - c[i] / top) * 100
            if (mode == "dip" and below < 5) or (mode == "top" and below > 2):
                continue
            e = c[i]; Hh, Ll = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD]
            hs = np.nonzero(Hh >= e * (1 + sl / 100))[0]; ht = np.nonzero(Ll <= e * (1 - tp / 100))[0]
            a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
            r, secs = ((1 - c[min(i + HOLD, len(c) - 1)] / e) * 100, HOLD) if a == b == 10**9 else ((-sl, a + 1) if a <= b else (tp, b + 1))
            out.append((pair, f"{pair}:{t0}", int(ts[i]), vn, round(r, 4), first)); first = False; free = i + secs + 60
    return out


def swing(pair, d5):
    fz = d5.frenzy.values; h, c, t = d5.h.values, d5.c.values, d5.index.values; out = []; free = 0
    for i in np.nonzero(fz[:-1] & ~fz[1:])[0] + 1:                      # first bar with the flag off (flag is already shifted → known at its open)
        if i < free or i + 288 >= len(c):
            continue
        e = c[i - 1]; hh = h[i:i + 288]; k = np.nonzero(hh >= e * 1.10)[0]; r0 = (1 - c[i + 287] / e) * 100
        out.append((pair, int(t[i]), "SWING off-flag short, 24 h, 10 % stop", -10.0 if len(k) else r0)); out.append((pair, int(t[i]), "SWING off-flag short, 24 h, no stop", r0))
        out.append((pair, int(t[i]), "  (worst move against, %)", (hh.max() / e - 1) * 100)); free = i + 288
    return out


def stats(g, key):
    g = g.assign(day=pd.to_datetime(g.t, unit="ms").dt.strftime("%Y-%m-%d"), net=g.r - COST)
    h1, h2 = g[g.day < H.SPLIT].net.mean(), g[g.day >= H.SPLIT].net.mean(); dm = g.groupby("day").net.mean(); n = len(dm)
    se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n); lo, hi = dm.mean() - q * se, dm.mean() + q * se
    ft = g[g.first_].net.mean() if "first_" in g else np.nan; ok = h1 > 0 and h2 > 0 and lo > 0 and (np.isnan(ft) or ft > 0)
    return f"| {key} | {len(g):,} | {n} | {(g.net > 0).mean() * 100:.0f}% | {h1:+.3f} / {h2:+.3f} | {dm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {ft:+.3f} | {'✅' if ok else '—'} |"


if __name__ == "__main__":
    rows, sw = [], []
    for f in sorted(glob.glob(os.path.join(H.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d5, eps = F.frame(pair)
        if not eps:
            continue
        sw += swing(pair, d5)
        for t0, _ in eps:
            rows += scalps(pair, d5, t0)
    T = pd.DataFrame(rows, columns=["pair", "eid", "t", "variant", "r", "first_"]); S = pd.DataFrame(sw, columns=["pair", "t", "variant", "r"])
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "frenzy_short_trades.csv"), index=False); S.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "frenzy_short_swing.csv"), index=False)
    L = ["# 🌋 FRENZY SHORTS — the short side of every volume frenzy of the year", "", f"Same frozen frenzy and episodes as the dip test ({T.eid.nunique()} episodes with 1-second data). After 0.11 % costs; funding not included.", "",
         "| Variant | trades | days | won | per trade Jan–Apr / May–Sep | by day [95 %] | first trade of each episode | PASS |", "|---|---|---|---|---|---|---|---|"]
    L += [stats(T[T.variant == v], v) for v in SC]
    for v in ("SWING off-flag short, 24 h, 10 % stop", "SWING off-flag short, 24 h, no stop"):
        g = S[S.variant == v]; L.append(stats(g, f"{v} ({g.pair.nunique()} pairs, all futures)"))
    w = S[S.variant == "  (worst move against, %)"].r; g = S[S.variant == "SWING off-flag short, 24 h, no stop"].r
    L += ["", f"Swing short: median 24 h result {g.median():+.1f} % · worst {g.min():+.0f} % · best {g.max():+.0f} % · moved ≥ 10 % against in {(w >= 10).mean() * 100:.0f} % of trades, ≥ 30 % in {(w >= 30).mean() * 100:.0f} %, ≥ 100 % in {(w >= 100).mean() * 100:.1f} %."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
