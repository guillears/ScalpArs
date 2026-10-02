#!/usr/bin/env python3
"""🪜 STAIRCASE v2 — the MOVR (Sep-30/Oct-1) and SAND (Oct-2) shape, tightened from what separated them from ARK / NOM / ALICE / GTC /
SCR that week (9 pairs, hour by hour): the two winners (1) stayed ABOVE the volume-weighted average price anchored at the first
spike (MOVR 83 % of hourly closes, SAND 100 %; the faders 4–33 %) and (2) kept trading ≥ ~100× their normal hourly volume
(median 154× / 156×; the faders decayed to 9–56×). Both are knowable at entry. FROZEN before the year run (the year cache ends
2026-09-28 — neither MOVR's nor SAND's move is in it → out of sample):
  onset     first 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M (searched from 24 h before
            each cached frenzy episode); anchored VWAP = Σ(typical price × volume) / Σ volume from that bar
  STATE     ≥ 2 h after the onset ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥ 100× the normal hour
            (closed bars only)
  entries   ANY minute in the state · PULLBACK = in the state and price ≤ 5 % above the VWAP
  exits     +1.0 / −3.0 (60 min, the operator's manual setup) · +3.09 / −1.51 (30 min); 1-second path, stop first; cost 0.11 %
  reference the same exits at minutes of the same episodes that are NOT in the state
  PASS      trade-weighted mean > 0 in both halves ∧ day-mean 95 % interval above 0 ∧ first-trade-of-episode mean > 0
Usage: venv/bin/python scripts/staircase_test.py → reports/STAIRCASE_TEST_2026-10-02.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_dip_test as F  # noqa: E402
sys.argv = _a
H = F.H; COST = 0.11; OUT = os.path.join(H.ROOT, "reports", "STAIRCASE_TEST_2026-10-02.md")
EX = {"+1.0 / −3.0, 60 min": (1.0, 3.0, 3600), "+3.09 / −1.51, 30 min": (3.09, 1.51, 1800)}


def state_frame(d, t0, t1):
    """5m frame → per-bar (known during the bar): in_state, vwap, hours since onset. d = raw 5m frame indexed by open_time."""
    q1h = d.qvol.rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); r30 = (d.c / d.c.shift(6) - 1) * 100
    lead = (r30 >= 5) & (q1h / norm >= 20) & (q1h >= 2e6)
    w = lead.loc[t0 - 24 * 3600_000:t1]; on = int(w.index[w.values][0]) if w.values.any() else int(t0)
    e = d.loc[on:t1 + 2 * 3600_000]
    vwap = ((e.h + e.l + e.c) / 3 * e.qvol).cumsum() / e.qvol.cumsum()
    ok = ((e.c >= vwap).rolling(12).min() >= 1) & ((q1h.loc[e.index] / norm.loc[e.index]) >= 100) & ((e.index.values - on) >= 2 * 3600_000 - H.BAR)
    return ok.shift(1).fillna(False).astype(bool), vwap.shift(1), on


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(os.path.join(H.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d5, eps = F.frame(pair)
        if not eps:
            continue
        for t0, t1 in eps:
            fn = os.path.join(F.RAW, f"{pair}_{t0}.npz")
            if not os.path.exists(fn):
                continue
            z = np.load(fn); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]; b = ts // H.BAR * H.BAR
            ok, vw, on = state_frame(d5, t0, t1); st = ok.reindex(b).fillna(False).values; vv = vw.reindex(b).values
            for grp in ("ANY in state", "PULLBACK in state", "not in state"):
                for en, (tp, sl, hold) in EX.items():
                    free = 0; first = True
                    for i in range(59, len(c) - 60, 60):
                        if i < free:
                            continue
                        ins = bool(st[i]); pb = ins and vv[i] == vv[i] and c[i] <= vv[i] * 1.05
                        if (grp == "ANY in state" and not ins) or (grp == "PULLBACK in state" and not pb) or (grp == "not in state" and ins):
                            continue
                        e = c[i]; Hh, Ll = h[i + 1:i + 1 + hold], l[i + 1:i + 1 + hold]
                        a = np.nonzero(Ll <= e * (1 - sl / 100))[0]; bb = np.nonzero(Hh >= e * (1 + tp / 100))[0]
                        a = a[0] if len(a) else 10**9; bb = bb[0] if len(bb) else 10**9
                        r, secs = ((c[min(i + hold, len(c) - 1)] / e - 1) * 100, hold) if a == bb == 10**9 else ((-sl, a + 1) if a <= bb else (tp, bb + 1))
                        rows.append((pair, f"{pair}:{t0}", int(ts[i]), grp, en, r, first)); first = False; free = i + secs + 60
    T = pd.DataFrame(rows, columns=["pair", "eid", "t", "grp", "ex", "r", "first_"]); T["net"] = T.r - COST; T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "staircase_trades.csv"), index=False)
    L = ["# 🪜 STAIRCASE v2 — above the anchored VWAP for an hour with volume still ≥ 100× normal (the MOVR / SAND shape), year test", "",
         "| Entry | Exit | trades | episodes | days | won | per trade Jan–Apr / May–Sep | by day [95 %] | first trade of each episode | PASS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for grp in ("ANY in state", "PULLBACK in state", "not in state"):
        for en in EX:
            g = T[(T.grp == grp) & (T.ex == en)]
            if len(g) < 20:
                L.append(f"| {grp} | {en} | {len(g)} | – | – | – | – | – | – | — |"); continue
            a1, a2 = g[g.day < H.SPLIT].net.mean(), g[g.day >= H.SPLIT].net.mean(); dm = g.groupby("day").net.mean(); n = len(dm)
            se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n); lo, hi = dm.mean() - q * se, dm.mean() + q * se; ft = g[g.first_].net.mean()
            L.append(f"| {grp} | {en} | {len(g):,} | {g.eid.nunique()} | {n} | {(g.net > 0).mean() * 100:.0f}% | {a1:+.3f} / {a2:+.3f} | {dm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {ft:+.3f} | {'✅' if a1 > 0 and a2 > 0 and lo > 0 and ft > 0 else '—'} |")
    g = T[(T.grp == "ANY in state") & (T.ex == list(EX)[0])]
    if len(g):
        e = g.groupby("eid").net.sum(); p = g.groupby("pair").net.sum()
        L += ["", f"ANY in state, +1.0 / −3.0, by episode: {(e > 0).mean() * 100:.0f} % of {len(e)} positive · best {e.max():+.1f} · worst {e.min():+.1f} · median {e.median():+.1f}. "
              f"By pair: {(p > 0).mean() * 100:.0f} % of {len(p)} positive; best 3 pairs {p.nlargest(3).sum():+.0f} of {p.sum():+.0f}."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
