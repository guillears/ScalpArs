#!/usr/bin/env python3
"""📉 EMA200 BREAK SHORT — operator (2026-10-02, the MOVR chart): "as soon as price dropped below the EMA200 it was an obvious short".
The mirror of the staircase swing: after a volume-spike run, SHORT when the run breaks. FROZEN before the year run:
  episode   onset = a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M (none in the prior 24 h);
            the run must have reached ≥ +G % above the pre-spike price at some point before the break
  trigger   a 5m CLOSE below the 5m EMA200, ≥ 4 h after the onset and ≤ 96 h after it, with the previous 12 closes all above the EMA200
            (a real break of a held line, not a chop around it)
  trade     SHORT at the next open · exit on a 5m close back above the EMA200 · hard stop S above entry (inside the bar, first) · 48 h cap
  grid      G ∈ {30, 50} × S ∈ {8 %, none} = 4 cells · all futures pairs, Jan–Sep 2026 · cost 0.11 %, funding not included
  PASS      trade-weighted mean > 0 in both halves ∧ day-clustered 95 % interval above 0. Also: best 5 % removed, first break of the episode only.
Usage: venv/bin/python scripts/ema200_break_short_test.py [PAIR ...]"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import staircase_swing_test as ST  # noqa: E402  (api5m, tq)
sys.argv = _a
OUT = os.path.join(ST.ROOT, "reports", "EMA200_BREAK_SHORT_TEST_2026-10-02.md"); COST = 0.11; SPLIT = "2026-05-01"
GRID = [(30, 8.0), (30, None), (50, 8.0), (50, None)]


def trades(pair, d, G, S, SPAN=200):
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values; n = len(c)
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median(); volx = (q1h / norm).values
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]; ema = pd.Series(c).ewm(span=SPAN, adjust=False).mean().values
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    above12 = pd.Series(c >= ema).rolling(12).min().shift(1).fillna(0).values >= 1          # the 12 closes BEFORE this bar all above
    out = []; last_on = -10**9
    for on in lead:
        on = int(on)
        if t[on] - last_on < 24 * 3600_000:
            last_on = t[on]; continue
        last_on = t[on]; base = c[on - 6]; j = on + 48; k_ep = 0; peak = h[on:on + 48].max() if on + 48 <= n else h[on:].max()
        while j < n - 1 and t[j] - t[on] <= 96 * 3600_000:
            peak = max(peak, h[j])
            if not (c[j] < ema[j] and above12[j] and (peak / base - 1) * 100 >= G):
                j += 1; continue
            e_i = j + 1; ent = o[e_i]; res = None
            for m in range(e_i, min(e_i + 576, n)):
                if S is not None and h[m] >= ent * (1 + S / 100):
                    res = (-S, m); break
                if c[m] > ema[m]:
                    res = ((1 - c[m] / ent) * 100, m); break
            if res is None:
                m = min(e_i + 575, n - 1); res = ((1 - c[m] / ent) * 100, m)
            k_ep += 1
            out.append(dict(pair=pair, onset=int(t[on]), t=int(t[e_i]), r=res[0], bars=res[1] - e_i + 1, k=k_ep, peak_gain=(peak / base - 1) * 100, hrs=(t[e_i] - t[on]) / 3600e3))
            j = res[1] + 1
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            d = ST.api5m(p)
            for G, S in GRID[:2]:
                tr = [x for x in trades(p, d, G, S) if x["t"] >= (time.time() - 6 * 86400) * 1000]
                print(f"{p} · run ≥ +{G}% · stop {S}: " + (" | ".join(f"{pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} → {x['r']:+.1f}% in {x['bars'] * 5 / 60:.1f} h" for x in tr) or "no trade") + f" | sum {sum(x['r'] for x in tr):+.1f}")
        sys.exit(0)
    rows = {g: [] for g in GRID}
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        for g in GRID:
            rows[g] += trades(pair, d, *g)
    L = ["# 📉 EMA200 BREAK SHORT — short the first close below the 5m EMA200 after a volume-spike run, year test", "",
         "% of position at 1×, after 0.11 % costs, funding not included.", "",
         "| Run reached | Stop | trades | episodes | days | won | avg win / loss | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | first break only | median hold | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (G, S), r in rows.items():
        T = pd.DataFrame(r); T["net"] = T.r - COST; T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
        T.to_csv(os.path.join(ST.ROOT, "reports", "backtest_cache", f"ema200_short_{G}_{S}.csv"), index=False)
        a1, a2 = T[T.day < SPLIT].net.mean(), T[T.day >= SPLIT].net.mean(); dm = T.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n)
        lo, hi = dm.mean() - ST.tq(n) * se, dm.mean() + ST.tq(n) * se; s = T.net.sort_values(); tr = s.iloc[:-max(1, int(len(s) * 0.05))].mean(); f1 = T[T.k == 1]
        L.append(f"| +{G} % | {'−' + str(int(S)) + ' %' if S else 'none'} | {len(T):,} | {T.groupby(['pair', 'onset']).ngroups} | {n} | {(T.net > 0).mean() * 100:.0f}% | {T[T.net > 0].net.mean():+.1f} / {T[T.net <= 0].net.mean():+.1f} | "
                 f"{a1:+.2f} / {a2:+.2f} | {dm.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {tr:+.2f} | {len(f1)}× {f1.net.mean():+.2f} | {T.bars.median() * 5 / 60:.1f} h | {'✅' if a1 > 0 and a2 > 0 and lo > 0 else '—'} |")
    T = pd.DataFrame(rows[GRID[0]]); T["net"] = T.r - COST; T["mon"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m")
    L += ["", "+30 % / −8 %, by month: " + " · ".join(f"{m} {len(x)}×{x.net.mean():+.2f}" for m, x in T.groupby("mon")),
          f"+30 % / −8 %, tails: best {T.net.max():+.0f} % · trades ≥ +10 %: {(T.net >= 10).sum()} ({(T.net >= 10).mean() * 100:.1f} %) · stopped at −8 %: {(T.r <= -8).mean() * 100:.0f} %"]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
