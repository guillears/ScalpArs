#!/usr/bin/env python3
"""🪜 STAIRCASE SWING — operator (2026-10-02): "we do not need thousands of trades — 2 trades of MOVR on Sep-30, 2 on Oct-1 and 2 of SAND
today would have been game changers". So: FEW, LONG trades that ride the staircase, not scalps. FROZEN before the year run:
  onset    a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M, with no such bar in the prior 24 h;
           anchored VWAP = Σ(typical price × volume) / Σ volume from that bar; the episode lasts until 24 h pass without the state
  STATE    ≥ 2 h after the onset ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥ V× the normal hour
  entry    LONG at the open of the first bar after the state turns on (and again each time it turns back on after an exit)
  exit     the first 5m CLOSE below the anchored VWAP (the staircase is broken) · hard stop S below entry (inside the bar, checked
           first) · 48 h cap.  cost 0.11 %; funding NOT included (multi-hour holds — it can cost or pay ~0.01–0.5 % per 8 h in a frenzy)
  grid     V ∈ {100, 50} × S ∈ {8 %, none} = 4 cells · all futures pairs in the 5m year cache (Jan–Sep 28; MOVR's and SAND's moves are after it)
  PASS     trade-weighted mean > 0 in both halves ∧ day-clustered 95 % interval above 0. Also shown: best 5 % removed, episodes positive.
Usage: venv/bin/python scripts/staircase_swing_test.py [PAIR ...]  (with pairs: the same rule on fresh API data, as an illustration)"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); K5 = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full")
OUT = os.path.join(ROOT, "reports", "STAIRCASE_SWING_TEST_2026-10-02.md"); BAR = 300_000; COST = 0.11; SPLIT = "2026-05-01"
GRID = [(100, 8.0), (100, None), (50, 8.0), (50, None)]


def tq(n):
    z = 1.959964; d = max(n - 1, 1); return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)


def trades(pair, d, V, S):
    t = d.open_time.values.astype("int64"); o, h, l, c, q = d.o.values, d.h.values, d.l.values, d.c.values, d.qvol.values
    q1h = pd.Series(q).rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30, min_periods=288 * 10).median()
    volx = (q1h / norm).values; r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]
    lead = np.nonzero((r30 >= 5) & (volx >= 20) & (q1h.values >= 2e6))[0]
    out = []; i = 0; n = len(c); k = 0
    while k < len(lead):
        on = lead[k]
        if on < i:
            k += 1; continue
        pv = (h[on:] + l[on:] + c[on:]) / 3 * q[on:]; vw = np.cumsum(pv) / np.maximum(np.cumsum(q[on:]), 1e-12)     # vw[j] = VWAP through bar on+j
        j = on + 24; last_state = on; ntr = 0; end = on
        while j < n - 1:
            if (t[j] - t[last_state]) > 24 * 3600_000:
                break
            jj = j - on
            ok = (c[j - 11:j + 1] >= vw[jj - 11:jj + 1]).all() and volx[j] >= V
            if not ok:
                j += 1; continue
            last_state = j; e_i = j + 1; ent = o[e_i]; res = None                      # state known at the close of bar j → enter at the next open
            for m in range(e_i, min(e_i + 576, n)):
                if S is not None and l[m] <= ent * (1 - S / 100):
                    res = (-S, m); break
                if c[m] < vw[m - on]:
                    res = ((c[m] / ent - 1) * 100, m); break
            if res is None:
                m = min(e_i + 575, n - 1); res = ((c[m] / ent - 1) * 100, m)
            ntr += 1
            out.append(dict(pair=pair, onset=int(t[on]), t=int(t[e_i]), r=res[0], bars=res[1] - e_i + 1, n_in_ep=ntr, gain_at_entry=(ent / c[on - 6] - 1) * 100 if on >= 6 else np.nan))
            j = res[1] + 1; last_state = res[1]; end = res[1]
        i = max(j, end) + 1
        while k < len(lead) and lead[k] < i:
            k += 1
    return out


def api5m(pair, days=45):
    out = []; cur = int((time.time() - days * 86400) * 1000)
    while True:
        r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={cur}&limit=1500", timeout=25))
        if not r:
            break
        out += r; cur = int(r[-1][0]) + 1; time.sleep(0.05)
        if len(r) < 1500:
            break
    d = pd.DataFrame(out).iloc[:, [0, 1, 2, 3, 4, 7]].astype(float); d.columns = ["open_time", "o", "h", "l", "c", "qvol"]
    return d.drop_duplicates("open_time").iloc[:-1]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            d = api5m(p)
            for V, S in GRID[:2]:
                tr = [x for x in trades(p, d, V, S) if x["t"] >= (time.time() - 5 * 86400) * 1000]
                print(f"\n{p} · vol ≥ {V}× · stop {S}: " + (" | ".join(f"{pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} entered at +{x['gain_at_entry']:.0f}% → {x['r']:+.1f}% in {x['bars'] * 5 / 60:.1f} h" for x in tr) or "no trade"))
        sys.exit(0)
    rows = {g: [] for g in GRID}
    for f in sorted(glob.glob(os.path.join(K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        for g in GRID:
            rows[g] += trades(pair, d, *g)
    L = ["# 🪜 STAIRCASE SWING — ride the anchored VWAP after a volume spike (few, long trades), year test", "",
         "% of position at 1×, after 0.11 % costs, funding not included.", "",
         "| Volume ≥ | Hard stop | trades | episodes | days | won | avg win / avg loss | per trade Jan–Apr / May–Sep | by day [95 %] | best 5 % removed | episodes positive | median hold | PASS |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (V, S), r in rows.items():
        T = pd.DataFrame(r); T["net"] = T.r - COST; T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
        T.to_csv(os.path.join(ROOT, "reports", "backtest_cache", f"staircase_swing_{V}_{S}.csv"), index=False)
        a1, a2 = T[T.day < SPLIT].net.mean(), T[T.day >= SPLIT].net.mean(); dm = T.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n)
        lo, hi = dm.mean() - tq(n) * se, dm.mean() + tq(n) * se; s = T.net.sort_values(); tr = s.iloc[:-max(1, int(len(s) * 0.05))].mean(); ep = T.groupby(["pair", "onset"]).net.sum()
        L.append(f"| {V}× | {'−' + str(int(S)) + ' %' if S else 'none'} | {len(T):,} | {len(ep)} | {n} | {(T.net > 0).mean() * 100:.0f}% | {T[T.net > 0].net.mean():+.1f} / {T[T.net <= 0].net.mean():+.1f} | {a1:+.2f} / {a2:+.2f} | "
                 f"{dm.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {tr:+.2f} | {(ep > 0).mean() * 100:.0f}% | {T.bars.median() * 5 / 60:.1f} h | {'✅' if a1 > 0 and a2 > 0 and lo > 0 else '—'} |")
    T = pd.DataFrame(rows[GRID[0]]); T["net"] = T.r - COST; T["mon"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m")
    L += ["", "100× / −8 %, by month: " + " · ".join(f"{m} {len(x)}×{x.net.mean():+.2f}" for m, x in T.groupby("mon")),
          "100× / −8 %, by trade number inside the episode: " + " · ".join(f"#{k} {len(x)}×{x.net.mean():+.2f}" for k, x in T.groupby(T.n_in_ep.clip(upper=4))),
          f"100× / −8 %, tails: best {T.net.max():+.0f} % · worst {T.net.min():+.0f} % · trades ≥ +10 %: {(T.net >= 10).sum()} ({(T.net >= 10).mean() * 100:.1f} %) · top 10 trades carry {T.net.nlargest(10).sum():+.0f} of {T.net.sum():+.0f} pts"]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
