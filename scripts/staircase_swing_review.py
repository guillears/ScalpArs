#!/usr/bin/env python3
"""🔪 Hostile review of the STAIRCASE SWING (scripts/staircase_swing_test.py, cell 100× / −8 %: +0.24 / +0.13 %/trade, CI spans 0, 19 % winners,
top-10 trades carry the year) + the operator's exit question (2026-10-02): "also test different TP and SL — maybe 1 % is too much but 0.5 % has
100 % WR, then we go 0.5 % with 3× the size".
  1 tails      the 10 largest winners listed (verify they are real moves); wins capped at +50 % / +100 %; best 5 % / top-10 removed
  2 funding    Binance funding history while each trade is open (long pays +rate)
  3 liquidity  24 h volume at entry ≥ $20M / ≥ $50M / ≥ $100M
  4 stability  leave-one-month-out; first trade of each episode vs re-entries; day-clustered interval
  5 TP / SL    every state-on entry walked on futures 1-MINUTE bars for 24 h (stop first inside a bar):
               TP ∈ {0.3, 0.5, 1, 2, 3, 5, 10} × SL ∈ {1, 2, 3, 5, 8}; unresolved after 24 h → closed at that price.
               Size does not change the sign: 3× the size on a setup that loses 0.1 %/trade loses 0.3 %.
Usage: venv/bin/python scripts/staircase_swing_review.py → reports/STAIRCASE_SWING_REVIEW_2026-10-02.md"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_swing_review as FS  # noqa: E402  (funding cache)
sys.argv = _a
ROOT = FS.H.ROOT; BC = os.path.join(ROOT, "reports", "backtest_cache"); K5 = os.path.join(BC, "k5m_full"); C1 = os.path.join(BC, "k1m_stair")
OUT = os.path.join(ROOT, "reports", "STAIRCASE_SWING_REVIEW_2026-10-02.md"); COST = 0.11; SPLIT = "2026-05-01"
TPS, SLS = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0), (1.0, 2.0, 3.0, 5.0, 8.0)


def tq(n):
    z = 1.959964; d = max(n - 1, 1); return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)


def bars1m(pair, ms):
    f = os.path.join(C1, f"{pair}_{ms}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["h"], z["l"], z["c"]
    r = None
    for k in range(4):
        try:
            r = json.load(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?symbol={urllib.request.quote(pair)}&interval=1m&startTime={ms}&limit=1440", timeout=25)); break
        except Exception:
            time.sleep(2 + 2 * k)
    r = r or []; h, l, c = (np.array([float(x[i]) for x in r]) for i in (2, 3, 4))
    os.makedirs(C1, exist_ok=True); np.savez_compressed(f, h=h, l=l, c=c); time.sleep(0.06)
    return h, l, c


def ci(T, col):
    dm = T.groupby("day")[col].mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n)
    return f"{T[col].mean():+.2f} (Jan–Apr {T[T.day < SPLIT][col].mean():+.2f} / May–Sep {T[T.day >= SPLIT][col].mean():+.2f} · by day {dm.mean():+.2f} [{dm.mean() - tq(n) * se:+.2f}, {dm.mean() + tq(n) * se:+.2f}])"


if __name__ == "__main__":
    T = pd.read_csv(os.path.join(BC, "staircase_swing_100_8.0.csv")); T["net"] = T.r - COST
    T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d"); T["mon"] = T.day.str[:7]; T["exit_t"] = T.t + T.bars * 300_000
    q24, ent = [], []
    for p, g in T.groupby("pair"):
        d = pd.read_csv(os.path.join(K5, p + ".csv")).drop_duplicates("open_time").sort_values("open_time"); t = d.open_time.values.astype("int64")
        cq = np.concatenate([[0.0], np.cumsum(d.qvol.values)])
        for r in g.itertuples():
            i = int(np.searchsorted(t, r.t)); q24.append((r.Index, cq[i] - cq[max(i - 288, 0)])); ent.append((r.Index, d.o.values[min(i, len(t) - 1)]))
    T["q24"] = pd.Series(dict(q24)); T["entry"] = pd.Series(dict(ent))
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); fund = []
    for p, g in T.groupby("pair"):
        try:
            fr = FS.funding(p, lo, hi)
        except SystemExit:
            fr = None
        for r in g.itertuples():
            fund.append((r.Index, np.nan if fr is None else -fr[(fr.t > r.t) & (fr.t <= r.exit_t)].rate.sum() * 100))
    T["fund"] = pd.Series(dict(fund)); T["netf"] = T.net + T.fund.fillna(0)
    L = ["# 🔪 Hostile review — STAIRCASE SWING (100× volume, −8 % stop, exit on a 5m close below the spike VWAP)", "",
         f"{len(T):,} trades · {T.groupby(['pair', 'onset']).ngroups} episodes · {T.pair.nunique()} pairs · Jan–Sep 2026. % of position at 1×.", "", "## 1 · Tails", ""]
    top = T.nlargest(10, "net")
    L += ["| Pair | Entered (UTC) | already up | result | held | 24 h volume |", "|---|---|---|---|---|---|"]
    L += [f"| {r.pair} | {pd.Timestamp(r.t, unit='ms'):%m-%d %H:%M} | +{r.gain_at_entry:.0f}% | {r.net:+.0f}% | {r.bars * 5 / 60:.0f} h | ${r.q24 / 1e6:.0f}M |" for r in top.itertuples()]
    s = T.net.sort_values(); k = max(1, int(len(s) * 0.05))
    L += ["", f"- as tested: {ci(T, 'net')}", f"- wins capped at +100 %: {T.net.clip(upper=100).mean():+.2f} · capped at +50 %: {T.net.clip(upper=50).mean():+.2f} · capped at +25 %: {T.net.clip(upper=25).mean():+.2f}",
          f"- best 10 trades removed: {s.iloc[:-10].mean():+.2f} · best 5 % ({k}) removed: {s.iloc[:-k].mean():+.2f} · best 1 % removed: {s.iloc[:-max(1, len(s) // 100)].mean():+.2f}",
          f"- trades ≥ +10 %: {(T.net >= 10).sum()} ({(T.net >= 10).mean() * 100:.1f} %) carry {T[T.net >= 10].net.sum():+.0f} pts; all the others {T[T.net < 10].net.sum():+.0f} pts",
          "", "## 2 · Funding", "", f"- funding while long: mean {T.fund.mean():+.3f} %/trade · median {T.fund.median():+.3f} · worst {T.fund.min():+.2f} · missing {int(T.fund.isna().sum())}",
          f"- after funding: {ci(T, 'netf')}", "", "## 3 · Liquidity at entry (after funding)", "", "| 24 h volume | trades | won | per trade |", "|---|---|---|---|"]
    for a, b in ((0, 20e6), (20e6, 50e6), (50e6, 100e6), (100e6, 1e15)):
        g = T[(T.q24 >= a) & (T.q24 < b)]; L.append(f"| ${a / 1e6:.0f}M–{'∞' if b > 1e14 else f'${b / 1e6:.0f}M'} | {len(g)} | {(g.netf > 0).mean() * 100:.0f}% | {ci(g, 'netf') if len(g) > 20 else '–'} |")
    L += ["", "## 4 · Stability (after funding)", "", "By month: " + " · ".join(f"{m} {len(g)}×{g.netf.mean():+.2f} (without it {T[T.mon != m].netf.mean():+.2f})" for m, g in T.groupby("mon")),
          "By trade number in the episode: " + " · ".join(f"#{n} {len(g)}×{g.netf.mean():+.2f}" for n, g in T.groupby(T.n_in_ep.clip(upper=4))),
          "By how far the pair had already run at entry: " + " · ".join(f"+{a}…{b if b < 9e5 else '∞'}% {len(g)}×{g.netf.mean():+.2f}" for a, b in ((0, 15), (15, 30), (30, 60), (60, 10**6)) for g in [T[(T.gain_at_entry >= a) & (T.gain_at_entry < b)]]),
          f"Streaks: longest losing run {max((sum(1 for _ in grp) for key, grp in __import__('itertools').groupby(T.sort_values('t').netf < 0) if key), default=0)} trades · "
          f"worst running drawdown {(T.sort_values('t').netf.cumsum() - T.sort_values('t').netf.cumsum().cummax()).min():.0f} pts at 1×."]
    # 5 TP / SL grid on 1m bars
    res = {(tp, sl): [] for tp in TPS for sl in SLS}; ok = []
    for n_, r in enumerate(T.itertuples()):
        h, l, c = bars1m(r.pair, int(r.t))
        if len(c) < 60:
            ok.append(False); continue
        ok.append(True); e = r.entry if r.entry and abs(c[0] / r.entry - 1) < 0.05 else c[0]
        up = np.maximum.accumulate(h) / e - 1; dn = np.minimum.accumulate(l) / e - 1
        for tp in TPS:
            a = int(np.argmax(up >= tp / 100)) if (up >= tp / 100).any() else 10**9
            for sl in SLS:
                b = int(np.argmax(dn <= -sl / 100)) if (dn <= -sl / 100).any() else 10**9
                res[(tp, sl)].append((c[-1] / e - 1) * 100 if a == b == 10**9 else (-sl if b <= a else tp))
        if n_ % 300 == 0:
            print(n_, flush=True)
    G = T[np.array(ok)].copy()
    L += ["", f"## 5 · Fixed target / stop from the same {len(G):,} entries (1-minute bars, 24 h, stop first, after 0.11 %; funding not added)", "",
          "Each cell: hit the target % (needed to break even %) · per trade Jan–Apr / May–Sep", "", "| Target ↓ / Stop → | " + " | ".join(f"−{s:g} %" for s in SLS) + " |", "|---|" + "---|" * len(SLS)]
    best = []
    for tp in TPS:
        row = f"| +{tp:g} % |"
        for sl in SLS:
            x = np.array(res[(tp, sl)]) - COST; G["_x"] = x; a1, a2 = G[G.day < SPLIT]._x.mean(), G[G.day >= SPLIT]._x.mean()
            hit = (np.array(res[(tp, sl)]) == tp).mean() * 100; need = (sl + COST) / (tp + sl) * 100
            row += f" {hit:.0f}% ({need:.0f}%) · {a1:+.2f} / {a2:+.2f} |"; best.append((min(a1, a2), tp, sl, hit, need))
        L.append(row)
    b = max(best); L += ["", f"Best cell by its weaker half: +{b[1]:g} % / −{b[2]:g} % ({b[3]:.0f} % hit vs {b[4]:.0f} % needed, weaker half {b[0]:+.2f} %/trade) — the best of {len(best)} cells, so read it as a ceiling.",
                         f"No stop at all, +0.5 % target: reached within 24 h by {(np.array([(np.array(res[(0.5, 8.0)]) == 0.5)]).mean()) * 100:.0f} % of entries when the stop is 8 % (the rest lose 8 % or sit open)."]
    G.drop(columns="_x").to_csv(os.path.join(BC, "staircase_swing_review_rows.csv"), index=False)
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
