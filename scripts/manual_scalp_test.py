#!/usr/bin/env python3
"""🖐 Is the operator's MOVR hand-scalp an edge or luck? (2026-10-01: 8 manual MOVR longs in 6 minutes, 7 won, +$1,030 at 20–50×.)

The trades: LONG into a vertical thrust (pair RSI ~79, price ~5 % above its 5m EMA5, 5m ATR ~3.4 %), take-profit +0.5 % net,
stop −1.2 % net, held seconds. 1-minute candles cannot resolve this (both levels sit inside one minute) → 1-SECOND klines
(Binance SPOT public API, the futures contract tracks it; cached in reports/backtest_cache/k1s/).

PRE-DECLARED (before any result):
  entries   LONG every 15 s (the 1s close)
  states    HOT   = 5m ATR(14) ≥ 2 % ∧ price ≥ 3 % above the 5m EMA5 ∧ 5m RSI(14) ≥ 70   (the operator's entries, from CLOSED 5m bars)
            ATR   = 5m ATR(14) ≥ 2 % (any direction)          ALL = every sampled second
            COLD  = the mirror for SHORTS: ATR ≥ 2 % ∧ price ≤ 3 % below EMA5 ∧ RSI ≤ 30
  exits     gross TP / SL %: A 0.59 / 1.11 (the operator's +0.5 / −1.2 net) · B 1.09 / 1.11 · C 2.09 / 1.11 · D 3.09 / 1.51 ·
            E 0.59 / 0.61 · max hold 30 min then out at the close. Inside one second the STOP is checked first (adverse-first).
  costs     0.09 % fees round trip; a second line adds 0.10 % slippage (paper fills at the exact trigger price; a market order
            in a 3 %-per-5-min tape does not)
  reads     hit rate vs the no-edge rate SL/(TP+SL) (what a driftless price gives) and vs the breakeven rate after costs;
            net % per trade; EPISODES = runs of the state split by ≥ 30 min gaps (entries seconds apart are ONE observation)
Usage: venv/bin/python scripts/manual_scalp_test.py PAIR:START:END ...   (UTC, e.g. MOVRUSDT:2026-09-30T00:00:2026-10-01T16:30)"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "reports", "backtest_cache", "k1s")
OUT = os.path.join(ROOT, "reports", "MANUAL_SCALP_TEST_2026-10-01.md")
EXITS = {"A +0.59 / −1.11 (yours)": (0.59, 1.11), "B +1.09 / −1.11": (1.09, 1.11), "C +2.09 / −1.11": (2.09, 1.11),
         "D +3.09 / −1.51": (3.09, 1.51), "E +0.59 / −0.61": (0.59, 0.61)}
FEE, SLIP, HOLD, STEP = 0.09, 0.10, 1800, 15


def fetch(pair, start, end):
    f = os.path.join(CACHE, f"{pair}_{start}_{end}.csv".replace(":", ""))
    if os.path.exists(f):
        return pd.read_csv(f).set_index("t")
    s, e = int(pd.Timestamp(start).timestamp() * 1000), int(pd.Timestamp(end).timestamp() * 1000); rows = []
    while s < e:
        u = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1s&startTime={s}&endTime={e}&limit=1000"
        for k in range(4):
            try:
                r = json.load(urllib.request.urlopen(u, timeout=20)); break
            except Exception as ex:
                if k == 3:
                    raise SystemExit(f"fetch failed for {pair}: {ex}")
                time.sleep(2)
        if not r:
            break
        rows += [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])) for x in r]
        s = int(r[-1][0]) + 1000; time.sleep(0.12)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "q"]).drop_duplicates("t")
    os.makedirs(CACHE, exist_ok=True); d.to_csv(f, index=False)
    return d.set_index("t")


def states(d):
    """Per 1s row: 5m ATR %, stretch vs EMA5, RSI — all from the last CLOSED 5m bar (no look-ahead) + the live price."""
    ts = pd.to_datetime(d.index, unit="ms"); x = d.set_index(ts)
    b = x.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    pc = b.c.shift(1); tr = np.maximum(b.h - b.l, np.maximum((b.h - pc).abs(), (b.l - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean() / b.c * 100
    ema5 = b.c.ewm(span=5, adjust=False).mean()
    dl = b.c.diff(); up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    S = pd.DataFrame({"atr": atr, "ema5": ema5, "rsi": rsi}); S.index = S.index + pd.Timedelta(minutes=5)   # usable once the bar CLOSED
    S = S.iloc[20:]                                                    # indicator warm-up
    S = S.reindex(ts, method="ffill")
    S["stretch"] = (x.c.values / S.ema5.values - 1) * 100
    return S.reset_index(drop=True)


def walk(h, l, c, i, tp, sl, side):
    """One trade from the close of second i. Returns gross % (stop checked first inside a second)."""
    e = c[i]; j1 = min(i + 1 + HOLD, len(c))
    if side > 0:
        hs = np.nonzero(l[i + 1:j1] <= e * (1 - sl / 100))[0]; ht = np.nonzero(h[i + 1:j1] >= e * (1 + tp / 100))[0]
    else:
        hs = np.nonzero(h[i + 1:j1] >= e * (1 + sl / 100))[0]; ht = np.nonzero(l[i + 1:j1] <= e * (1 - tp / 100))[0]
    a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
    if a == b == 10**9:
        return side * (c[j1 - 1] / e - 1) * 100, "TIME"
    return (-sl, "STOP") if a <= b else (tp, "TP")


def episodes(idx):
    idx = np.asarray(idx); return 0 if not len(idx) else int(1 + (np.diff(idx) > 1800).sum())


if __name__ == "__main__":
    specs = [a.split(":", 1) for a in sys.argv[1:]] or [["MOVRUSDT", "2026-09-30T00:00:2026-10-01T16:30"]]
    L = ["# 🖐 Hand-scalp test on 1-second data — edge or luck?", "",
         "LONG every 15 s; HOT = the operator's entry state (5m ATR ≥ 2 %, price ≥ 3 % above EMA5, RSI ≥ 70). "
         "No-edge rate = SL ÷ (TP + SL): what a price with no drift gives. Breakeven = the hit rate needed after 0.09 % fees. "
         "Episodes = separate runs of the state (≥ 30 min apart) — the real number of observations.", ""]
    for pair, rng in specs:
        start, end = rng[:16], rng[17:]
        d = fetch(pair, start, end); S = states(d)
        h, l, c = d.h.values, d.l.values, d.c.values
        ok = S.atr.notna().values; grid = np.arange(0, len(c) - 120, STEP)                 # a trade cut by the data end exits at the last close; grid = grid[ok[grid]]
        m = {"HOT (long)": (S.atr.values >= 2) & (S.stretch.values >= 3) & (S.rsi.values >= 70),
             "ATR ≥ 2 % (long)": S.atr.values >= 2, "ALL (long)": np.ones(len(c), bool),
             "COLD (short)": (S.atr.values >= 2) & (S.stretch.values <= -3) & (S.rsi.values <= 30)}
        L += [f"## {pair} · {start} → {end} UTC · {len(c):,} seconds · price {c[0]:.4g} → {c[-1]:.4g}", "",
              "| State | Exit (gross TP / SL) | entries | episodes | hit rate | no-edge rate | breakeven | net % per trade | with 0.10 % slippage | TP / stop / time % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sn, mask in m.items():
            g = grid[mask[grid]]; side = -1 if "short" in sn else 1
            for en, (tp, sl) in EXITS.items():
                if not len(g):
                    L.append(f"| {sn} | {en} | 0 | 0 | – | – | – | – | – | – |"); continue
                res = [walk(h, l, c, int(i), tp, sl, side) for i in g]
                r = np.array([x[0] for x in res]); how = np.array([x[1] for x in res])
                hit = (how == "TP").mean() * 100; be = (sl + FEE) / (tp + sl) * 100
                L.append(f"| {sn} | {en} | {len(g):,} | {episodes(g)} | {hit:.0f}% | {sl / (tp + sl) * 100:.0f}% | {be:.0f}% | {r.mean() - FEE:+.3f} | "
                         f"{r.mean() - FEE - SLIP:+.3f} | {hit:.0f} / {(how == 'STOP').mean() * 100:.0f} / {(how == 'TIME').mean() * 100:.0f} |")
        # the HOT state hour by hour, exit A — is it one lucky thrust?
        g = grid[m["HOT (long)"][grid]]
        if len(g):
            res = np.array([walk(h, l, c, int(i), *EXITS["A +0.59 / −1.11 (yours)"], 1)[0] for i in g]) - FEE
            hr = pd.to_datetime(d.index.values[g], unit="ms").strftime("%m-%d %H:00")
            t = pd.DataFrame({"hr": hr, "r": res}).groupby("hr").r.agg(["count", "mean", lambda s: (s > 0).mean() * 100])
            L += ["", "HOT state, exit A, by hour (entries · net % per trade · hit rate):", "",
                  " · ".join(f"{k} → {int(v['count'])} · {v['mean']:+.2f} · {v.iloc[2]:.0f}%" for k, v in t.iterrows())]
        L.append("")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
