#!/usr/bin/env python3
"""🌊 Breadth at the EMA50-break short (operator 2026-10-02: "what about breadth instead of BTC?"). Breadth = the share of ALL futures pairs in a
state, on 5m bars closed before the entry: below their own EMA50 / EMA200 · last 1 h / 4 h / 24 h negative · and the CHANGE of the
below-EMA50 share over the last hour (market turning down vs turning up). Year: all EMA50-break shorts, strict ruler (as the market-state test).
Case: the MOVR / SAND shorts of the last 3 days (80 most-traded pairs). A screen read after the fact — window units (by-day range).
Usage: venv/bin/python scripts/break_short_breadth.py"""
import glob
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_market_state as M  # noqa: E402
sys.argv = _a
H, B, FS, ST, RC, BAR = M.H, M.B, M.FS, M.ST, M.RC, M.BAR
OUT = os.path.join(ST.ROOT, "reports", "BREAK_SHORT_BREADTH_2026-10-02.md")
COLS = ("below50", "below200", "down1h", "down4h", "down24h")


def breadth(frames, grid):
    n = len(grid); pos = {int(t): i for i, t in enumerate(grid)}; neg = {k: np.zeros(n) for k in COLS}; cnt = {k: np.zeros(n) for k in COLS}
    for d in frames:
        t = d.open_time.values.astype("int64"); c = d.c.values; s = pd.Series(c); ix = np.array([pos.get(int(x), -1) for x in t]); ok = ix >= 0
        if len(c) < 300:
            continue
        v = dict(below50=np.where(np.arange(len(c)) >= 150, (c < s.ewm(span=50, adjust=False).mean().values).astype(float), np.nan),
                 below200=np.where(np.arange(len(c)) >= 600, (c < s.ewm(span=200, adjust=False).mean().values).astype(float), np.nan))
        for k, lag in (("down1h", 12), ("down4h", 48), ("down24h", 288)):
            r = np.r_[[np.nan] * lag, c[lag:] / c[:-lag] - 1]; v[k] = np.where(np.isnan(r), np.nan, (r < 0).astype(float))
        for k in COLS:
            m = ok & ~np.isnan(v[k]); np.add.at(cnt[k], ix[m], 1); np.add.at(neg[k], ix[m], v[k][m])
    with np.errstate(invalid="ignore", divide="ignore"):
        F = pd.DataFrame({k: neg[k] / cnt[k] * 100 for k in COLS}, index=grid)
    F["d_below50"] = F.below50 - F.below50.shift(12); return F


if __name__ == "__main__":
    L = ["# 🌊 EMA50-break short — breadth of the whole market at entry", ""]
    s_ms = int(pd.Timestamp("2026-09-29 20:00", tz="UTC").timestamp() * 1000); bt = ST.api5m("BTCUSDT", days=8)
    tick = json.load(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=25))
    top = [x["symbol"] for x in sorted((x for x in tick if x["symbol"].endswith("USDT") and x["symbol"].isascii()), key=lambda x: -float(x["quoteVolume"]))[:80]]
    fr = []
    for p in top:
        try:
            fr.append(ST.api5m(p, days=8))
        except Exception:
            pass
    BR = breadth(fr, bt.open_time.values.astype("int64"))
    L += [f"## 1 · MOVR and SAND, last 3 days ({len(fr)} most-traded pairs)", "",
          "| Pair | Entry (UTC) | Result | Pairs below their EMA50 | change in the last hour | Pairs below their EMA200 | Falling 1 h | Falling 4 h | Falling 24 h |", "|---|---|---|---|---|---|---|---|---|"]
    for pair in ("MOVRUSDT", "SANDUSDT"):
        d = ST.api5m(pair); T1, H1, L1, C1 = RC.m1_all(pair, s_ms); free = 0
        for x in sorted((x for x in B.triggers(pair, d, 50, min_q24=0) if x["t"] >= s_ms), key=lambda x: x["t"]):
            i = int(np.searchsorted(T1, x["t"]))
            if x["t"] < free or i >= len(C1) - 2:
                continue
            r, mins, how = RC.walk("SHORT", H1[i:], L1[i:], C1[i:], x["entry"], 2.0, 5.0, 3.0); free = x["t"] + mins * 60_000; k = x["t"] - BAR
            if k in BR.index:
                b = BR.loc[k]; L.append(f"| {pair[:-4]} | {pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} | {r - 0.11:+.2f}% {how} | {b.below50:.0f}% | {b.d_below50:+.0f} pts | {b.below200:.0f}% | {b.down1h:.0f}% | {b.down4h:.0f}% | {b.down24h:.0f}% |")
    TR = pd.read_csv(os.path.join(B.BC, "break_short_triggers.csv")); TR = TR[TR.line == 50].copy(); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d")
    paths = {}
    for r in TR.itertuples():
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None
    grid = pd.read_csv(os.path.join(ST.K5, "BTCUSDT.csv"), usecols=["open_time"]).drop_duplicates().sort_values("open_time").open_time.values.astype("int64")
    frames = [pd.read_csv(f, usecols=["open_time", "c"]).drop_duplicates("open_time").sort_values("open_time") for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))) if os.path.basename(f).isascii()]
    YR = breadth(frames, grid); k = TR.t.values - BAR
    for col in YR.columns:
        TR[col] = YR[col].reindex(k).values
    E = H.run(TR, paths, *H.MAIN, slip=0.10, gap=True); X = TR.loc[E.index].copy(); X["mins"] = E.mins
    X["net"] = E.r.values - B.COST + np.array([0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])
    Q = (0, 30, 45, 55, 70, 101)
    CUTS = [(n, c, [(a, b, f"{a}–{min(b, 100)} %") for a, b in zip(Q, Q[1:])]) for n, c in (("Pairs below their 5m EMA50", "below50"), ("Pairs below their 5m EMA200", "below200"), ("Pairs falling, last 1 h", "down1h"),
                                                                                         ("Pairs falling, last 4 h", "down4h"), ("Pairs falling, last 24 h", "down24h"))]
    CUTS.append(("Change in pairs below EMA50, last hour", "d_below50", [(-101, -15, "improving fast (−15 pts or more)"), (-15, -5, "improving"), (-5, 5, "flat"), (5, 15, "worsening"), (15, 101, "worsening fast (+15 pts or more)")]))
    L += ["", "## 2 · The year — every EMA50-break short by breadth at entry", "",
          f"{len(X):,} trades, {len(frames)} pairs in the breadth, coverage {X.below200.notna().mean() * 100:.0f}%. Each cell: trades · won · per trade (Jan–Apr / May–Sep) · 95 % range by day.", "",
          "| Breadth | State | All runs (≥ +50 %) | Run > +200 % |", "|---|---|---|---|", f"| – | all trades | {M.line(X) if hasattr(M, 'line') else ''} | |"]
    def line(v):
        if len(v) < 30:
            return f"{len(v)} · –"
        lo_, hi_, n = H.boot(v, "day", 1000); return f"{len(v)} · {(v.net > 0).mean() * 100:.0f}% · {v.net.mean():+.2f} ({v[v.day < B.SPLIT].net.mean():+.2f} / {v[v.day >= B.SPLIT].net.mean():+.2f}) · [{lo_:+.2f}, {hi_:+.2f}] {n} d"
    L[-1] = f"| – | all trades | {line(X)} | {line(X[X.gain >= 200])} |"
    for name, col, bands in CUTS:
        for a, b, lab in bands:
            v = X[(X[col] >= a) & (X[col] < b)]; L.append(f"| {name} | {lab} | {line(v)} | {line(v[v.gain >= 200])} |")
    L += ["", "## NOT tested", "", "- Volume-weighted breadth, sector breadth, breadth of the pumped-pair group only.", "- Year breadth uses currently listed pairs; the case uses the 80 most-traded — not the same list.",
          "- 6 breadth measures × 5 states read after the fact: a stand-out cell is a hypothesis to freeze, not a rule."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
