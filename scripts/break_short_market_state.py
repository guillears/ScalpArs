#!/usr/bin/env python3
"""🌡️ Does the MARKET decide the EMA50-break short? Operator (2026-10-02): "there must be something different in the MOVR EMA50 short and the
SAND one, maybe market conditions — now with SAND everything is going down."
Part 1 (case): every MOVR / SAND EMA50 short of the last 3 days with the market state at its entry.
Part 2 (year): all 4,010 EMA50-break triggers (run ≥ +50 %) split by the SIGN of each market variable at entry — strict ruler (1-minute bars,
gap-aware stop, 0.10 % slippage, 0.11 % costs, funding), exit stop 2 % · trail from +5 % giving back 3 %.
  BTC      return over 1 h / 4 h / 24 h · price below its 5m EMA50 / EMA200
  breadth  share of all futures pairs whose last hour / last 4 h is negative ("everything is going down")
  pair     the pair's own last 1 h / 4 h
All read on bars CLOSED before the entry. Market variables repeat across same-day trades → the by-day interval is the read (window units).
A SCREEN (9 variables, read after the fact): a survivor is a hypothesis to freeze, not a rule.
Usage: venv/bin/python scripts/break_short_market_state.py"""
import glob
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import break_short_hostile_review as H  # noqa: E402
import run_case_study as RC  # noqa: E402
sys.argv = _a
B, FS, ST = H.B, H.FS, H.ST; BAR = 300_000
OUT = os.path.join(ST.ROOT, "reports", "BREAK_SHORT_MARKET_STATE_2026-10-02.md")


def btc_frame(d):
    c = d.c.values; s = pd.Series(c)
    return pd.DataFrame(dict(btc_r1h=(c / s.shift(12).values - 1) * 100, btc_r4h=(c / s.shift(48).values - 1) * 100, btc_r24h=(c / s.shift(288).values - 1) * 100,
                             btc_vs_e50=(c / s.ewm(span=50, adjust=False).mean().values - 1) * 100, btc_vs_e200=(c / s.ewm(span=200, adjust=False).mean().values - 1) * 100),
                        index=d.open_time.values.astype("int64"))


def breadth(frames, grid):
    """Share of pairs with a negative last hour / last 4 h on a common 5m grid."""
    n = len(grid); pos = {int(t): i for i, t in enumerate(grid)}; neg1 = np.zeros(n); neg4 = np.zeros(n); cnt1 = np.zeros(n); cnt4 = np.zeros(n)
    for d in frames:
        t = d.open_time.values.astype("int64"); c = d.c.values; ix = np.array([pos.get(int(x), -1) for x in t]); ok = ix >= 0
        for k, neg, cnt in ((12, neg1, cnt1), (48, neg4, cnt4)):
            r = np.r_[[np.nan] * k, c[k:] / c[:-k] - 1]; m = ok & ~np.isnan(r); np.add.at(cnt, ix[m], 1); np.add.at(neg, ix[m], (r[m] < 0).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(dict(down1h=neg1 / cnt1 * 100, down4h=neg4 / cnt4 * 100), index=grid)


if __name__ == "__main__":
    L = ["# 🌡️ EMA50-break short — does the market state at entry decide it?", ""]
    # ── Part 1: MOVR / SAND
    s_ms = int(pd.Timestamp("2026-09-29 20:00", tz="UTC").timestamp() * 1000)
    bt = ST.api5m("BTCUSDT", days=6); BF = btc_frame(bt)
    tick = json.load(urllib.request.urlopen("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=25))
    top = [x["symbol"] for x in sorted((x for x in tick if x["symbol"].endswith("USDT") and x["symbol"].isascii()), key=lambda x: -float(x["quoteVolume"]))[:80]]
    fr = []
    for p in top:
        try:
            fr.append(ST.api5m(p, days=5))
        except Exception:
            pass
    BR = breadth(fr, bt.open_time.values.astype("int64"))
    L += [f"## 1 · MOVR and SAND, last 3 days — each EMA50 short with the market at its entry (breadth = {len(fr)} most-traded pairs)", "",
          "| Pair | Entry (UTC) | Result | Exit | BTC 1 h | BTC 4 h | BTC vs its EMA50 | Pairs falling (1 h) | Pairs falling (4 h) | Pair's own 1 h | Pair vs peak |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for pair in ("MOVRUSDT", "SANDUSDT"):
        d = ST.api5m(pair); T1, H1, L1, C1 = RC.m1_all(pair, s_ms); pc = pd.Series(d.c.values, index=d.open_time.values.astype("int64")); free = 0
        for x in sorted((x for x in B.triggers(pair, d, 50, min_q24=0) if x["t"] >= s_ms), key=lambda x: x["t"]):
            if x["t"] < free:
                continue
            i = int(np.searchsorted(T1, x["t"]))
            if i >= len(C1) - 2:
                continue
            r, mins, how = RC.walk("SHORT", H1[i:], L1[i:], C1[i:], x["entry"], 2.0, 5.0, 3.0); free = x["t"] + mins * 60_000; k = x["t"] - BAR
            b = BF.loc[k] if k in BF.index else None; br = BR.loc[k] if k in BR.index else None; p1 = (pc.get(k, np.nan) / pc.get(k - 12 * BAR, np.nan) - 1) * 100
            L.append(f"| {pair[:-4]} | {pd.Timestamp(x['t'], unit='ms'):%m-%d %H:%M} | {r - 0.11:+.2f}% | {how} {mins} min | " + (f"{b.btc_r1h:+.2f}% | {b.btc_r4h:+.2f}% | {b.btc_vs_e50:+.2f}% | " if b is not None else "– | – | – | ")
                     + (f"{br.down1h:.0f}% | {br.down4h:.0f}% | " if br is not None else "– | – | ") + f"{p1:+.1f}% | {x['off_peak']:+.0f}% |")
    # ── Part 2: the year
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
    yb = pd.read_csv(os.path.join(ST.K5, "BTCUSDT.csv")).drop_duplicates("open_time").sort_values("open_time"); YB = btc_frame(yb); grid = yb.open_time.values.astype("int64")
    frames = []; own = {}
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        p = os.path.basename(f)[:-4]
        if not p.isascii():
            continue
        d = pd.read_csv(f, usecols=["open_time", "c"]).drop_duplicates("open_time").sort_values("open_time"); frames.append(d)
        if p in FR:
            own[p] = pd.Series(d.c.values, index=d.open_time.values.astype("int64"))
    YR = breadth(frames, grid); k = TR.t.values - BAR
    for col in YB.columns:
        TR[col] = YB[col].reindex(k).values
    for col in YR.columns:
        TR[col] = YR[col].reindex(k).values
    TR["pair_r1h"] = [(own[p].get(t, np.nan) / own[p].get(t - 12 * BAR, np.nan) - 1) * 100 if p in own else np.nan for p, t in zip(TR.pair, k)]
    TR["pair_r4h"] = [(own[p].get(t, np.nan) / own[p].get(t - 48 * BAR, np.nan) - 1) * 100 if p in own else np.nan for p, t in zip(TR.pair, k)]
    E = H.run(TR, paths, *H.MAIN, slip=0.10, gap=True); X = TR.loc[E.index].copy(); X["mins"] = E.mins
    X["net"] = E.r.values - B.COST + np.array([0.0 if FR.get(r.pair) is None else FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])

    def line(v):
        if len(v) < 30:
            return f"{len(v)} · –"
        lo_, hi_, n = H.boot(v, "day", 1000); return f"{len(v)} · {(v.net > 0).mean() * 100:.0f}% · {v.net.mean():+.2f} ({v[v.day < B.SPLIT].net.mean():+.2f} / {v[v.day >= B.SPLIT].net.mean():+.2f}) · [{lo_:+.2f}, {hi_:+.2f}] {n} d"

    CUTS = (("BTC last 1 h", "btc_r1h", ((-99, -0.5, "fell > 0.5 %"), (-0.5, 0, "fell 0–0.5 %"), (0, 0.5, "rose 0–0.5 %"), (0.5, 99, "rose > 0.5 %"))),
            ("BTC last 4 h", "btc_r4h", ((-99, -1, "fell > 1 %"), (-1, 0, "fell 0–1 %"), (0, 1, "rose 0–1 %"), (1, 99, "rose > 1 %"))),
            ("BTC last 24 h", "btc_r24h", ((-99, -2, "fell > 2 %"), (-2, 0, "fell 0–2 %"), (0, 2, "rose 0–2 %"), (2, 99, "rose > 2 %"))),
            ("BTC vs its 5m EMA50", "btc_vs_e50", ((-99, 0, "below"), (0, 99, "above"))), ("BTC vs its 5m EMA200", "btc_vs_e200", ((-99, 0, "below"), (0, 99, "above"))),
            ("Pairs falling, last 1 h", "down1h", ((0, 40, "< 40 %"), (40, 60, "40–60 %"), (60, 75, "60–75 %"), (75, 101, "≥ 75 %"))),
            ("Pairs falling, last 4 h", "down4h", ((0, 40, "< 40 %"), (40, 60, "40–60 %"), (60, 75, "60–75 %"), (75, 101, "≥ 75 %"))),
            ("The pair's own last 1 h", "pair_r1h", ((-999, -5, "fell > 5 %"), (-5, -2, "fell 2–5 %"), (-2, 0, "fell 0–2 %"), (0, 999, "rose"))),
            ("The pair's own last 4 h", "pair_r4h", ((-999, -10, "fell > 10 %"), (-10, 0, "fell 0–10 %"), (0, 10, "rose 0–10 %"), (10, 999, "rose > 10 %"))))
    L += ["", "## 2 · The year — every EMA50-break short split by the market at entry", "",
          f"{len(X):,} trades. Each cell: trades · won · per trade (Jan–Apr / May–Sep) · 95 % range by day. Coverage: BTC {X.btc_r1h.notna().mean() * 100:.0f}% · breadth {X.down1h.notna().mean() * 100:.0f}% · pair {X.pair_r1h.notna().mean() * 100:.0f}%.", "",
          "| Variable | State | All runs (≥ +50 %) | Run > +200 % |", "|---|---|---|---|", f"| – | all trades | {line(X)} | {line(X[X.gain >= 200])} |"]
    for name, col, bands in CUTS:
        for a, b, lab in bands:
            v = X[(X[col] >= a) & (X[col] < b)]; L.append(f"| {name} | {lab} | {line(v)} | {line(v[v.gain >= 200])} |")
    L += ["", "## NOT tested", "", "- Order-book or funding state at entry, sector moves, news.", "- Breadth on the year uses the cached pairs (currently listed); part 1 uses the 80 most-traded pairs — not the same list.",
          "- 9 variables × 2–4 states read after the fact: anything that stands out is a hypothesis to freeze and re-test, not a rule."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
