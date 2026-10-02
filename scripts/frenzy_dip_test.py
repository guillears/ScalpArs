#!/usr/bin/env python3
"""🌋 FRENZY DIP — does the one rule that worked on MOVR (2026-09-30 / 10-01) hold on every other volume frenzy of the year?

MOVR, 48 h: "LONG after a ≥ 5 % drop from the 4-hour high, exit +3.09 % / −1.51 %" = 77 trades +31 % and 127 trades +25 % on its two
days. The same rule lost on ARK, NOM and ALICE the same week (volume only 7–34× normal). Hypothesis: it needs a real FRENZY.

PRE-DECLARED (before any result of this script):
  frenzy   a 5m bar where the pair's 24 h quote volume ≥ 100× its normal day (median 24 h volume over the 30 days ending 2 days
           earlier) ∧ 24 h return ≥ +30 % ∧ 5m ATR(14) ≥ 2 % ∧ 24 h volume ≥ $20M — all from CLOSED bars. MOVR: 112–213×, +84 %, 3.4 %.
           Bars < 6 h apart = one episode. Pairs with a spot 1-second feed only (others counted, not tested).
  entry    inside a frenzy bar, at a minute close, LONG when the price is ≥ 5 % below the highest high of the prior 4 hours
  exit     +3.09 % / −1.51 % (gross), 30-min limit, stop first inside a second; one position per pair; 1 min pause after an exit
  compare  the same exit entered at EVERY frenzy minute (no dip), and the dip entry with +0.59 / −1.11 and +1.09 / −1.11
  costs    0.09 % fees + 0.02 % slippage
  reads    TRADE-weighted (every trade equal), DAY-weighted (t-interval on UTC-day means) and FIRST trade of each episode;
           halves Jan–Apr / May–Sep. MOVR itself is after the cache end → not in this sample.
  PASS     trade-weighted mean > 0 in BOTH halves ∧ day-mean 95 % interval above 0 over the year ∧ first-trade mean > 0
Usage: venv/bin/python scripts/frenzy_dip_test.py [--report-only] → reports/FRENZY_DIP_TEST_2026-10-01.md"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import hot_scalp_backtest as H  # noqa: E402  (k1s fetch, t-quantile)
sys.argv = _a
RAW = os.path.join(H.ROOT, "reports", "backtest_cache", "k1s_frenzy"); TR = os.path.join(H.ROOT, "reports", "backtest_cache", "frenzy_trades.csv")
OUT = os.path.join(H.ROOT, "reports", "FRENZY_DIP_TEST_2026-10-01.md"); COST, HOLD, VOLX = 0.11, 1800, 100
VARIANTS = {"DIP → +3.09 / −1.51 (the MOVR rule)": (True, 3.09, 1.51), "DIP → +1.09 / −1.11": (True, 1.09, 1.11), "DIP → +0.59 / −1.11": (True, 0.59, 1.11),
            "ANY frenzy minute → +3.09 / −1.51": (False, 3.09, 1.51), "ANY frenzy minute → +0.59 / −1.11": (False, 0.59, 1.11)}


def frame(pair):
    d = pd.read_csv(os.path.join(H.K5, pair + ".csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    if len(d) < 288 * 40:
        return None, []
    q24 = d.qvol.rolling(288).sum(); norm = q24.shift(288 * 2).rolling(288 * 30).median()
    pc = d.c.shift(1); tr = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100
    # everything shifted one bar: usable DURING the next bar
    d["frenzy"] = (((q24 / norm) >= VOLX) & ((d.c / d.c.shift(288) - 1) * 100 >= 30) & (atr >= 2) & (q24 >= 20e6)).shift(1).fillna(False).astype(bool)
    d["hi4"] = d.h.rolling(48).max().shift(1)                          # highest high of the 4 h of CLOSED bars before this bar
    t = d.index.values[d.frenzy.values]
    eps = [] if not len(t) else [(int(e[0]), int(e[-1])) for e in np.split(t, np.where(np.diff(t) > 6 * 3600_000)[0] + 1)]
    return d, eps


def episode(pair, d5, t0, t1):
    f = os.path.join(RAW, f"{pair}_{t0}.npz")
    if os.path.exists(f):
        z = np.load(f); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]
    else:
        s = H.k1s(pair, t0, t1 + H.BAR + HOLD * 1000)
        if s is None or len(s) < 600:
            return None
        os.makedirs(RAW, exist_ok=True); ts, h, l, c = s.index.values.astype("int64"), s.h.values, s.l.values, s.c.values
        np.savez_compressed(f, t=ts, h=h, l=l, c=c)
    st = d5.reindex(ts // H.BAR * H.BAR); fz = st.frenzy.values.astype(bool); hi4 = st.hi4.values
    out = []
    for vn, (dip, tp, sl) in VARIANTS.items():
        free = 0; first = True
        for i in range(59, len(c) - 60, 60):                           # minute closes
            if i < free or not fz[i]:
                continue
            top = max(hi4[i], h[max(0, i - 14400):i + 1].max())
            if dip and not (c[i] <= top * 0.95):
                continue
            e = c[i]; Hh, Ll = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD]
            hs = np.nonzero(Ll <= e * (1 - sl / 100))[0]; ht = np.nonzero(Hh >= e * (1 + tp / 100))[0]
            a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
            r, secs = ((c[min(i + HOLD, len(c) - 1)] / e - 1) * 100, HOLD) if a == b == 10**9 else ((-sl, a + 1) if a <= b else (tp, b + 1))
            out.append((pair, f"{pair}:{t0}", int(ts[i]), vn, round(r, 4), first)); first = False; free = i + secs + 60
    return out


def report():
    T = pd.read_csv(TR); meta = json.load(open(TR + ".meta.json")); T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d"); T["net"] = T.r - COST
    L = ["# 🌋 FRENZY DIP — the MOVR rule on every volume frenzy of the year", "",
         f"Frenzy = 24 h volume ≥ {VOLX}× the pair's normal day ∧ +30 % in 24 h ∧ 5m ATR ≥ 2 %. {meta['episodes']} episodes on {meta['pairs']} pairs in Jan–Sep 2026; "
         f"{meta['tested']} had a spot 1-second feed and were tested ({meta['nospot']} futures-only skipped). After 0.11 % costs. MOVR itself is not in this sample.", "",
         "| Variant | trades | episodes | days | won | per trade (trade-weighted) Jan–Apr / May–Sep | by day [95 %] | first trade of each episode | PASS |", "|---|---|---|---|---|---|---|---|---|"]
    for vn in VARIANTS:
        g = T[T.variant == vn]
        if len(g) < 20:
            L.append(f"| {vn} | {len(g)} | – | – | – | – | – | – | — |"); continue
        h1, h2 = g[g.day < H.SPLIT].net.mean(), g[g.day >= H.SPLIT].net.mean()
        dm = g.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n); lo, hi = dm.mean() - q * se, dm.mean() + q * se
        ft = g[g.first_].net.mean(); ok = h1 > 0 and h2 > 0 and lo > 0 and ft > 0
        L.append(f"| {vn} | {len(g):,} | {g.eid.nunique()} | {n} | {(g.net > 0).mean() * 100:.0f}% | {h1:+.3f} / {h2:+.3f} | {dm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {ft:+.3f} | {'✅' if ok else '—'} |")
    g = T[T.variant == list(VARIANTS)[0]]
    if len(g):
        e = g.groupby("eid").net.sum(); p = g.groupby("pair").net.sum()
        L += ["", f"The MOVR rule by episode: {(e > 0).mean() * 100:.0f} % of {len(e)} episodes positive · best {e.max():+.1f} % · worst {e.min():+.1f} % · median {e.median():+.1f} %. "
              f"By pair: {(p > 0).mean() * 100:.0f} % of {len(p)} positive; the best 3 pairs carry {p.sort_values().tail(3).sum():+.0f} % of a total {p.sum():+.0f} %."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        spot = {s["symbol"] for s in (H.get("https://api.binance.com/api/v3/exchangeInfo") or {}).get("symbols", [])}
        if not spot:
            raise SystemExit("spot exchangeInfo failed")
        meta = dict(episodes=0, pairs=0, tested=0, nospot=0); rows = []; t_start = time.time()
        files = sorted(glob.glob(os.path.join(H.K5, "*.csv")))
        for n_, f in enumerate(files):
            pair = os.path.basename(f)[:-4]
            if pair in ("BTCUSDT", "ETHUSDT"):
                continue
            d5, eps = frame(pair)
            if not eps:
                continue
            meta["episodes"] += len(eps); meta["pairs"] += 1
            if pair not in spot or pair.startswith("1000"):
                meta["nospot"] += len(eps); continue
            for t0, t1 in eps:
                r = episode(pair, d5, t0, t1)
                if r is not None:
                    meta["tested"] += 1; rows += r
            print(f"[{n_ + 1}/{len(files)}] {pair}: {len(eps)} episodes · tested {meta['tested']} · {time.time() - t_start:.0f}s", flush=True)
        pd.DataFrame(rows, columns=["pair", "eid", "t", "variant", "r", "first_"]).to_csv(TR, index=False); json.dump(meta, open(TR + ".meta.json", "w"))
    report()
