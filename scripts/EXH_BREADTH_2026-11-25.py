#!/usr/bin/env python3
"""EXH_BREADTH (2026-11-25, research only) — does EXHAUSTION_SHORT lose when the market's short-term breadth is bearish?

Operator (11-25): the two BIGTIME losses + YB came with > 65 % of coins bearish ("why would > 65 % be bad for a short? it shouldn't" ·
"breadth was increasing for bearish or decreasing?" · "can we run the backtest?").

PRE-REGISTERED (written before any backtest outcome was read):
  Breadth = the bot's own statistic (services/trading_engine market breadth): share of the top-50 USDT perps by trailing 24 h quote
  volume whose 5m EMA20 changed by < −0.02 % over 3 bars (BEARISH; > +0.02 % BULLISH; market_breadth_flat_threshold 0.02), read on
  the last CLOSED 5m bar before the entry. Universe = every pair in the local 5m store with data at that bar (the bot also drops its
  blacklist — not modelled). Majors exist in the store only from 2024-06 → trades 2024-06-02 → 2026-11-16 (DESIGN < 2025-07,
  HOLDOUT ≥ 2025-07, the TP_GRID split).
  Trades: TP_GRID 2026-11-20 FRENZY rows (3,368 kept EXHAUSTION fills, DL 582 applied, one position per pair) priced at TODAY's exit
  (cell F7 = TP 7 / SL min(1.5 × ATR15, 6 %) / 24 h, live costs, real funding).
  Theory (why bearish breadth should HELP a short: no market bid under the pump, risk-off deleveraging) vs three ways it could HURT:
    H1 relative strength — a coin pumping while most of the market falls has its own buyer; the RSI roll-over is a pause, not the end
       → zone trades lose WHILE the market keeps falling (BTC 60 min after entry ≤ 0);
    H2 crowded shorts — sell-offs fill alts with shorts; a coin that will not fall squeezes them → zone losers carry negative funding;
    H3 market bounce — breadth at an extreme reverts and lifts every alt → zone losers come with BTC UP in the 60 min after entry.
  Zone Z = bear % > 65 (the live observation, not re-fitted). Direction = bear % now − bear % 60 min earlier (> 0 rising).
  Verdict on the locked Pattern-C expectancy bar (CLAUDE.md): blocked cohort WR < the sleeve's breakeven WR · avg < 0 with ≥ 95 %
  confidence by a DAY-block bootstrap · ≥ 8 distinct days, no day / pair ≥ 50 % of the cohort loss · N ≥ 15; per half-year; HOLDOUT.
  Market-wide variable → evidence in DAY units (window-units rule).
Run: venv/bin/python -I scripts/EXH_BREADTH_2026-11-25.py [--procs 7] → reports/EXH_BREADTH_2026-11-25{.csv,_trades.csv,_halves.csv}
     … --parity → rebuilt breadth vs the bot's entry_bear_pct stamp on 400 MASTER_POOL_stacked fills (2024-06 → 2026-11-15, seed 1)
"""
import argparse
import os
import sys
import multiprocessing as mp

import numpy as np
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_ta_build as TB  # noqa: E402

GD = os.path.join(R, "reports", "TP_GRID_2026-11-20_EXH_data")
OUT = os.path.join(R, "reports", "EXH_BREADTH_2026-11-25")
BAR, H1, DAY = 300_000, 3_600_000, 86_400_000
START = int(pd.Timestamp("2024-06-02", tz="UTC").value // 1_000_000)
HOLD_START = int(pd.Timestamp("2025-07-01", tz="UTC").value // 1_000_000)
FLAT, TOPN, ZONE = 0.02, 50, 65.0
OFFS = (-12, 0, 12)                                   # 5m bars: −60 min · entry · +60 min
RNG = np.random.default_rng(20261125)


def need_times(entry_ms):
    """bar open times to read: the last bar CLOSED before each entry (open = floor(entry) − 5m) and ±60 min around it."""
    b0 = (np.asarray(entry_ms, "int64") // BAR) * BAR - BAR
    return np.unique(np.concatenate([b0 + k * BAR for k in OFFS])), b0


def pair_reads(args):
    """one pair → (times present, 24 h quote volume, EMA20 3-bar slope %, close) at the needed bar times."""
    p, T = args
    z = TB.load(p)
    if z is None:
        return p, None
    t, o, h, l, c, v, qv = z
    lo, hi = t[0], t[-1]
    T = T[(T >= lo + 300 * BAR) & (T <= hi)]
    if not len(T):
        return p, None
    e = pd.Series(c).ewm(span=20, adjust=False).mean().values
    sl = np.full(len(e), np.nan)
    sl[3:] = (e[3:] / e[:-3] - 1) * 100
    q24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    i = ((T - lo) // BAR).astype(int)
    return p, (T, q24[i], sl[i], c[i])


def breadth(results, times):
    """bear % / bull % / n per bar time: top-N by 24 h quote volume among the pairs present."""
    idx = {t: k for k, t in enumerate(times)}
    rows = [[] for _ in times]
    for p, r in results:
        if r is None:
            continue
        T, q, s, _c = r
        for t, qq, ss in zip(T, q, s):
            if np.isfinite(qq) and np.isfinite(ss):
                rows[idx[t]].append((qq, ss))
    bear = np.full(len(times), np.nan); bull = bear.copy(); n = np.zeros(len(times), int)
    for k, rr in enumerate(rows):
        if len(rr) < TOPN:
            continue
        a = np.array(rr); a = a[np.argsort(-a[:, 0])][:TOPN]
        bear[k] = (a[:, 1] < -FLAT).mean() * 100; bull[k] = (a[:, 1] > FLAT).mean() * 100; n[k] = len(a)
    return bear, bull, n


def wilson_floor(k, n, z=1.645):
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def day_boot(v, days, nb=4000, block=10):
    """day-unit CI = the TP_GRID boot_ci method: 10-day MOVING blocks over every calendar day of the cohort's span (days without a
    trade count as 0 trades), mean = Σ resampled P&L ÷ Σ resampled trades. → (one-sided 95 % upper bound, 2.5 %, 97.5 %)."""
    s_ = pd.Series(v).groupby(np.asarray(days)).agg(["sum", "count"])
    dd = np.arange(s_.index.min(), s_.index.max() + 1); s_ = s_.reindex(dd, fill_value=0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    m = s_["sum"].values[idx].sum(1) / np.maximum(s_["count"].values[idx].sum(1), 1)
    return float(np.percentile(m, 95)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohort(v, days, pairs, be):
    if len(v) < 2:
        return dict(N=len(v))
    hi95, lo, hi = day_boot(v, days)
    neg = v[v < 0]; dl = pd.Series(np.minimum(v, 0)).groupby(days).sum(); pl = pd.Series(np.minimum(v, 0)).groupby(pairs).sum()
    return dict(N=len(v), days=int(pd.Series(days).nunique()), WR=(v > 0).mean() * 100, BE=be, avg=v.mean(), ci_lo=lo, ci_hi=hi,
                ub95=hi95, top_day_loss_share=(dl.min() / neg.sum() * 100) if len(neg) else 0.0,
                top_pair_loss_share=(pl.min() / neg.sum() * 100) if len(neg) else 0.0,
                kelly_floor=(lambda w, lo_: wilson_floor(len(w), len(v)) - (1 - wilson_floor(len(w), len(v))) / (w.mean() / -lo_.mean())
                             if len(w) and len(lo_) else np.nan)(v[v > 0], v[v <= 0]))


def parity(procs):
    M = pd.read_csv(os.path.join(R, "reports", "MASTER_POOL_stacked.csv"), low_memory=False, usecols=["opened_at", "entry_bear_pct"]).dropna()
    M["ms"] = [pd.Timestamp(x).value // 1_000_000 for x in M.opened_at]
    M = M[(M.ms >= START) & (M.ms < int(pd.Timestamp("2026-11-15", tz="UTC").value // 1_000_000))].drop_duplicates("ms")
    M = M.sample(min(400, len(M)), random_state=1)
    times, b0 = need_times(M.ms.values); M["b0"] = b0
    pairs = sorted({f[:-4] for d in TB.K5M_SRC if os.path.isdir(d) for f in os.listdir(d) if f.endswith(".csv")})
    with mp.get_context("fork").Pool(procs) as pool:
        res = list(pool.imap_unordered(pair_reads, [(p, times) for p in pairs], chunksize=4))
    bear, _bull, _n = breadth(res, times); ti = {t: k for k, t in enumerate(times)}
    M["rebuilt"] = [bear[ti[t]] for t in M.b0]; M = M.dropna()
    z1, z2 = M.entry_bear_pct > ZONE, M.rebuilt > ZONE
    print("parity N", len(M), "corr", round(float(np.corrcoef(M.entry_bear_pct, M.rebuilt)[0, 1]), 3),
          "median |diff|", round(float((M.entry_bear_pct - M.rebuilt).abs().median()), 1), "zone agreement %", round(float((z1 == z2).mean() * 100), 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=7); ap.add_argument("--parity", action="store_true")
    A = ap.parse_args()
    if A.parity:
        return parity(A.procs)
    X = pd.read_pickle(os.path.join(GD, "rows_FRENZY.pkl")).reset_index(drop=True)
    Z = dict(np.load(os.path.join(GD, "sim_FRENZY.npz"), allow_pickle=True))
    names = [str(x) for x in Z.pop("names")]
    assert (Z["xi"] == X.xi.values).all()
    j = names.index("F7")
    s = np.maximum(X.slip.values.astype(float), 0.10)
    tk = Z["k"][:, j] != 0
    X["net"] = Z["r"][:, j].astype(float) + Z["f"][:, j].astype(float) - s - 0.05 - np.where(tk, s + 0.05, 0.02)
    X["fund"] = Z["f"][:, j].astype(float)
    X = X[Z["ok"] & (X.entry_ms.values >= START)].reset_index(drop=True)
    times, b0 = need_times(X.entry_ms.values)
    X["b0"] = b0
    pairs = sorted(f[:-4] for d in TB.K5M_SRC if os.path.isdir(d) for f in os.listdir(d) if f.endswith(".csv"))
    pairs = sorted(set(pairs))
    print("trades", len(X), "bar times", len(times), "pairs in store", len(pairs), flush=True)
    with mp.get_context("fork").Pool(A.procs) as pool:
        res = list(pool.imap_unordered(pair_reads, [(p, times) for p in pairs], chunksize=4))
    bear, bull, n = breadth(res, times)
    ti = {t: k for k, t in enumerate(times)}
    btc = dict(res)["BTCUSDT"]; bt = dict(zip(btc[0], btc[3]))
    for k, nm in ((0, "bear"), (-12, "bear_m60"), (12, "bear_p60")):
        X[nm] = [bear[ti[t + k * BAR]] if (t + k * BAR) in ti else np.nan for t in X.b0]
    X["bull"] = [bull[ti[t]] for t in X.b0]; X["n_univ"] = [n[ti[t]] for t in X.b0]
    X["btc_p60"] = [(bt.get(t + 12 * BAR, np.nan) / bt.get(t, np.nan) - 1) * 100 for t in X.b0]
    X["btc_m60"] = [(bt.get(t, np.nan) / bt.get(t - 12 * BAR, np.nan) - 1) * 100 for t in X.b0]
    X["day"] = X.entry_ms // DAY
    X["blk"] = np.where(X.entry_ms >= HOLD_START, "HOLDOUT", "DESIGN")
    print("fills without a breadth read (< 50 pairs with data at the bar) — dropped:", int((~np.isfinite(X.bear)).sum()), flush=True)
    X = X[np.isfinite(X.bear)].reset_index(drop=True)
    w, lo = X.net[X.net > 0], X.net[X.net <= 0]
    be = -lo.mean() / (w.mean() - lo.mean()) * 100
    X["zone"] = X.bear > ZONE
    X["rising"] = (X.bear - X.bear_m60) > 0
    rows = []
    for blk in ("ALL", "DESIGN", "HOLDOUT"):
        B = X if blk == "ALL" else X[X.blk == blk]
        for nm, m in (("ALL", np.ones(len(B), bool)), ("ZONE bear>65", B.zone.values), ("rest", ~B.zone.values),
                      ("ZONE · rising", (B.zone & B.rising).values), ("ZONE · falling", (B.zone & ~B.rising).values),
                      ("bear 0-20", (B.bear <= 20).values), ("bear 20-40", ((B.bear > 20) & (B.bear <= 40)).values),
                      ("bear 40-65", ((B.bear > 40) & (B.bear <= 65)).values), ("bear 65-80", ((B.bear > 65) & (B.bear <= 80)).values),
                      ("bear >80", (B.bear > 80).values)):
            sub = B[m]
            r = cohort(sub.net.values, sub.day.values, sub.pair.values, be)
            r.update(block=blk, cohort=nm,
                     btc_p60_mean=float(sub.btc_p60.mean()) if len(sub) else np.nan,
                     losers_btc_p60_up_share=float((sub.btc_p60[sub.net <= 0] > 0).mean() * 100) if (sub.net <= 0).any() else np.nan,
                     losers_fund_mean=float(sub.fund[sub.net <= 0].mean()) if (sub.net <= 0).any() else np.nan,
                     winners_fund_mean=float(sub.fund[sub.net > 0].mean()) if (sub.net > 0).any() else np.nan)
            rows.append(r)
    T = pd.DataFrame(rows)
    halves = X.groupby(["half", "zone"]).net.agg(["count", "mean"]).unstack()
    T.to_csv(OUT + ".csv", index=False)
    X[["pair", "entry_ms", "half", "blk", "net", "fund", "bear", "bear_m60", "bear_p60", "bull", "n_univ", "btc_m60", "btc_p60",
       "zone", "rising"]].to_csv(OUT + "_trades.csv", index=False)
    halves.to_csv(OUT + "_halves.csv")
    pd.set_option("display.width", 250)
    print("breakeven WR", round(be, 1))
    print(T.round(3).to_string(index=False))
    print(halves.round(3).to_string())


if __name__ == "__main__":
    main()
