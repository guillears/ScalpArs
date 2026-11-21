"""Route B signal-level loader — the NEW-SLEEVE trigger table H1–H6 (reports/ROUTE_B_REPORT_2026-11-21.md §4).

PRE-REGISTRATION FIRST (operator): outcomes are COMPUTED and STORED (triggers_H.csv.gz) but never printed or summarised by this code — the
report shows counts and coverage only. Definitions = §4 of the Route B report, made strictly causal: every coin "norm" (99th pct, median)
is computed over rows BEFORE the trigger minute (the §4 counts used whole-sample norms; these counts are therefore not identical).
Alts only (BTC / ETH out), USDT perpetuals. One POSSIBILITY per coin per 60 min (H6: 24 h), per trigger and side; 15-min windows.

  H1  long-liquidation cascade → LONG   5-min long liqs ≥ the coin's prior 99th pct (≥ 20 non-zero) and ≥ $50k (book recorded)
  H2  short-squeeze fade → SHORT        same on short liquidations
  H3  thin-book imbalance → follow      imb_05 ≥ +0.5 for 3 recorded minutes in a row (≤ −0.5 → SHORT), bid+ask 0.5 % < $40k
  H4  book vacuum → toward the vacuum   ask $ within 0.5 % < 0.35 × its prior median (≥ 60 rows) → LONG; bid side → SHORT
  H5  taker climax → fade               15-min traded $ ≥ 5 × prior norm and taker-buy share ≥ 0.6 → SHORT (≤ 0.4 → LONG)
  H6  funding extreme                   funding ≤ −0.5 % per settlement → LONG (squeeze) · ≥ +0.1 % → SHORT (crowded longs); all coins
Signal time T = the END of trigger minute m (m + 60 s) → state = minute m (no look-ahead) · entry = the open of the first minute starting
≥ T + 1 s = the open of minute m + 2 (one full minute after the trigger minute closed).
"""
import numpy as np
import pandas as pd

from common import MIN_MS

EXCL = ("BTCUSDT", "ETHUSDT")
DEFS = [("H1_liq_long_cascade", "LONG"), ("H2_liq_short_squeeze", "SHORT"), ("H3_imb_bid_heavy_thin", "LONG"), ("H3_imb_ask_heavy_thin", "SHORT"),
        ("H4_ask_vacuum", "LONG"), ("H4_bid_vacuum", "SHORT"), ("H5_taker_buy_climax", "SHORT"), ("H5_taker_sell_climax", "LONG"),
        ("H6_funding_le_-0.5pct", "LONG"), ("H6_funding_ge_+0.1pct", "SHORT")]


def episodes(sym_min, gap_min=60):
    """[(sym, minute)] trigger minutes → the first of each run per coin with ≥ gap_min between openers. Pure."""
    out = []
    for s, g in pd.DataFrame(sym_min, columns=["sym", "minute"]).sort_values(["sym", "minute"]).groupby("sym"):
        last = -10 ** 18
        for m in g.minute:
            if m - last >= gap_min * MIN_MS:
                out.append((s, int(m)))
                last = m
    return out


def _prior_expanding(x, fn, min_n):
    """value at row i over rows < i (shifted expanding stat)."""
    e = getattr(x.expanding(min_periods=min_n), fn)()
    return e.shift(1)


def detect(book, mark, coins):
    """→ DataFrame(trigger, direction, sym, minute) of trigger OPENERS (episodes). coins: {sym: state.Coin} for the liquidation baseline."""
    hits = {k: [] for k, _ in DEFS}
    for sym, g in book.groupby("sym"):
        if sym in EXCL or len(g) < 5:
            continue
        g = g.sort_values("minute").reset_index(drop=True)
        m = g.minute.to_numpy(np.int64)
        idx = pd.to_datetime(m, unit="ms")
        # H1 / H2 — liquidation 5-min sums vs the coin's prior 99th pct (state.Coin baseline: minutes with a liquidation, before m)
        c = coins.get(sym)
        if c is not None and len(c.lm):
            lm, l5, s5 = c.liq5_series()
            for side, arr, key in (("long", l5, "H1_liq_long_cascade"), ("short", s5, "H2_liq_short_squeeze")):
                cum = np.concatenate([[0], np.cumsum(c.liq[f"liq_{side}_usd"].fillna(0).to_numpy(float))])
                j, i = np.searchsorted(c.lm, m, "right"), np.searchsorted(c.lm, m - 5 * MIN_MS, "right")
                r5 = cum[j] - cum[i]
                for q in np.nonzero(r5 >= 50_000)[0]:
                    p = c.p99_before(side, m[q])
                    if p == p and r5[q] >= p:
                        hits[key].append((sym, m[q]))
        # H3 — imbalance 3 consecutive recorded minutes, thin book
        imb = pd.Series(g.imb_05.to_numpy(float), index=idx)
        cnt = imb.rolling("3min").count()
        thin = (g.bid_usd_05 + g.ask_usd_05).to_numpy(float) < 40_000
        mn, mx = imb.rolling("3min").min().to_numpy(), imb.rolling("3min").max().to_numpy()
        ok3 = cnt.to_numpy() >= 3
        for q in np.nonzero(ok3 & thin & (mn >= 0.5))[0]:
            hits["H3_imb_bid_heavy_thin"].append((sym, m[q]))
        for q in np.nonzero(ok3 & thin & (mx <= -0.5))[0]:
            hits["H3_imb_ask_heavy_thin"].append((sym, m[q]))
        # H4 — vacuum vs the prior median
        for side, key in (("ask", "H4_ask_vacuum"), ("bid", "H4_bid_vacuum")):
            v = g[f"{side}_usd_05"].astype(float)
            med = _prior_expanding(v, "median", 60).to_numpy()
            for q in np.nonzero(v.to_numpy() < 0.35 * med)[0]:
                hits[key].append((sym, m[q]))
        # H5 — taker climax vs the prior norm (median recorded minute × 15, before the window)
        b = pd.Series(g.buy_usd.fillna(0).to_numpy(float), index=idx)
        s_ = pd.Series(g.sell_usd.fillna(0).to_numpy(float), index=idx)
        f15 = (b + s_).rolling("15min").sum().to_numpy()
        b15 = b.rolling("15min").sum().to_numpy()
        n15 = b.rolling("15min").count().to_numpy()
        medall = (b + s_).expanding(min_periods=60).median()
        # norm = the expanding median as of the last row ≤ m − 15 min (rows before the window)
        mv = medall.to_numpy()
        k0 = np.searchsorted(m, m - 15 * MIN_MS, "right") - 1
        norm15 = np.where(k0 >= 0, mv[np.clip(k0, 0, None)], np.nan) * 15
        with np.errstate(invalid="ignore", divide="ignore"):
            share = b15 / f15
            big = (n15 >= 10) & (f15 >= 5 * norm15) & (norm15 > 0)
        for q in np.nonzero(big & (share >= 0.6))[0]:
            hits["H5_taker_buy_climax"].append((sym, m[q]))
        for q in np.nonzero(big & (share <= 0.4))[0]:
            hits["H5_taker_sell_climax"].append((sym, m[q]))
    # H6 — funding (every coin with a mark row)
    mk = mark[~mark.sym.isin(EXCL)]
    lo = mk[mk.funding <= -0.005]
    hi = mk[mk.funding >= 0.001]
    out = []
    for key, d in DEFS:
        if key.startswith("H6"):
            src = lo if "le" in key else hi
            ep = episodes(list(zip(src.sym, src.minute)), gap_min=24 * 60)
        else:
            ep = episodes(hits[key], gap_min=60)
        out += [(key, d, s, m) for s, m in ep]
    T = pd.DataFrame(out, columns=["trigger", "direction", "pair", "minute"])
    T["signal_time"] = pd.to_datetime(T.minute + MIN_MS, unit="ms", utc=True)
    T["window15"] = T.signal_time.dt.floor("15min").dt.strftime("%Y-%m-%dT%H:%M")
    return T
