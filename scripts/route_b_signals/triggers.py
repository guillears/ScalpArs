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

Weekly-scan PREREG §5 (build item 5):
  H7  OI flush / build (hot coins with the recorder's OI sweep, DL 611 — from 2026-11-21 22:02). At trigger minute m the newest USABLE OI
      row is the one stamped m − 1 (the §2 lag rule: a row stamped at the sweep START is usable from the next minute); its oi_chg_5m is
      the 5-min OI change; the mark return over the same span = mark(m − 1) ÷ mark(m − 6) − 1 (the last mark row ≤ each minute, ≤ 2 min
      old). Three keys, each with its own episodes:
        H7a  OI ≤ −3 % AND mark ≤ −2 % → LONG     H7b  OI ≤ −3 % AND mark ≥ +2 % → SHORT     H7c  OI ≥ +5 % AND mark ≥ +2 % → SHORT
  Conditioned variants (key + "|c"): H1, H2, H4, H6, H7a / b / c AND the full-book $ within 0.5 % on the side that absorbs our adverse
      move (LONG: bid · SHORT: ask) at minute m (the newest fb row ≤ m, ≤ 2 min old) ≥ that side's prior median over the 24 h before
      that row (rows in [row − 24 h, row), ≥ 60 recorded minutes). The OI leg is not used. Each variant has its own episodes (variant hit minutes = parent
      hit minutes where the depth holds). Every parent / variant opener also carries cond_depth_scored / cond_depth_ok.
  family = the key's first token ("H1", "H4", "H6", "H7a" …): the control-exclusion unit (controls.py). Both sides of H3 / H4 / H5 / H6
      share one family (a control near ANY hit of that trigger is excluded — conservative); H7a / b / c are separate keys and families
      (PREREG §5); a "|c" variant shares its parent's family. H7's mark reads carry at most H7_MARK_AGE_MIN (2) minutes; H7 needs no book.
"""
import numpy as np
import pandas as pd

from common import MIN_MS

EXCL = ("BTCUSDT", "ETHUSDT")
DEFS = [("H1_liq_long_cascade", "LONG"), ("H2_liq_short_squeeze", "SHORT"), ("H3_imb_bid_heavy_thin", "LONG"), ("H3_imb_ask_heavy_thin", "SHORT"),
        ("H4_ask_vacuum", "LONG"), ("H4_bid_vacuum", "SHORT"), ("H5_taker_buy_climax", "SHORT"), ("H5_taker_sell_climax", "LONG"),
        ("H6_funding_le_-0.5pct", "LONG"), ("H6_funding_ge_+0.1pct", "SHORT"),
        ("H7a_oi_flush_dump", "LONG"), ("H7b_oi_flush_pump", "SHORT"), ("H7c_oi_build_pump", "SHORT")]
CONDITIONED = ("H1_liq_long_cascade", "H2_liq_short_squeeze", "H4_ask_vacuum", "H4_bid_vacuum", "H6_funding_le_-0.5pct",
               "H6_funding_ge_+0.1pct", "H7a_oi_flush_dump", "H7b_oi_flush_pump", "H7c_oi_build_pump")
COND_SUFFIX = "|c"
H7_OI_LAG_MIN = 1
H7_MARK_AGE_MIN = 2
DEPTH_MED_WIN_MIN = 1440
DEPTH_MED_MIN_N = 60
DEPTH_MAX_AGE_MIN = 2


def family(key):
    """the control-exclusion family of a trigger key ("H1", "H7a" …; a "|c" variant shares its parent's). Pure."""
    return str(key).split("|")[0].split("_")[0]


def h7_hits(oi_rows, mark_rows):
    """{key: [trigger minute m]} for one coin. oi_rows = DataFrame(minute, oi_chg_5m) (OI rows, any order); mark_rows = DataFrame(minute,
    mark). Trigger minute m = OI row minute + H7_OI_LAG_MIN (the lag rule). Pure."""
    out = {"H7a_oi_flush_dump": [], "H7b_oi_flush_pump": [], "H7c_oi_build_pump": []}
    if oi_rows is None or not len(oi_rows) or mark_rows is None or not len(mark_rows):
        return out
    o = oi_rows.dropna(subset=["oi_chg_5m"]).sort_values("minute")
    mk = mark_rows.dropna(subset=["mark"]).sort_values("minute")
    mm, mv = mk.minute.to_numpy(np.int64), mk.mark.to_numpy(float)

    def mark_at(t):
        k = np.searchsorted(mm, t, "right") - 1
        ok = (k >= 0) & (t - mm[np.clip(k, 0, None)] <= H7_MARK_AGE_MIN * MIN_MS)
        return np.where(ok, mv[np.clip(k, 0, None)], np.nan)
    q = o.minute.to_numpy(np.int64)
    d_oi = o.oi_chg_5m.to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = (mark_at(q) / mark_at(q - 5 * MIN_MS) - 1) * 100
    m = q + H7_OI_LAG_MIN * MIN_MS
    for key, cond in (("H7a_oi_flush_dump", (d_oi <= -3) & (ret <= -2)), ("H7b_oi_flush_pump", (d_oi <= -3) & (ret >= 2)),
                      ("H7c_oi_build_pump", (d_oi >= 5) & (ret >= 2))):
        out[key] = [int(x) for x in m[np.nan_to_num(cond, nan=False).astype(bool)]]
    return out


class Depth:
    """one coin's full-book depth within 0.5 % per side, with the causal 24-h prior median (rows in [m − 24 h, m), ≥ 60 rows)."""

    def __init__(self, fb_rows):
        f = fb_rows.sort_values("minute") if fb_rows is not None and len(fb_rows) else None
        self.m = f.minute.to_numpy(np.int64) if f is not None else np.array([], np.int64)
        self.v, self.med = {}, {}
        for side in ("bid", "ask"):
            if f is None:
                self.v[side] = self.med[side] = np.array([], float)
                continue
            x = pd.Series(pd.to_numeric(f[f"fb_{side}_usd_05"], errors="coerce").to_numpy(float), index=pd.to_datetime(self.m, unit="ms"))
            r = x.rolling(f"{DEPTH_MED_WIN_MIN}min", closed="left")
            med, cnt = r.median().to_numpy(), r.count().to_numpy()
            self.v[side] = x.to_numpy()
            self.med[side] = np.where(cnt >= DEPTH_MED_MIN_N, med, np.nan)

    def ok(self, minute, side):
        """(scored, holds) at trigger minute m for the absorbing side ('bid' for LONG, 'ask' for SHORT): the newest fb row ≤ m (≤ 2 min
        old) vs the median of the 24 h of rows BEFORE that row's own minute (≥ 60 rows)."""
        k = int(np.searchsorted(self.m, int(minute), "right")) - 1
        if k < 0 or int(minute) - int(self.m[k]) > DEPTH_MAX_AGE_MIN * MIN_MS:
            return False, None
        v, md = self.v[side][k], self.med[side][k]
        if not (v == v and md == md):
            return False, None
        return True, bool(v >= md)


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


def detect(book, mark, coins, fb=None, pos=None, return_hits=False):
    """→ DataFrame(trigger, direction, pair, minute, family, cond_depth_scored, cond_depth_ok) of trigger OPENERS (episodes), parents and
    "|c" variants. coins: {sym: state.Coin} for the liquidation baseline; fb / pos = the DL 611 frames (H7 + the depth condition).
    return_hits → also {key: [(sym, minute)]} of every hit minute (the control exclusion)."""
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
    hits["H6_funding_le_-0.5pct"] = [(s, int(m)) for s, m in zip(lo.sym, lo.minute)]
    hits["H6_funding_ge_+0.1pct"] = [(s, int(m)) for s, m in zip(hi.sym, hi.minute)]
    # H7 — OI flush / build (DL 611 OI rows, hot alts)
    if pos is not None and len(pos):
        po = pos[pos.oi.notna() & ~pos.sym.isin(EXCL)]
        mg = {s_: g for s_, g in mark[mark.sym.isin(set(po.sym))].groupby("sym")}
        for sym, g in po.groupby("sym"):
            for key, ms in h7_hits(g[["minute", "oi_chg_5m"]], mg.get(sym)).items():
                hits[key] += [(sym, m) for m in ms]
    # conditioned variants: parent hit minutes where the absorbing side's full-book depth ≥ its prior 24-h median
    depth = {s_: Depth(g) for s_, g in fb.groupby("sym")} if fb is not None and len(fb) else {}
    dirs = dict(DEFS)

    def cond(sym, m, key):
        dp = depth.get(sym)
        if dp is None:
            return False, None
        return dp.ok(m, "bid" if dirs[key] == "LONG" else "ask")
    for key in CONDITIONED:
        hits[key + COND_SUFFIX] = [(s_, m) for s_, m in hits[key] if cond(s_, m, key)[1] is True]
    out = []
    for key in list(dict(DEFS)) + [k + COND_SUFFIX for k in CONDITIONED]:
        d = dirs[key.split("|")[0]]
        ep = episodes(hits[key], gap_min=24 * 60 if key.startswith("H6") else 60)
        out += [(key, d, s_, m) for s_, m in ep]
    T = pd.DataFrame(out, columns=["trigger", "direction", "pair", "minute"])
    T["family"] = T.trigger.map(family)
    cd = [cond(r.pair, r.minute, r.trigger.split("|")[0]) if r.trigger.split("|")[0] in CONDITIONED else (False, None)
          for r in T.itertuples()]
    T["cond_depth_scored"] = [c[0] for c in cd]
    T["cond_depth_ok"] = [c[1] for c in cd]
    T["signal_time"] = pd.to_datetime(T.minute + MIN_MS, unit="ms", utc=True)
    T["window15"] = T.signal_time.dt.floor("15min").dt.strftime("%Y-%m-%dT%H:%M")
    return (T, hits) if return_hits else T
