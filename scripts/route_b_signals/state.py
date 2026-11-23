"""Route B signal-level loader — the recorded STATE at a signal, with NO LOOK-AHEAD (unit-tested).

A per-minute summary row (minute = UTC minute START) is complete only at minute + 60 s. For a signal at time T the STATE MINUTE is
s = floor_minute(T) − 1 min: the last minute that ended at or before T. Every column below reads only rows with minute ≤ s (rolling
windows end at s; coin "norms" end before the window they normalise). Nothing at or after floor_minute(T) is ever read.

Coin columns (prefix rb_):
  book (the last book row in [s − BOOK_MAX_AGE_MIN, s]): spread_bps · bid/ask $ within 0.2 / 0.5 / 1 % (+ _lb flag when the 20 levels
  reach less than that band: a LOWER bound) · imb_05 · reach · modelled cost bps of a $6k market order each side (linear between the
  recorded $2.5k / $10k walks; NaN + cost_lb when the $10k walk did not fit in 20 levels) · book_age_min
  tape (only minutes with the book recorded = the coin was on the hot list): taker-buy share 5 / 15 min · traded $ 15 min ÷ the coin's
  own norm (15 × the median recorded minute before the window, ≥ 60 minutes of history)
  liquidations (market-wide stream, every coin): long / short $ last 5 / 15 min · burst flag = the 5-min sum ≥ the coin's own 99th pct of
  non-zero 5-min sums over minutes BEFORE s (≥ 20 of them)
  mark: funding % (current rate) · mark-index premium % · coin return 15 / 60 min (mark)
DL 611 coin columns (build item 1 of the weekly-scan PREREG; recorded from 2026-11-21 22:02, ≤ 30 priority coins for the full book, hot
coins for OI / positioning). NaN = UNSCORED (never "rest"); has_fb / has_oi / has_ratio = the coverage flag of each family:
  full book (the last fb row in [s − BOOK_MAX_AGE_MIN, s], the same rule as the 20-level book): fb_spread_bps · fb_reach_bid_pct /
  fb_reach_ask_pct · fb_bid_usd_<01|02|05|1|2|5> / fb_ask_usd_… · fb_cost_<buy|sell>_<1000|3000|6000|10000|25000> (bps) · fb_imb_05
  (derived: (bid − ask) ÷ (bid + ask) within 0.5 %) · fb_age_min
  open interest — LAG RULE: the recorder stamps an OI row at its sweep START (the read lands up to ~60 s later), so a row is usable only
  if its minute ≤ s − 1. oi_chg_1m / oi_chg_5m = the recorder's live % of the newest usable row (≤ OI_MAX_AGE_MIN old) · oi_chg_<15|60|240>m
  (derived: % of that row's oi vs the newest row ≤ its minute − w, at most OI_BACK_TOL_MIN older) · oi_age_min
  positioning — LAG RULE: a 5-min ratio row is usable only if ratio_ts + 10 min ≤ T AND its row minute ≤ s − 1 (stamped at the sweep
  start, like OI); rows are de-duplicated per ratio_ts (the earliest recorded minute = when it was first seen). The newest usable one, at
  most RATIO_MAX_AGE_MIN from ratio_ts to T: top_ls_acct · top_ls_pos · global_ls_acct · taker_bs_ratio · each <x>_chg60_pct (derived:
  % vs the usable row whose ratio_ts is nearest ratio_ts − 60 min, within ± RATIO_BACK_TOL_MIN) · ratio_age_min (T − ratio_ts, in
  FRACTIONAL minutes from T; fb_age_min / oi_age_min are whole minutes before s)
  The full-book bands carry fb_<side>_usd_<t>_lb = True when the trusted full book reached less than that band (a LOWER bound).
  Every family emits its full key set (NaN when unscored) so the schema never depends on the data.
Market columns (prefix mk_, the same state minute): breadth % of coins whose mark is up over 5 / 15 / 60 min · market liquidations long /
short 1 / 5 / 15 min · BTC $ within 0.2 % (bid + ask) · BTC taker-buy share 5 / 15 min · BTC return 15 / 60 min · mean funding % and mean
mark-index premium % across coins (build item 6). market_days() = the Q4 S0 row in DAY units (PREREG §4).
"""
import math

import numpy as np
import pandas as pd

from common import BOOK_MAX_AGE_MIN, MIN_MS, NOTIONAL_USD

BANDS = (("02", 0.2), ("05", 0.5), ("1", 1.0))
FB_BAND_TAGS = ("01", "02", "05", "1", "2", "5")
FB_SIZES = (1000, 3000, 6000, 10000, 25000)
OI_LAG_MIN = 1                 # an OI row is usable only at state minute ≥ its minute + 1 (sweep-start stamp; PREREG §2 item 1)
OI_MAX_AGE_MIN = 3             # the newest usable OI row may be at most this many minutes before s (the sweep runs every 60 s)
OI_BACK_TOL_MIN = 5            # the derived w-min OI change: the earlier row may be at most this much older than minute − w
OI_DERIVED = (15, 60, 240)
RATIO_LAG_MS = 10 * MIN_MS     # a positioning row is usable only if ratio_ts + 10 min ≤ T (PREREG §2 item 1)
RATIO_ROW_LAG_MIN = 1          # … and its row minute ≤ s − 1 (sweep-start stamp, PREREG §2 item 1)
RATIO_MAX_AGE_MIN = 30         # … and T − ratio_ts ≤ 30 min (the sweep runs every 5 min; older = unscored)
RATIO_BACK_TOL_MIN = 10        # the 60-min change: the earlier ratio row's ratio_ts within ± 10 min of ratio_ts − 60 min
RATIOS = ("top_ls_acct", "top_ls_pos", "global_ls_acct", "taker_bs_ratio")


def state_minute(T_ms):
    return (int(T_ms) // MIN_MS) * MIN_MS - MIN_MS


def cost_at(c2500, c10000, usd=NOTIONAL_USD):
    """bps for `usd` by linear interpolation between the recorded walks; (value, lower_bound_flag)."""
    if c2500 is None or not np.isfinite(c2500):
        return math.nan, True
    if c10000 is None or not np.isfinite(c10000):
        return math.nan, True
    return float(c2500 + (c10000 - c2500) * (min(usd, 10000) - 2500) / 7500), False


class Coin:
    """one coin's recorded rows (book / mark / liq), sorted by minute, with prefix sums for O(log n) window reads."""

    def __init__(self, sym, book, mark, liq, fb=None, pos=None):
        self.sym = sym
        self.fb = (fb if fb is not None else pd.DataFrame(columns=["minute"])).sort_values("minute").reset_index(drop=True)
        self.fm = self.fb.minute.to_numpy(np.int64)
        pos = pos if pos is not None else pd.DataFrame(columns=["minute", "oi", "ratio_ts"])
        o = pos[pos["oi"].notna()] if "oi" in pos.columns else pos.iloc[:0]
        self.oi = o.sort_values("minute").reset_index(drop=True)
        self.om = self.oi.minute.to_numpy(np.int64)
        rt = pos[pos["ratio_ts"].notna()] if "ratio_ts" in pos.columns else pos.iloc[:0]
        self.rt = rt.sort_values(["ratio_ts", "minute"]).drop_duplicates("ratio_ts", keep="first").reset_index(drop=True)
        self.rt_m = self.rt.minute.to_numpy(np.int64) if len(self.rt) else np.array([], np.int64)
        self.rt_ts = self.rt.ratio_ts.to_numpy(float) if len(self.rt) else np.array([], float)
        self.book = book.sort_values("minute").reset_index(drop=True)
        self.mark = mark.sort_values("minute").reset_index(drop=True)
        self.liq = liq.sort_values("minute").reset_index(drop=True)
        self.bm = self.book.minute.to_numpy(np.int64)
        self.mm = self.mark.minute.to_numpy(np.int64)
        self.lm = self.liq.minute.to_numpy(np.int64)
        b, s = self.book.buy_usd.fillna(0).to_numpy(float), self.book.sell_usd.fillna(0).to_numpy(float)
        self.cb, self.cs = np.concatenate([[0], np.cumsum(b)]), np.concatenate([[0], np.cumsum(s)])
        self.flow = b + s
        self.cll = np.concatenate([[0], np.cumsum(self.liq.liq_long_usd.fillna(0).to_numpy(float))])
        self.cls = np.concatenate([[0], np.cumsum(self.liq.liq_short_usd.fillna(0).to_numpy(float))])
        self._l5 = None

    # sums over minutes in (a, b]
    def _win(self, arr_m, cum, a, b):
        i, j = np.searchsorted(arr_m, a, "right"), np.searchsorted(arr_m, b, "right")
        return float(cum[j] - cum[i]), int(j - i)

    def liq5_series(self):
        """5-min long / short liquidation sums at every minute that had a liquidation of that coin (the baseline sample)."""
        if self._l5 is None:
            m = self.lm
            j = np.searchsorted(m, m, "right")
            i = np.searchsorted(m, m - 5 * MIN_MS, "right")
            self._l5 = (m, self.cll[j] - self.cll[i], self.cls[j] - self.cls[i])
        return self._l5

    def p99_before(self, side, s):
        """99th pct of the coin's non-zero 5-min liquidation sums at minutes < s (≥ 20 of them), else NaN."""
        m, l5, s5 = self.liq5_series()
        v = (l5 if side == "long" else s5)[m < s]
        v = v[v > 0]
        return float(np.quantile(v, 0.99)) if len(v) >= 20 else math.nan

    def state(self, T_ms):
        s = state_minute(T_ms)
        r = {}
        # ── book (last row ≤ s, at most BOOK_MAX_AGE_MIN old)
        k = np.searchsorted(self.bm, s, "right") - 1
        if k >= 0 and s - self.bm[k] <= BOOK_MAX_AGE_MIN * MIN_MS:
            b = self.book.iloc[k]
            r["rb_book_age_min"] = (s - int(self.bm[k])) // MIN_MS
            for c in ("spread_bps", "imb_05", "reach_bid_pct", "reach_ask_pct", "mid"):
                r[f"rb_{c}"] = float(b[c])
            for tag, pct in BANDS:
                for side in ("bid", "ask"):
                    r[f"rb_{side}_usd_{tag}"] = float(b[f"{side}_usd_{tag}"])
                    reach = b[f"reach_{side}_pct"]
                    r[f"rb_{side}_usd_{tag}_lb"] = bool(not (reach >= pct)) if reach == reach else True
            for side in ("buy", "sell"):
                v, lb = cost_at(b[f"cost_{side}_2500"], b[f"cost_{side}_10000"])
                r[f"rb_cost6k_{side}_bps"] = v
                r[f"rb_cost6k_{side}_lb"] = lb
                r[f"rb_cost2500_{side}_bps"] = float(b[f"cost_{side}_2500"])
            r["rb_has_book"] = True
        else:
            r["rb_has_book"] = False
        # ── tape (book-recorded minutes only)
        for w in (5, 15):
            bs, n = self._win(self.bm, self.cb, s - w * MIN_MS, s)
            ss, _ = self._win(self.bm, self.cs, s - w * MIN_MS, s)
            r[f"rb_taker_buy_share_{w}"] = bs / (bs + ss) if (bs + ss) > 0 and n >= max(3, w // 2) else math.nan
            r[f"rb_tape_minutes_{w}"] = n
            if w == 15:
                r["rb_traded_usd_15"] = bs + ss if n >= 8 else math.nan
        j0 = np.searchsorted(self.bm, s - 15 * MIN_MS, "right")          # norm: recorded minutes BEFORE the 15-min window
        hist = self.flow[:j0]
        med = float(np.median(hist)) if len(hist) >= 60 else math.nan
        r["rb_traded_norm_usd_15"] = med * 15 if med == med else math.nan
        r["rb_traded_vs_norm_15"] = (r["rb_traded_usd_15"] / r["rb_traded_norm_usd_15"]
                                     if (r["rb_traded_norm_usd_15"] or 0) > 0 and r["rb_traded_usd_15"] == r["rb_traded_usd_15"] else math.nan)
        # ── liquidations (all coins)
        for w in (5, 15):
            r[f"rb_liq_long_{w}"] = self._win(self.lm, self.cll, s - w * MIN_MS, s)[0]
            r[f"rb_liq_short_{w}"] = self._win(self.lm, self.cls, s - w * MIN_MS, s)[0]
        for side in ("long", "short"):
            p = self.p99_before(side, s)
            r[f"rb_liq_{side}_p99_5"] = p
            r[f"rb_liq_{side}_burst"] = (bool(r[f"rb_liq_{side}_5"] >= p and r[f"rb_liq_{side}_5"] > 0) if p == p else None)
        # ── mark / funding
        k = np.searchsorted(self.mm, s, "right") - 1
        if k >= 0 and s - self.mm[k] <= 5 * MIN_MS:
            m = self.mark.iloc[k]
            r["rb_has_mark"] = True
            r["rb_funding_pct"] = float(m["funding"]) * 100 if m["funding"] == m["funding"] else math.nan
            r["rb_premium_pct"] = (float(m["mark"]) / float(m["index"]) - 1) * 100 if (m["index"] or 0) > 0 else math.nan
            r["rb_mark"] = float(m["mark"])
            for w in (15, 60):
                kk = np.searchsorted(self.mm, s - w * MIN_MS, "right") - 1
                ok = kk >= 0 and (s - w * MIN_MS) - self.mm[kk] <= 5 * MIN_MS
                r[f"rb_ret_{w}m_pct"] = (float(m["mark"]) / float(self.mark.mark.iloc[kk]) - 1) * 100 if ok else math.nan
        else:
            r["rb_has_mark"] = False
        r.update(self.state_fb(s))
        r.update(self.state_oi(s))
        r.update(self.state_ratio(int(T_ms), s))
        return r

    # ── DL 611 families (build item 1) — every one reads only rows that were public by T
    def state_fb(self, s):
        k = np.searchsorted(self.fm, s, "right") - 1
        if k < 0 or s - self.fm[k] > BOOK_MAX_AGE_MIN * MIN_MS:
            return {"has_fb": False, **{c: math.nan for c in FB_KEYS}}
        b = self.fb.iloc[k]
        r = {"has_fb": True, "fb_age_min": (s - int(self.fm[k])) // MIN_MS}
        for c in ("fb_spread_bps", "fb_reach_bid_pct", "fb_reach_ask_pct"):
            r[c] = _num(b.get(c))
        for t in FB_BAND_TAGS:
            pct = FB_BAND_PCT[t]
            for side in ("bid", "ask"):
                r[f"fb_{side}_usd_{t}"] = _num(b.get(f"fb_{side}_usd_{t}"))
                reach = r[f"fb_reach_{side}_pct"]
                r[f"fb_{side}_usd_{t}_lb"] = bool(not (reach >= pct)) if reach == reach else math.nan
        for z in FB_SIZES:
            for side in ("buy", "sell"):
                r[f"fb_cost_{side}_{z}"] = _num(b.get(f"fb_cost_{side}_{z}"))
        bb, aa = r["fb_bid_usd_05"], r["fb_ask_usd_05"]
        r["fb_imb_05"] = (bb - aa) / (bb + aa) if (bb == bb and aa == aa and bb + aa > 0) else math.nan
        return r

    def oi_row(self, s):
        """index of the newest USABLE OI row at state minute s (minute ≤ s − OI_LAG_MIN, ≤ OI_MAX_AGE_MIN before s) or −1."""
        k = np.searchsorted(self.om, s - OI_LAG_MIN * MIN_MS, "right") - 1
        return k if (k >= 0 and s - self.om[k] <= OI_MAX_AGE_MIN * MIN_MS) else -1

    def oi_chg(self, s, w):
        """% change of OI over w minutes ending at the newest usable row (NaN when unscored). Pure on the recorded rows."""
        k = self.oi_row(s)
        if k < 0:
            return math.nan
        q = int(self.om[k])
        j = np.searchsorted(self.om, q - w * MIN_MS, "right") - 1
        if j < 0 or (q - w * MIN_MS) - self.om[j] > OI_BACK_TOL_MIN * MIN_MS:
            return math.nan
        a, b = float(self.oi.oi.iloc[j]), float(self.oi.oi.iloc[k])
        return (b / a - 1) * 100 if a > 0 else math.nan

    def state_oi(self, s):
        k = self.oi_row(s)
        if k < 0:
            return {"has_oi": False, **{c: math.nan for c in OI_KEYS}}
        o = self.oi.iloc[k]
        r = {"has_oi": True, "oi_age_min": (s - int(self.om[k])) // MIN_MS, "oi_chg_1m": _num(o.get("oi_chg_1m")),
             "oi_chg_5m": _num(o.get("oi_chg_5m"))}
        for w in OI_DERIVED:
            r[f"oi_chg_{w}m"] = self.oi_chg(s, w)
        return r

    def ratio_rows(self, T_ms, s):
        """the positioning rows public by T: ratio_ts + RATIO_LAG_MS ≤ T and first-seen row minute ≤ s − RATIO_ROW_LAG_MIN (by ratio_ts)."""
        if not len(self.rt):
            return self.rt
        n = int(np.searchsorted(self.rt_ts, T_ms - RATIO_LAG_MS, "right"))
        x = self.rt.iloc[:n]
        return x[self.rt_m[:n] <= s - RATIO_ROW_LAG_MIN * MIN_MS]

    def state_ratio(self, T_ms, s):
        none = {"has_ratio": False, **{c: math.nan for c in RATIO_KEYS}}
        x = self.ratio_rows(T_ms, s)
        if not len(x):
            return none
        last = x.iloc[-1]
        ts = float(last.ratio_ts)
        if T_ms - ts > RATIO_MAX_AGE_MIN * MIN_MS:
            return none
        r = {"has_ratio": True, "ratio_age_min": (T_ms - ts) / MIN_MS}
        tgt = ts - 60 * MIN_MS
        d = (x.ratio_ts.to_numpy(float) - tgt)
        ok = np.abs(d) <= RATIO_BACK_TOL_MIN * MIN_MS
        j = int(np.argmin(np.where(ok, np.abs(d), np.inf))) if ok.any() else -1
        for c in RATIOS:
            v = _num(last.get(c))
            r[c] = v
            v0 = _num(x.iloc[j].get(c)) if j >= 0 else math.nan
            r[f"{c}_chg60_pct"] = (v / v0 - 1) * 100 if (v == v and v0 == v0 and v0 > 0) else math.nan
        return r


class Market:
    """per-minute MARKET row (computed once on the recorded minute grid; trailing windows only)."""

    def __init__(self, mark, liq, btc):
        m3 = mark.drop_duplicates(["minute", "sym"], keep="last").set_index(["minute", "sym"])
        piv = m3["mark"].unstack()                                  # one reshape per column (the 3 pivot_tables cost 3× time / memory)
        grid = np.arange(piv.index.min(), piv.index.max() + MIN_MS, MIN_MS)
        up = piv.notna().any(axis=1).reindex(grid, fill_value=False).to_numpy(bool)   # a minute in which the recorder wrote ANY mark
        piv = piv.reindex(grid).ffill(limit=5)
        f = pd.DataFrame(index=grid)
        for w in (5, 15, 60):
            prev = piv.shift(w)
            both = piv.notna() & prev.notna()
            f[f"mk_breadth_up_{w}"] = ((piv > prev) & both).sum(axis=1) / both.sum(axis=1).replace(0, np.nan) * 100
        lq = liq.groupby("minute")[["liq_long_usd", "liq_short_usd"]].sum().reindex(grid).fillna(0)
        lq[~up] = np.nan                                            # a recorder outage is UNSCORED, never "zero liquidations"
        for w in (1, 5, 15):
            f[f"mk_liq_long_{w}"] = lq.liq_long_usd.rolling(w, min_periods=w).sum()
            f[f"mk_liq_short_{w}"] = lq.liq_short_usd.rolling(w, min_periods=w).sum()
        bt = btc.set_index("minute").reindex(grid)
        f["mk_btc_depth_02_usd"] = (bt.bid_usd_02 + bt.ask_usd_02).ffill(limit=BOOK_MAX_AGE_MIN)
        f["mk_btc_bid_02_usd"] = bt.bid_usd_02.ffill(limit=BOOK_MAX_AGE_MIN)
        f["mk_btc_ask_02_usd"] = bt.ask_usd_02.ffill(limit=BOOK_MAX_AGE_MIN)
        for w in (5, 15):
            b, s_ = bt.buy_usd.rolling(w, min_periods=max(3, w // 2)).sum(), bt.sell_usd.rolling(w, min_periods=max(3, w // 2)).sum()
            f[f"mk_btc_taker_buy_share_{w}"] = b / (b + s_)
        if "BTCUSDT" in piv.columns:
            for w in (15, 60):
                f[f"mk_btc_ret_{w}m_pct"] = (piv["BTCUSDT"] / piv["BTCUSDT"].shift(w) - 1) * 100
        # S0 (weekly-scan PREREG §4 Q4, build item 6): mean funding and mean mark-index premium across every coin with a mark row
        # (the last row ≤ the minute, ≤ 5 min old — the same carry as the breadth columns)
        # funding = the raw per-settlement rate, 1 h and 8 h settlement coins mixed (as H6), BTC / ETH included
        fu = m3["funding"].astype("float32").unstack().reindex(grid).ffill(limit=5)
        f["mk_funding_mean_pct"] = fu.mean(axis=1) * 100
        ratio = (m3["mark"] / m3["index"].where(m3["index"] > 0)).unstack().reindex(grid).ffill(limit=5)   # per ROW (never a mark
        f["mk_premium_mean_pct"] = (ratio.mean(axis=1) - 1) * 100                                              # paired with a stale index)
        self.f = f
        self.up_grid, self.up = np.asarray(grid, np.int64), up
        self.up_cum = np.concatenate([[0], np.cumsum(~up)])
        self.grid = grid

    def covered(self, a, b):
        """True when the recorder wrote marks in EVERY minute of (a, b] (inside the recorded grid) — an outage minute makes a window
        unscored, never "zero". Pure on the grid."""
        g = self.up_grid
        if not len(g) or a + MIN_MS < g[0] or b > g[-1]:
            return False
        i, j = np.searchsorted(g, a, "right"), np.searchsorted(g, b, "right")
        return bool(j > i and self.up_cum[j] - self.up_cum[i] == 0)

    def state(self, T_ms):
        s = state_minute(T_ms)
        if s not in self.f.index:
            return {"mk_has_row": False}
        r = self.f.loc[s].to_dict()
        r["mk_has_row"] = True
        return r


FB_BAND_PCT = {"01": 0.1, "02": 0.2, "05": 0.5, "1": 1.0, "2": 2.0, "5": 5.0}
FB_KEYS = (["fb_age_min", "fb_spread_bps", "fb_reach_bid_pct", "fb_reach_ask_pct", "fb_imb_05"]
           + [f"fb_{sd}_usd_{t}{x}" for t in FB_BAND_TAGS for sd in ("bid", "ask") for x in ("", "_lb")]
           + [f"fb_cost_{sd}_{z}" for z in FB_SIZES for sd in ("buy", "sell")])
OI_KEYS = ["oi_age_min", "oi_chg_1m", "oi_chg_5m"] + [f"oi_chg_{w}m" for w in OI_DERIVED]
RATIO_KEYS = ["ratio_age_min"] + list(RATIOS) + [f"{c}_chg60_pct" for c in RATIOS]
DL611_FLAGS = {"has_fb": False, "has_oi": False, "has_ratio": False}


S0_COLS = {"breadth_up_15": "mk_breadth_up_15", "breadth_up_60": "mk_breadth_up_60", "liq_long_15": "mk_liq_long_15",
           "liq_short_15": "mk_liq_short_15", "btc_depth_02_usd": "mk_btc_depth_02_usd", "btc_taker_buy_share_15": "mk_btc_taker_buy_share_15",
           "funding_mean_pct": "mk_funding_mean_pct", "premium_mean_pct": "mk_premium_mean_pct"}


def market_days(market, start_ms=None, end_ms=None):
    """the Q4 MARKET row in DAY units (PREREG §4 S0): per 15-min window the value at the window's LAST minute (its state at the window's
    close), then the daily MEAN over the windows that have it. → DataFrame(day, s0_<x> …, s0_<x>_n windows, n_windows, complete).
    complete = every S0 column is scored in all 96 windows of the day (a recorder outage, a late hour or a partial day → False; Q4 reads
    complete days only). Pure on the Market frame."""
    f = market.f
    if start_ms is not None:
        f = f[f.index >= start_ms]
    if end_ms is not None:
        f = f[f.index < end_ms]
    if not len(f):
        return pd.DataFrame(columns=["day", "n_windows", "complete"])
    idx = np.asarray(f.index, np.int64)
    last = f[(idx // MIN_MS) % 15 == 14]                       # the last minute of each 15-min window
    d = pd.DataFrame({f"s0_{k}": (last[v] if v in last.columns else np.nan) for k, v in S0_COLS.items()})   # every column, always
    d["day"] = pd.to_datetime(np.asarray(last.index, np.int64), unit="ms", utc=True).strftime("%Y-%m-%d")
    g = d.groupby("day")
    out = g.mean(numeric_only=True)
    for c in [c for c in d.columns if c.startswith("s0_")]:
        out[c + "_n"] = g[c].count()
    out["n_windows"] = g.size()
    out = out.reset_index()
    ncols = [c for c in out.columns if c.startswith("s0_") and c.endswith("_n")]
    out["complete"] = (out[ncols] == 96).all(axis=1) if ncols else False
    return out


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def coins_from(book, mark, liq, syms, fb=None, pos=None):
    bg = {s: g for s, g in book[book.sym.isin(syms)].groupby("sym")}
    mg = {s: g for s, g in mark[mark.sym.isin(syms)].groupby("sym")}
    lg = {s: g for s, g in liq[liq.sym.isin(syms)].groupby("sym")}
    fg = {s: g for s, g in fb[fb.sym.isin(syms)].groupby("sym")} if fb is not None and len(fb) else {}
    pg = {s: g for s, g in pos[pos.sym.isin(syms)].groupby("sym")} if pos is not None and len(pos) else {}
    e_b, e_m, e_l = book.iloc[:0], mark.iloc[:0], liq.iloc[:0]
    return {s: Coin(s, bg.get(s, e_b), mg.get(s, e_m), lg.get(s, e_l), fg.get(s), pg.get(s)) for s in syms}
