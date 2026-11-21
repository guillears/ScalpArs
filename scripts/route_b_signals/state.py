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
Market columns (prefix mk_, the same state minute): breadth % of coins whose mark is up over 5 / 15 / 60 min · market liquidations long /
short 1 / 5 / 15 min · BTC $ within 0.2 % (bid + ask) · BTC taker-buy share 5 / 15 min · BTC return 15 / 60 min.
"""
import math

import numpy as np
import pandas as pd

from common import BOOK_MAX_AGE_MIN, MIN_MS, NOTIONAL_USD

BANDS = (("02", 0.2), ("05", 0.5), ("1", 1.0))


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

    def __init__(self, sym, book, mark, liq):
        self.sym = sym
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
        return r


class Market:
    """per-minute MARKET row (computed once on the recorded minute grid; trailing windows only)."""

    def __init__(self, mark, liq, btc):
        piv = mark.pivot_table(index="minute", columns="sym", values="mark", aggfunc="last")
        grid = np.arange(piv.index.min(), piv.index.max() + MIN_MS, MIN_MS)
        piv = piv.reindex(grid).ffill(limit=5)
        f = pd.DataFrame(index=grid)
        for w in (5, 15, 60):
            prev = piv.shift(w)
            both = piv.notna() & prev.notna()
            f[f"mk_breadth_up_{w}"] = ((piv > prev) & both).sum(axis=1) / both.sum(axis=1).replace(0, np.nan) * 100
        lq = liq.groupby("minute")[["liq_long_usd", "liq_short_usd"]].sum().reindex(grid).fillna(0)
        for w in (1, 5, 15):
            f[f"mk_liq_long_{w}"] = lq.liq_long_usd.rolling(w, min_periods=1).sum()
            f[f"mk_liq_short_{w}"] = lq.liq_short_usd.rolling(w, min_periods=1).sum()
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
        self.f = f
        self.grid = grid

    def state(self, T_ms):
        s = state_minute(T_ms)
        if s not in self.f.index:
            return {"mk_has_row": False}
        r = self.f.loc[s].to_dict()
        r["mk_has_row"] = True
        return r


def coins_from(book, mark, liq, syms):
    bg = {s: g for s, g in book[book.sym.isin(syms)].groupby("sym")}
    mg = {s: g for s, g in mark[mark.sym.isin(syms)].groupby("sym")}
    lg = {s: g for s, g in liq[liq.sym.isin(syms)].groupby("sym")}
    e_b, e_m, e_l = book.iloc[:0], mark.iloc[:0], liq.iloc[:0]
    return {s: Coin(s, bg.get(s, e_b), mg.get(s, e_m), lg.get(s, e_l)) for s in syms}
