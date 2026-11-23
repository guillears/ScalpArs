"""Route B weekly-scan PREREG build item 7 — the Q2 theory-line variables (§4), exactly as written. No network."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import q2 as Q           # noqa: E402
import state as ST       # noqa: E402

M = 60_000
T0 = 1_795_000_000_000 // M * M


def BASE(lm):
    """a deterministic NON-constant liquidation baseline (100 … 892 $) — a tie with the quantile never decides a line."""
    return 100.0 + (np.asarray(lm) // M % 7919) / 10


def _coin(liq_short=None, liq_long=None, n=12_000, fb_ask=None, fb_bid=None):
    mins = T0 + np.arange(n) * M
    book = pd.DataFrame(dict(sym="A", minute=mins[:1], n_book=1.0, buy_usd=0.0, sell_usd=0.0))
    mark = pd.DataFrame(dict(sym="A", minute=mins[:1], mark=1.0, index=1.0, funding=0.0))
    lm = mins[::30]
    ls = BASE(lm) if liq_short is None else liq_short(lm)
    ll = BASE(lm) if liq_long is None else liq_long(lm)
    liq = pd.DataFrame(dict(sym="A", minute=lm, liq_long_usd=ll, liq_short_usd=ls))
    fb = None
    if fb_ask is not None:
        fb = pd.DataFrame(dict(sym="A", minute=mins, fb_ask_usd_05=fb_ask(mins), fb_bid_usd_05=fb_bid(mins)))
    return ST.Coin("A", book, mark, liq, fb, None), mins


def test_s1a_p99_over_7_days_before_s():
    big = lambda lm: np.where(lm == lm[-1], 50_000.0, 1000.0)            # noqa: E731 — the last liquidation is a burst
    c, mins = _coin(liq_short=big)
    s = int(c.lm[-1])
    o = Q.lines(c, s, {})
    assert o["q2_S1a"] == 1.0 and o["q2_liq_short_5_p99_7d"] == pytest.approx(1000.0)
    o2 = Q.lines(c, s - 31 * M, {})                                      # a normal minute → not a burst
    assert o2["q2_S1a"] == 0.0


def test_p99_window_excludes_older_than_7_days_and_thin_samples():
    K = lambda lm: np.full(len(lm), 1000.0)                              # noqa: E731
    c, mins = _coin(n=600, liq_short=K)                                  # 20 liquidation minutes, all of them before s
    assert Q.liq_pct_before(c, "short", 5, int(mins[-1]) + M, 0.99) == pytest.approx(1000.0)
    assert math.isnan(Q.liq_pct_before(c, "short", 5, int(mins[300]), 0.99))      # only 10 before s → unscored
    old = lambda lm: np.where(lm < T0 + Q.DAY_MS, 9e9, 1000.0)                # noqa: E731 — 9e9 only before the 7-day window
    c2, mins2 = _coin(liq_short=old)
    s = int(mins2[-1])                                                   # > 7 d after the 9e9 values
    assert Q.liq_pct_before(c2, "short", 5, s, 0.99) == pytest.approx(1000.0)


def test_s1b_s1c_s4b_from_state_and_boundaries():
    c, mins = _coin(n=50)
    s = int(mins[-1])
    o = Q.lines(c, s, dict(oi_chg_15m=0.5, rb_ret_15m_pct=0.1, rb_taker_buy_share_15=0.45, rb_taker_buy_share_5=0.55))
    assert (o["q2_S1b"], o["q2_S1c"], o["q2_S4b"]) == (1.0, 1.0, 1.0)                  # ≤ 0.45 and ≥ 0.55 inclusive
    o = Q.lines(c, s, dict(oi_chg_15m=0.5, rb_ret_15m_pct=-0.1, rb_taker_buy_share_15=0.4501, rb_taker_buy_share_5=0.5499))
    assert (o["q2_S1b"], o["q2_S1c"], o["q2_S4b"]) == (0.0, 0.0, 0.0)
    assert math.isnan(Q.lines(c, s, dict(oi_chg_15m=math.nan, rb_ret_15m_pct=-0.3))["q2_S1b"])   # strict: one unscored leg → NaN


def test_s2a_strict_and_thresholds():
    burst_long = lambda lm: np.where(lm == lm[-1], 50_000.0, BASE(lm))   # noqa: E731
    c, mins = _coin(liq_long=burst_long)
    s = int(c.lm[-1]) + 5 * M
    assert Q.lines(c, s, dict(rb_ret_15m_pct=-10.0))["q2_S2a"] == 1.0                    # ≤ −10 inclusive, long burst ≥ p99
    assert Q.lines(c, s, dict(rb_ret_15m_pct=-9.99))["q2_S2a"] == 0.0
    c2, mins2 = _coin(n=50)                                                              # baseline too thin → the liq leg unscored
    assert math.isnan(Q.lines(c2, int(mins2[-1]), dict(rb_ret_15m_pct=-3.0))["q2_S2a"])  # strict: no leg decides alone
    shorts = lambda lm: np.where(lm == lm[-1], 50_000.0, BASE(lm))       # noqa: E731
    c3, _ = _coin(liq_short=shorts)
    assert Q.lines(c3, int(c3.lm[-1]) + 5 * M, dict(rb_ret_15m_pct=-12.0))["q2_S2a"] == 0.0   # a SHORT burst is not the long leg


def test_s1a_reads_short_liquidations_only():
    longs = lambda lm: np.where(lm == lm[-1], 50_000.0, BASE(lm))        # noqa: E731
    shorts = lambda lm: np.where(lm == lm[-1], 1.0, BASE(lm))            # noqa: E731 — the short side is quiet
    c, _ = _coin(liq_long=longs, liq_short=shorts)
    assert Q.lines(c, int(c.lm[-1]), {})["q2_S1a"] == 0.0


def test_quantiles_and_min_n():
    vals = lambda lm: np.arange(len(lm), dtype=float) + 1                                # noqa: E731 — 1, 2, 3 …
    c, mins = _coin(liq_short=vals)
    s = int(c.lm[-1]) + M
    m, v = Q.liq_series(c, "short", 5)
    x = v[(m >= s - Q.P_DAYS * Q.DAY_MS) & (m < s)]
    assert Q.liq_pct_before(c, "short", 5, s, 0.99) == pytest.approx(np.quantile(x[x > 0], 0.99))
    assert Q.liq_pct_before(c, "short", 15, s, 0.90) == pytest.approx(np.quantile(Q.liq_series(c, "short", 15)[1][(m >= s - Q.P_DAYS * Q.DAY_MS) & (m < s)], 0.90))
    c20, m20 = _coin(n=19 * 30 + 1)                                                      # exactly 20 liquidation minutes before s
    s20 = int(c20.lm[-1]) + M
    assert Q.P_MIN_N == 20 and not math.isnan(Q.liq_pct_before(c20, "short", 5, s20, 0.99))
    assert math.isnan(Q.liq_pct_before(c20, "short", 5, int(c20.lm[-1]), 0.99))         # 19 before → unscored


def test_s2b_s4a_prior_median_before_the_read_row():
    ask = lambda m: np.where(m >= T0 + 1500 * M, 1.0, 10.0)                              # noqa: E731
    bid = lambda m: np.full(len(m), 5.0)                                                  # noqa: E731
    c, mins = _coin(n=1600, fb_ask=ask, fb_bid=bid)
    s = int(mins[1550])
    o = Q.lines(c, s, c.state_fb(s))
    assert o["q2_S2b"] == 1.0 and o["q2_S4a"] == 1.0                                     # 1 < median 10 · 5 ≥ 5
    assert o["q2_fb_ask_05_prior_median"] == pytest.approx(10.0)                         # 24 h before the row: 1,440 rows, 50 of them 1.0
    assert math.isnan(Q.lines(c, int(mins[30]), c.state_fb(int(mins[30])))["q2_S2b"])   # < 60 prior rows
    assert math.isnan(Q.fb_prior_median(c, "fb_ask_usd_05", int(mins[59])))
    assert not math.isnan(Q.fb_prior_median(c, "fb_ask_usd_05", int(mins[60])))          # exactly 60 rows before → scored
    one_day = lambda m: np.where(m < T0 + 100 * M, 1e9, 1.0)                             # noqa: E731 — older than 24 h: excluded
    c2, mins2 = _coin(n=1600, fb_ask=one_day, fb_bid=bid)
    assert Q.fb_prior_median(c2, "fb_ask_usd_05", int(mins2[1599])) == pytest.approx(1.0)


def test_s3a_s3b_s5_strict():
    c, mins = _coin(n=50)
    o = Q.lines(c, int(mins[-1]), dict(oi_chg_240m=0.2, oi_chg_15m=-0.1))
    assert o["q2_S3a"] == 1.0 and math.isnan(o["q2_S3b"]) and math.isnan(o["q2_S5"])
    assert math.isnan(Q.lines(c, int(mins[-1]), dict(oi_chg_15m=0.3))["q2_S5"])          # strict OR: the unscored leg → NaN
    c2, mins2 = _coin()
    s = int(mins2[-1]) - 7 * M
    assert Q.lines(c2, s, dict(oi_chg_15m=-0.3))["q2_S5"] == 0.0
    assert Q.lines(c2, s, dict(oi_chg_15m=0.3))["q2_S5"] == 1.0
    assert Q.lines(c2, s, dict(oi_chg_240m=0.0))["q2_S3a"] == 0.0                          # > 0 strictly


@pytest.mark.parametrize("ahead_min", [1, 2, 60])
def test_lines_are_causal_on_future_liquidations(ahead_min):
    c, mins = _coin()
    s = int(mins[8000])
    a = Q.lines(c, s, {})
    fut = lambda lm: np.where(lm > s + (ahead_min - 1) * M, 1e12, BASE(lm))              # noqa: E731
    c2, _ = _coin(liq_short=fut, liq_long=fut)
    b = Q.lines(c2, s, {})
    for k in a:
        assert (a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]), k


def test_s2c_cuts_on_rows_with_both_and_strict_logic():
    d = pd.DataFrame(dict(rb_funding_pct=[-0.03, -0.01, 0.0, 0.01, 0.02, 0.05, -9.0], global_ls_acct=[0.8, 1.0, 1.2, 1.4, 1.6, 2.0, np.nan]))
    cuts = Q.s2c_cuts(d)
    assert cuts["n"] == 6 and cuts["funding"] == pytest.approx(pd.Series([-0.03, -0.01, 0.0, 0.01, 0.02, 0.05]).quantile(1 / 3))
    rows = pd.DataFrame(dict(rb_funding_pct=[-0.05, -0.05, 0.05, np.nan, cuts["funding"]],
                             global_ls_acct=[0.5, 3.0, np.nan, 0.5, cuts["global_ls_acct"]]))
    out = Q.s2c(rows, cuts)
    assert list(out.iloc[[0, 1, 4]]) == [1.0, 0.0, 1.0] and out.iloc[2] != out.iloc[2] and out.iloc[3] != out.iloc[3]
    assert Q.s2c(pd.DataFrame(dict(x=[1.0])), cuts).isna().all()                          # missing columns: unscored, no crash


def test_no_coin_is_all_unscored():
    o = Q.lines(None, T0, {"has_fb": False})
    assert all(o[f"q2_{k}"] != o[f"q2_{k}"] for k in Q.LINES)


def test_build_wires_q2_and_market_days():
    """the build stamps every Q2 line on the signal rows and writes the S0 day table (source check: no network run in tests)."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals", "build.py")).read()
    assert "Q2.lines(c, ST.state_minute(T), st, market.covered)" in src and 'ST.market_days(market)' in src and "market_days.csv.gz" in src


def test_frozen_burst_definitions_are_called_exactly(monkeypatch):
    """each liquidation line reads exactly its PREREG side · window · quantile (S1a short 5 p99 · S2a long 15 p99 · S3b short 240 p90 ·
    S5 short 15 p90)."""
    calls = []
    monkeypatch.setattr(Q, "burst", lambda coin, side, w, s, q, covered=None: (calls.append((side, w, q)), (math.nan,) * 3)[1])
    Q.lines(None, T0, {})
    assert calls == [("short", 5, 0.99), ("long", 15, 0.99), ("short", 240, 0.90), ("short", 15, 0.90)]
    assert Q.P_DAYS == 7 and Q.P_MIN_N == 20 and Q.MED_MIN_N == 60 and Q.DAY_MS == 1440 * M


def test_liq_series_keeps_sides_and_windows_apart():
    longs = lambda lm: np.full(len(lm), 7.0)                                             # noqa: E731
    shorts = lambda lm: np.full(len(lm), 3.0)                                            # noqa: E731
    c, _ = _coin(n=600, liq_long=longs, liq_short=shorts)
    assert Q.liq_series(c, "long", 5)[1].max() == 7.0 and Q.liq_series(c, "short", 5)[1].max() == 3.0
    assert Q.liq_series(c, "short", 60)[1].max() == 6.0                                  # two liquidation minutes inside 60 min


def test_zero_sums_are_not_in_the_sample():
    """a liquidation minute with only LONG $ gives a zero SHORT sum: it must not enter the short sample."""
    shorts = lambda lm: np.where(np.arange(len(lm)) % 2 == 0, 0.0, 500.0)               # noqa: E731
    c, mins = _coin(n=2000, liq_short=shorts)
    s = int(mins[-1])
    assert Q.liq_pct_before(c, "short", 5, s, 0.10) == pytest.approx(500.0)


def test_s1b_needs_strictly_positive_legs():
    c, mins = _coin(n=50)
    assert Q.lines(c, int(mins[-1]), dict(oi_chg_15m=0.0, rb_ret_15m_pct=0.5))["q2_S1b"] == 0.0
    assert Q.lines(c, int(mins[-1]), dict(oi_chg_15m=0.5, rb_ret_15m_pct=0.0))["q2_S1b"] == 0.0


def test_prior_median_uses_the_read_row_not_s():
    """the state read an fb row 2 min old: the median window ends at THAT row (rows after it, before s, are not in it)."""
    ask = lambda m: np.where(m >= T0 + 1499 * M, 100.0, 10.0)                            # noqa: E731
    c, mins = _coin(n=1500, fb_ask=ask, fb_bid=lambda m: np.full(len(m), 5.0))
    keep = c.fb.minute < T0 + 1498 * M                                                   # fb ends 2 min before s
    c.fb, c.fm = c.fb[keep].reset_index(drop=True), c.fm[keep]
    s = int(mins[1499])
    r = c.state_fb(s)
    assert r["fb_age_min"] == 2
    o = Q.lines(c, s, r)
    assert o["q2_fb_ask_05_prior_median"] == pytest.approx(10.0)


def test_prior_median_lower_bound_is_24h():
    ask = lambda m: np.where(m < T0 + 59 * M, 1e9, 1.0)                                  # noqa: E731
    c, mins = _coin(n=1600, fb_ask=ask, fb_bid=lambda m: np.full(len(m), 5.0))
    row = int(mins[1499])                                                                # window [row − 1440, row) = minutes 59 … 1498
    assert Q.fb_prior_median(c, "fb_ask_usd_05", row) == pytest.approx(1.0)
    assert Q.fb_prior_median(c, "fb_ask_usd_05", row - M) == pytest.approx(1.0)


def test_outage_window_is_unscored():
    burst_short = lambda lm: np.where(lm == lm[-1], 50_000.0, BASE(lm))                  # noqa: E731
    c, mins = _coin(liq_short=burst_short)
    s = int(c.lm[-1])
    assert Q.lines(c, s, {})["q2_S1a"] == 1.0
    down = lambda a, b: not (a < s - 2 * M < b)                                          # noqa: E731 — the recorder was down at s − 2
    o = Q.lines(c, s, {}, covered=down)
    assert math.isnan(o["q2_S1a"]) and math.isnan(o["q2_S3b"]) and math.isnan(o["q2_S5"])
