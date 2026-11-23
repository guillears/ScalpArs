"""Route B weekly-scan PREREG build item 6 — the S0 MARKET columns (mean funding / premium) and the Q4 day table. No network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import state as ST       # noqa: E402

M = 60_000
D0 = 1_795_000_000_000 // (1440 * M) * (1440 * M)


def _market(n=2 * 1440, start=D0):
    mins = start + np.arange(n) * M
    rows = []
    for sym, fund, prem in (("AUSDT", 1e-4, 0.01), ("BUSDT", 3e-4, -0.03), ("BTCUSDT", 2e-4, 0.0)):
        rows.append(pd.DataFrame(dict(sym=sym, minute=mins, mark=1 + prem / 100, index=1.0, funding=fund)))
    mark = pd.concat(rows, ignore_index=True)
    liq = pd.DataFrame(dict(sym="AUSDT", minute=mins[::10], liq_long_usd=100.0, liq_short_usd=50.0))
    btc = pd.DataFrame(dict(minute=mins, bid_usd_02=1e6, ask_usd_02=2e6, buy_usd=3.0, sell_usd=1.0))
    return ST.Market(mark, liq, btc), mins


def test_mean_funding_and_premium():
    mk, mins = _market(n=100)
    r = mk.state(int(mins[50]) + 1000)
    assert r["mk_funding_mean_pct"] == pytest.approx(2e-4 * 100)
    assert r["mk_premium_mean_pct"] == pytest.approx((0.01 - 0.03 + 0.0) / 3, abs=1e-9)


def _day(k):
    return pd.Timestamp(D0 + k * 1440 * M, unit="ms").strftime("%Y-%m-%d")


def test_market_days_window_close_and_completeness():
    mk, mins = _market(n=3 * 1440 - 30)                   # day 3 is missing its last 30 minutes (2 windows)
    d = ST.market_days(mk)
    assert list(d.day) == [_day(0), _day(1), _day(2)]
    assert list(d.n_windows) == [96, 96, 94] and list(d.complete) == [False, True, False]   # day 1: the 60-min breadth warms up
    r = d.iloc[1]
    assert r.s0_btc_depth_02_usd == pytest.approx(3e6) and r.s0_btc_taker_buy_share_15 == pytest.approx(0.75)
    assert r.s0_funding_mean_pct == pytest.approx(0.02) and r.s0_liq_long_15_n == 96


def test_outage_is_unscored_not_zero_liquidations():
    mk, mins = _market(n=3 * 1440)
    keep = (mins < D0 + 1440 * M + 300 * M) | (mins >= D0 + 1440 * M + 600 * M)            # a 5-h recorder outage on day 2
    rows = []
    for sym, fund, prem in (("AUSDT", 1e-4, 0.01), ("BUSDT", 3e-4, -0.03), ("BTCUSDT", 2e-4, 0.0)):
        rows.append(pd.DataFrame(dict(sym=sym, minute=mins[keep], mark=1 + prem / 100, index=1.0, funding=fund)))
    mark = pd.concat(rows, ignore_index=True)
    liq = pd.DataFrame(dict(sym="AUSDT", minute=mins[keep][::10], liq_long_usd=100.0, liq_short_usd=50.0))
    btc = pd.DataFrame(dict(minute=mins[keep], bid_usd_02=1e6, ask_usd_02=2e6, buy_usd=3.0, sell_usd=1.0))
    d = ST.market_days(ST.Market(mark, liq, btc))
    r = d[d.day == _day(1)].iloc[0]
    assert not r.complete and r.s0_liq_long_15_n < 96                                      # never 96 fake zeros
    full = d[d.day == _day(2)].iloc[0]
    assert full.complete and r.s0_liq_long_15 == pytest.approx(full.s0_liq_long_15)         # the mean is not diluted by zeros


def test_market_days_bounds():
    mk, mins = _market()
    d = ST.market_days(mk, start_ms=D0 + 1440 * M)
    assert len(d) == 1 and d.complete.iloc[0]
    assert len(ST.market_days(mk, start_ms=D0 + 10 * 1440 * M)) == 0


def _mk_rows(mins, coins=(("AUSDT", 1e-4, 0.01), ("BUSDT", 3e-4, -0.03), ("BTCUSDT", 2e-4, 0.0)), drop=None):
    rows = []
    for sym, fund, prem in coins:
        mm = mins if drop is None or sym != drop[0] else mins[~np.isin(mins, drop[1])]
        rows.append(pd.DataFrame(dict(sym=sym, minute=mm, mark=1 + prem / 100, index=1.0, funding=fund)))
    return pd.concat(rows, ignore_index=True)


def _btc(mins):
    return pd.DataFrame(dict(minute=mins, bid_usd_02=1e6, ask_usd_02=2e6, buy_usd=3.0, sell_usd=1.0))


def test_up_means_any_coin_and_covered():
    mins = D0 + np.arange(200) * M
    mark = _mk_rows(mins, drop=("AUSDT", mins[100:105]))                              # one coin misses minutes: the recorder was up
    liq = pd.DataFrame(dict(sym="AUSDT", minute=mins[::7], liq_long_usd=100.0, liq_short_usd=50.0))
    mk = ST.Market(mark, liq, _btc(mins))
    assert mk.up.all() and mk.covered(int(mins[90]), int(mins[110]))
    assert not np.isnan(mk.f.loc[int(mins[103]), "mk_liq_long_15"])
    gap = _mk_rows(mins[(mins < mins[100]) | (mins > mins[104])])                     # every coin misses 100–104: an outage
    mk2 = ST.Market(gap, liq, _btc(mins))
    assert not mk2.covered(int(mins[90]), int(mins[110])) and mk2.covered(int(mins[106]), int(mins[150]))
    assert not mk2.covered(int(mins[-1]), int(mins[-1]) + 5 * M)                       # past the grid
    assert np.isnan(mk2.f.loc[int(mins[110]), "mk_liq_long_15"]) and not np.isnan(mk2.f.loc[int(mins[130]), "mk_liq_long_15"])


def test_liq_window_warm_up_is_unscored():
    mins = D0 + np.arange(60) * M
    liq = pd.DataFrame(dict(sym="AUSDT", minute=mins[::3], liq_long_usd=100.0, liq_short_usd=50.0))
    mk = ST.Market(_mk_rows(mins), liq, _btc(mins))
    assert np.isnan(mk.f.loc[int(mins[13]), "mk_liq_long_15"]) and not np.isnan(mk.f.loc[int(mins[14]), "mk_liq_long_15"])


def test_premium_skips_bad_index_and_pairs_mark_with_its_own_row():
    mins = D0 + np.arange(30) * M
    mark = _mk_rows(mins)
    mark.loc[mark.sym == "BUSDT", "index"] = 0.0                                      # no usable index → out of the mean, not −100 %
    mk = ST.Market(mark, pd.DataFrame(columns=["sym", "minute", "liq_long_usd", "liq_short_usd"]), _btc(mins))
    assert mk.state(int(mins[20]) + 1000)["mk_premium_mean_pct"] == pytest.approx((0.01 + 0.0) / 2, abs=1e-9)


def test_funding_carry_is_limited_to_5_minutes():
    mins = D0 + np.arange(40) * M
    mark = _mk_rows(mins, coins=(("AUSDT", 1e-4, 0.0), ("BUSDT", 3e-4, 0.0)))
    mark = mark[~((mark.sym == "BUSDT") & (mark.minute > mins[10]))]                   # BUSDT stops at minute 10
    mk = ST.Market(mark, pd.DataFrame(columns=["sym", "minute", "liq_long_usd", "liq_short_usd"]), _btc(mins))
    assert mk.f.loc[int(mins[15]), "mk_funding_mean_pct"] == pytest.approx(0.02)       # carried ≤ 5 min
    assert mk.f.loc[int(mins[16]), "mk_funding_mean_pct"] == pytest.approx(0.01)       # then out


def test_market_days_mean_and_end_bound():
    mk, mins = _market(n=3 * 1440)
    mk.f.loc[mk.f.index >= D0 + 1440 * M + 720 * M, "mk_btc_depth_02_usd"] = 9e6         # the second half of day 2 differs
    d = ST.market_days(mk)
    r = d[d.day == _day(1)].iloc[0]
    assert r.s0_btc_depth_02_usd == pytest.approx((3e6 * 48 + 9e6 * 48) / 96)           # a MEAN over the windows (not a median)
    e = ST.market_days(mk, end_ms=D0 + 2 * 1440 * M)
    assert list(e.day) == [_day(0), _day(1)]                                             # end is exclusive


def test_s0_columns_always_present():
    mk, mins = _market(n=2 * 1440)
    mk.f = mk.f.drop(columns=["mk_premium_mean_pct"])
    d = ST.market_days(mk)
    assert "s0_premium_mean_pct" in d.columns and not d.complete.any()
