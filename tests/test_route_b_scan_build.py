"""Route B weekly-scan PREREG build items (reports/ROUTE_B_WEEKLY_SCAN_PREREG.md §2) — no network.

Item 1: the DL 611 full-book / OI / positioning state at a signal — the OI lag rule (a row stamped at its sweep START is usable only at
state minute ≥ its minute + 1), the positioning lag rule (ratio_ts + 10 min ≤ T), derived changes, coverage flags, no look-ahead."""
import gzip
import io
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import common as C       # noqa: E402
import sources as SRC    # noqa: E402
import state as ST       # noqa: E402

M = C.MIN_MS
T0 = 1_795_000_000_000 // M * M


def _base(n=400, sym="AAAUSDT"):
    mins = T0 + np.arange(n) * M
    book = pd.DataFrame(dict(sym=sym, minute=mins, n_book=120.0, mid=1.0, mid_hi=1.0, mid_lo=1.0, spread_bps=2.0, bid_usd_02=1e4,
                             ask_usd_02=1e4, bid_usd_05=3e4, ask_usd_05=3e4, bid_usd_1=6e4, ask_usd_1=6e4, imb_05=0.0, reach_bid_pct=2.0,
                             reach_ask_pct=2.0, cost_buy_2500=5.0, cost_sell_2500=5.0, cost_buy_10000=12.0, cost_sell_10000=12.0,
                             n_trades=10.0, buy_usd=1e3, sell_usd=1e3, max_trade_usd=100.0))
    mark = pd.DataFrame(dict(sym=sym, minute=mins, mark=1.0, index=1.0, funding=1e-4))
    liq = pd.DataFrame(dict(sym=sym, minute=mins[:0], liq_long_usd=[], liq_short_usd=[]))
    return mins, book, mark, liq


def _fb(mins, sym="AAAUSDT", seed=3):
    rng = np.random.default_rng(seed)
    n = len(mins)
    d = dict(sym=sym, minute=mins, n_fbook=50.0, fb_spread_bps=rng.random(n), fb_reach_bid_pct=5.0, fb_reach_ask_pct=5.0)
    for t in ST.FB_BAND_TAGS:
        d[f"fb_bid_usd_{t}"] = rng.random(n) * 1e5
        d[f"fb_ask_usd_{t}"] = rng.random(n) * 1e5
    for z in ST.FB_SIZES:
        d[f"fb_cost_buy_{z}"] = rng.random(n) * 10
        d[f"fb_cost_sell_{z}"] = rng.random(n) * 10
    return pd.DataFrame(d)


def _pos(mins, sym="AAAUSDT", seed=4):
    rng = np.random.default_rng(seed)
    n = len(mins)
    oi = 1e6 * np.cumprod(1 + rng.normal(0, 0.001, n))
    p = pd.DataFrame(dict(sym=sym, minute=mins, oi=oi, oi_chg_1m=rng.normal(0, 0.1, n), oi_chg_5m=rng.normal(0, 0.3, n)))
    rts = (mins // (5 * M)) * (5 * M)                     # the period stamp read at that minute (carried forward like features.py)
    for c in ST.RATIOS:
        p[c] = 1 + (rts - T0) / (1000 * M) + (0.1 if c == "top_ls_pos" else 0)
    p["ratio_ts"] = rts.astype(float)
    return p


def _coin(fb=True, pos=True, n=400):
    mins, book, mark, liq = _base(n)
    return ST.Coin("AAAUSDT", book, mark, liq, _fb(mins) if fb else None, _pos(mins) if pos else None), mins


def test_oi_lag_rule_never_reads_the_state_minute_row():
    c, mins = _coin()
    T = int(mins[300]) + 30_000                            # state minute s = mins[299]
    s = ST.state_minute(T)
    r = c.state(T)
    k = int(np.searchsorted(c.om, s, "left"))
    assert r["has_oi"] and r["oi_age_min"] == 1            # the row at s itself is NOT usable (sweep-start stamp): s − 1 is the newest
    assert r["oi_chg_5m"] == pytest.approx(float(c.oi.oi_chg_5m.iloc[k - 1]))
    want = (c.oi.oi.iloc[k - 1] / c.oi.oi.iloc[k - 1 - 60] - 1) * 100
    assert r["oi_chg_60m"] == pytest.approx(want)


def test_oi_stale_or_missing_is_unscored():
    c, mins = _coin()
    keep = c.om < mins[200]                                 # OI stops at minute 200
    c.oi, c.om = c.oi[keep].reset_index(drop=True), c.om[keep]
    r = c.state(int(mins[300]) + 1000)
    assert r["has_oi"] is False and math.isnan(r["oi_chg_5m"])
    r2 = c.state(int(mins[202]) + 1000)                    # s = 201 → newest usable 199 ≤ s − 1, 2 min old → still scored
    assert r2["has_oi"] and math.isnan(r2["oi_chg_240m"])  # no row 240 min back → unscored, never 0


TEN_MIN = 10 * 60_000                                           # the PREREG's frozen lag — hard-coded (never read from the code)


@pytest.mark.parametrize("off_s", [61, 125, 199, 241, 299])
def test_ratio_lag_rule_ts_plus_10_min(off_s):
    c, mins = _coin()
    T = int(mins[300]) + off_s * 1000                       # T at several offsets inside the 5-min cycle
    r = c.state(T)
    newest_public = max(t for t in c.rt.ratio_ts if t + TEN_MIN <= T)
    assert r["has_ratio"] and r["ratio_age_min"] == pytest.approx((T - newest_public) / M)


@pytest.mark.parametrize("delta,usable", [(0, True), (-1, False), (1, True)])
def test_ratio_lag_exact_boundary(delta, usable):
    """one ratio row with ts = X, first seen at minute X: usable exactly from T = X + 10 min, never 1 ms earlier."""
    mins, book, mark, liq = _base(100)
    X = int(mins[40])
    pos = pd.DataFrame(dict(sym="AAAUSDT", minute=[X], oi=[np.nan], oi_chg_1m=[np.nan], oi_chg_5m=[np.nan], ratio_ts=[float(X)],
                            top_ls_acct=[1.3], top_ls_pos=[1.1], global_ls_acct=[0.9], taker_bs_ratio=[1.0]))
    c = ST.Coin("AAAUSDT", book, mark, liq, None, pos)
    r = c.state(X + TEN_MIN + delta)
    assert r["has_ratio"] is usable and ((r["top_ls_acct"] == 1.3) if usable else math.isnan(r["top_ls_acct"]))


def test_ratio_late_read_row_minute_guard():
    """an OLD ratio_ts read late (first seen at row minute s or s − 0) is not usable; seen at s − 1 it is."""
    mins, book, mark, liq = _base(100)
    T = int(mins[60]) + 1000
    s = ST.state_minute(T)
    for seen, usable in ((s, False), (s - M, True)):
        pos = pd.DataFrame(dict(sym="AAAUSDT", minute=[seen], oi=[np.nan], oi_chg_1m=[np.nan], oi_chg_5m=[np.nan],
                                ratio_ts=[float(T - 20 * M)], top_ls_acct=[1.3], top_ls_pos=[1.1], global_ls_acct=[0.9], taker_bs_ratio=[1.0]))
        assert ST.Coin("AAAUSDT", book, mark, liq, None, pos).state(T)["has_ratio"] is usable


def test_ratio_dedup_keeps_first_seen_minute():
    mins, book, mark, liq = _base(100)
    X = int(mins[10])
    pos = pd.DataFrame(dict(sym="AAAUSDT", minute=[X + 30 * M, X + 2 * M], oi=[np.nan] * 2, oi_chg_1m=[np.nan] * 2, oi_chg_5m=[np.nan] * 2,
                            ratio_ts=[float(X)] * 2, top_ls_acct=[1.3] * 2, top_ls_pos=[1.1] * 2, global_ls_acct=[0.9] * 2,
                            taker_bs_ratio=[1.0] * 2))
    c = ST.Coin("AAAUSDT", book, mark, liq, None, pos)
    assert len(c.rt) == 1 and int(c.rt.minute.iloc[0]) == X + 2 * M


def test_unscored_families_keep_a_stable_schema():
    c, mins = _coin(fb=False, pos=False)
    r = c.state(int(mins[150]) + 1000)
    for k in list(ST.FB_KEYS) + ST.OI_KEYS + ST.RATIO_KEYS:
        assert k in r and r[k] != r[k], k


def test_ratio_too_old_is_unscored():
    mins, book, mark, liq = _base()
    pos = _pos(mins)
    pos = pos[pos.ratio_ts < mins[100]]
    assert ST.Coin("AAAUSDT", book, mark, liq, None, pos).state(int(mins[300]) + 1000)["has_ratio"] is False


def test_ratio_chg60_against_the_row_60_min_earlier():
    c, mins = _coin()
    T = int(mins[300]) + 1000
    r = c.state(T)
    ts = T - r["ratio_age_min"] * M
    v1 = 1 + (ts - T0) / (1000 * M)
    v0 = 1 + (ts - 60 * M - T0) / (1000 * M)
    assert r["top_ls_acct_chg60_pct"] == pytest.approx((v1 / v0 - 1) * 100)


def test_fb_columns_and_imbalance_and_coverage():
    c, mins = _coin()
    T = int(mins[150]) + 1000
    r = c.state(T)
    s = ST.state_minute(T)
    row = c.fb[c.fb.minute == s].iloc[0]
    assert r["has_fb"] and r["fb_age_min"] == 0 and r["fb_cost_buy_6000"] == pytest.approx(row.fb_cost_buy_6000)
    assert r["fb_imb_05"] == pytest.approx((row.fb_bid_usd_05 - row.fb_ask_usd_05) / (row.fb_bid_usd_05 + row.fb_ask_usd_05))
    c2, _ = _coin(fb=False, pos=False)
    r2 = c2.state(T)
    assert r2["has_fb"] is False and r2["has_oi"] is False and r2["has_ratio"] is False and math.isnan(r2["fb_imb_05"])


@pytest.mark.parametrize("cut", [150, 250, 330])
def test_dl611_no_look_ahead_on_shuffled_future(cut):
    """rows at/after floor_minute(T) (and OI rows at s; ratio rows not yet public) shuffled → identical DL 611 state."""
    c, mins = _coin()
    T = int(mins[cut]) + 20_000
    a = c.state(T)
    rng = np.random.default_rng(9)
    fb, pos = c.fb.copy(), pd.concat([c.oi, c.rt]).drop_duplicates("minute").sort_values("minute").reset_index(drop=True)
    _, book, mark, liq = _base()
    fut = fb.minute >= (T // M) * M
    for col in [x for x in fb.columns if x.startswith("fb_")]:
        fb.loc[fut, col] = rng.permutation(fb.loc[fut, col].to_numpy())
    s = ST.state_minute(T)
    of = pos.minute >= s                                         # OI rows not yet usable (the sweep-start row of s included)
    for col in ["oi", "oi_chg_1m", "oi_chg_5m"]:
        pos.loc[of, col] = rng.permutation(pos.loc[of, col].to_numpy()) * 3.7
    rf = (pos.minute >= s) | (pos.ratio_ts + TEN_MIN > T)         # positioning rows not yet public (hard-coded rule)
    for col in ST.RATIOS:
        pos.loc[rf, col] = rng.permutation(pos.loc[rf, col].to_numpy()) * 3.7
    b = ST.Coin("AAAUSDT", book, mark, liq, fb, pos).state(T)
    keys = [k for k in a if k.startswith(("fb_", "oi_", "has_", "ratio_")) or k in ST.RATIOS or k.endswith("_chg60_pct")]
    for k in keys:
        assert (a[k] == b[k]) or (a[k] != a[k] and b[k] != b[k]), k


def test_merge_fb_pos_split_minutes_whole_families():
    fb = pd.DataFrame(dict(sym=["A", "A"], minute=[1, 1], n_fbook=[10.0, 40.0], fb_bid_usd_05=[1.0, 2.0]))
    pos = pd.DataFrame(dict(sym=["A", "A"], minute=[1, 1], oi=[np.nan, 5.0], oi_chg_1m=[np.nan, 0.1], oi_chg_5m=[np.nan, 0.2],
                            ratio_ts=[9.0, 14.0], top_ls_acct=[1.2, 1.5], top_ls_pos=[np.nan, 0.7], global_ls_acct=[np.nan] * 2,
                            taker_bs_ratio=[np.nan] * 2))
    f, p = SRC.merge_fb_pos(fb, pos)
    assert len(f) == 1 and f.fb_bid_usd_05.iloc[0] == 2.0
    assert len(p) == 1 and p.oi.iloc[0] == 5.0 and p.oi_chg_5m.iloc[0] == 0.2
    assert p.ratio_ts.iloc[0] == 14.0 and p.top_ls_acct.iloc[0] == 1.5 and p.top_ls_pos.iloc[0] == 0.7   # one row, never mixed
    pos2 = pos.assign(ratio_ts=[14.0, 9.0], top_ls_pos=[np.nan, 0.7])
    _, p2 = SRC.merge_fb_pos(fb, pos2)
    assert p2.ratio_ts.iloc[0] == 14.0 and math.isnan(p2.top_ls_pos.iloc[0])                            # 0.7 never glued to ts 14


def test_coins_from_passes_dl611_frames():
    mins, book, mark, liq = _base(50)
    fb = _fb(mins).iloc[:0]
    pos = _pos(mins)
    cs = ST.coins_from(book, mark, liq, ["AAAUSDT", "ZZZUSDT"], fb=fb, pos=pos)
    assert len(cs["AAAUSDT"].oi) == 50 and len(cs["AAAUSDT"].fb) == 0 and len(cs["ZZZUSDT"].oi) == 0
    assert cs["ZZZUSDT"].state(int(mins[30]))["has_ratio"] is False


def _hour_csv(with_dl611):
    mins, book, mark, liq = _base(5)
    d = book.merge(mark, on=["sym", "minute"])
    d["liq_long_usd"], d["liq_short_usd"], d["vwap"], d["mid_open"] = 0.0, 0.0, 1.0, 1.0
    if with_dl611:
        d = d.merge(_fb(mins), on=["sym", "minute"]).merge(_pos(mins), on=["sym", "minute"])
        d["fb_cost_na_25000"] = 0
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        g.write(d.to_csv(index=False).encode())
    return buf.getvalue()


@pytest.mark.parametrize("with_dl611", [True, False])
def test_compact_hour_keeps_dl611_frames(tmp_path, monkeypatch, with_dl611):
    monkeypatch.setattr(SRC, "DATA", str(tmp_path))
    monkeypatch.setattr(SRC, "_HOUR_MEMO", {})
    os.makedirs(tmp_path / "minutes")
    raw = _hour_csv(with_dl611)
    monkeypatch.setattr(SRC, "fetch_feature", lambda key: raw)
    res = SRC.compact_hour("features/2026-11-22/01.csv.gz")
    assert list(res["fb"].columns) == ["sym", "minute"] + SRC.FB_COLS
    assert list(res["pos"].columns) == ["sym", "minute"] + SRC.OI_COLS + SRC.RATIO_COLS
    assert len(res["fb"]) == (5 if with_dl611 else 0) and len(res["pos"]) == (5 if with_dl611 else 0)
    assert os.path.exists(tmp_path / "minutes" / f"features_2026-11-22_01.v{SRC.MIN_CACHE_V}.pkl.gz")
    fb, pos = SRC.load_fb_pos(["features/2026-11-22/01.csv.gz"])
    assert len(fb) == len(res["fb"]) and fb.minute.dtype == np.int64
