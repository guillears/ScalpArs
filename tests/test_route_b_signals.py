"""Route B signal-level loader (scripts/route_b_signals) — no network: no look-ahead on the state join, the exit walkers per sleeve on
synthetic paths, the possibility / dedup logic, the journal → signal mapping."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import common as C       # noqa: E402
import exits as X        # noqa: E402
import signals as SG     # noqa: E402
import state as ST       # noqa: E402
import triggers as TR    # noqa: E402
import sources as SRC    # noqa: E402  (no network at import; network functions are monkeypatched below)
import build as BLD      # noqa: E402

M = C.MIN_MS
T0 = 1_795_000_000_000 // M * M          # a minute boundary


# ───────────────────────────────────────────────────────── synthetic recorded coin
def _coin(n=200, seed=1, sym="AAAUSDT"):
    rng = np.random.default_rng(seed)
    mins = T0 + np.arange(n) * M
    book = pd.DataFrame(dict(sym=sym, minute=mins, n_book=120.0, mid=1 + rng.random(n) * 0.01, mid_hi=1.01, mid_lo=0.99,
                             spread_bps=rng.random(n) * 5, bid_usd_02=rng.random(n) * 1e4, ask_usd_02=rng.random(n) * 1e4,
                             bid_usd_05=rng.random(n) * 3e4, ask_usd_05=rng.random(n) * 3e4, bid_usd_1=rng.random(n) * 6e4,
                             ask_usd_1=rng.random(n) * 6e4, imb_05=rng.random(n) - 0.5, reach_bid_pct=0.8, reach_ask_pct=0.3,
                             cost_buy_2500=5 + rng.random(n), cost_sell_2500=5 + rng.random(n), cost_buy_10000=12 + rng.random(n),
                             cost_sell_10000=np.where(rng.random(n) < 0.3, np.nan, 12.0), n_trades=10.0, buy_usd=rng.random(n) * 1e3,
                             sell_usd=rng.random(n) * 1e3, max_trade_usd=100.0))
    mark = pd.DataFrame(dict(sym=sym, minute=mins, mark=1 + rng.random(n) * 0.01, index=1.0, funding=rng.random(n) * 1e-4))
    lm = mins[rng.random(n) < 0.4]
    liq = pd.DataFrame(dict(sym=sym, minute=lm, liq_long_usd=rng.random(len(lm)) * 1e4, liq_short_usd=rng.random(len(lm)) * 1e4))
    return book, mark, liq


def _poison(df, cut):
    d = df.copy()
    num = [c for c in d.columns if c not in ("sym", "minute")]
    d.loc[d.minute >= cut, num] = d.loc[d.minute >= cut, num] * 1000 + 7
    return d


def _eq(a, b):
    assert a.keys() == b.keys()
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            assert isinstance(y, float) and math.isnan(y), k
        else:
            assert x == y, (k, x, y)


@pytest.mark.parametrize("T", [T0 + 150 * M, T0 + 150 * M + 1, T0 + 150 * M + 30_000, T0 + 150 * M + 59_999])
def test_state_no_look_ahead_coin(T):
    book, mark, liq = _coin()
    base = ST.Coin("AAAUSDT", book, mark, liq).state(T)
    cut = (T // M) * M                                     # every row whose minute has not ENDED by T
    poisoned = ST.Coin("AAAUSDT", _poison(book, cut), _poison(mark, cut), _poison(liq, cut)).state(T)
    _eq(base, poisoned)
    # and the state minute IS used: poisoning the state minute itself changes it
    s = ST.state_minute(T)
    assert s == cut - M and s + M <= T
    changed = ST.Coin("AAAUSDT", _poison(book, s), _poison(mark, s), _poison(liq, s)).state(T)
    assert changed["rb_spread_bps"] != base["rb_spread_bps"]


def test_state_no_look_ahead_market():
    book, mark, liq = _coin(sym="BTCUSDT")
    b2, m2, l2 = _coin(seed=2, sym="BBBUSDT")
    mark_all, liq_all = pd.concat([mark, m2]), pd.concat([liq, l2])
    T = T0 + 120 * M + 5_000
    base = ST.Market(mark_all, liq_all, book).state(T)
    cut = (T // M) * M
    pz = ST.Market(_poison(mark_all, cut), _poison(liq_all, cut), _poison(book, cut)).state(T)
    _eq(base, pz)


def test_book_age_and_lower_bound_flags():
    book, mark, liq = _coin()
    T = T0 + 100 * M
    b = book[book.minute < T0 + 95 * M]                    # last book row 5 min before the state minute → too old
    st = ST.Coin("AAAUSDT", b, mark, liq).state(T)
    assert st["rb_has_book"] is False and st["rb_has_mark"] is True
    st = ST.Coin("AAAUSDT", book, mark, liq).state(T)
    assert st["rb_has_book"] is True
    assert st["rb_ask_usd_05_lb"] is True and st["rb_bid_usd_05_lb"] is False      # reach_ask 0.3 % < 0.5 % band
    assert st["rb_bid_usd_1_lb"] is True


def test_cost_interpolation():
    v, lb = ST.cost_at(5.0, 12.5, usd=6000)
    assert lb is False and v == pytest.approx(5 + 7.5 * 3500 / 7500)
    v, lb = ST.cost_at(5.0, float("nan"))
    assert lb is True and math.isnan(v)


def test_liq_burst_uses_only_prior_baseline():
    sym = "LQUSDT"
    mins = T0 + np.arange(100) * M
    book = _coin(100, sym=sym)[0]
    mark = _coin(100, sym=sym)[1]
    liq = pd.DataFrame(dict(sym=sym, minute=mins[:60], liq_long_usd=1000.0, liq_short_usd=0.0))
    liq = pd.concat([liq, pd.DataFrame(dict(sym=sym, minute=[mins[80]], liq_long_usd=[1e6], liq_short_usd=[0.0]))])
    c = ST.Coin(sym, book, mark, liq)
    st = c.state(mins[80] + M)                              # state minute = 80: the burst minute is included in the 5-min sum
    assert st["rb_liq_long_5"] == pytest.approx(1e6)
    assert st["rb_liq_long_p99_5"] == pytest.approx(5000.0)   # baseline = prior 5-min sums of 1000/min only
    assert st["rb_liq_long_burst"] is True


# ───────────────────────────────────────────────────────── exit walkers
def _path(prices_hl, start=T0, o=None):
    h = np.array([p[0] for p in prices_hl], float)
    l = np.array([p[1] for p in prices_hl], float)
    c = (h + l) / 2
    o = np.array(o if o is not None else c, float)
    t = start + np.arange(len(h)) * M
    return o, h, l, c, t


def test_short_tp_maker_needs_trade_through():
    o, h, l, c, t = _path([(100.5, 99.5), (100, 95.0), (100, 94.9)])
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 60, entry_slip_pct=0.1, exit_slip_pct=0.3)
    assert w["reason"] == "TP" and w["minutes"] == 3                # 95.0 touches the level but not 0.05 % through
    assert w["exit_px"] == pytest.approx(95.0)
    assert w["net_pct"] == pytest.approx(5.0 - C.TAKER_PCT - C.MAKER_PCT - 0.1)   # no exit slip on a maker TP


def test_short_stop_first_and_gap_aware():
    o, h, l, c, t = _path([(101, 99), (107, 90)])                    # minute 2 touches both → STOP
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 60)
    assert w["reason"] == "STOP_LOSS" and w["exit_px"] == pytest.approx(106.0)
    o, h, l, c, t = _path([(101, 99), (112, 109)], o=[100, 110])       # gaps over the stop → the open
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 60, exit_slip_pct=0.2)
    assert w["exit_px"] == pytest.approx(110.0) and w["gross_pct"] == pytest.approx(-10.0)
    assert w["net_pct"] == pytest.approx(-10.0 - 2 * C.TAKER_PCT - 0.2)


def test_time_cap_and_censored():
    o, h, l, c, t = _path([(100.5, 99.5)] * 10)
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 5)
    assert w["reason"] == "MAX_HOLD_TIME" and w["minutes"] == 5 and w["final"]
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 60)
    assert w["reason"] == "CENSORED" and not w["final"] and w["minutes"] == 10


def test_long_ht_net_levels():
    th = dict(C.load_thresholds())
    sp = C.exit_specs(th)["FRENZY_WILLY_HT"]
    assert sp is not None and sp["kind"] == "net"
    tp, sl = C.levels_for(sp)
    assert tp == pytest.approx(sp["tp_net"] + th["_taker_fee"] * 100 + th["_maker_fee"] * 100)
    assert sl == pytest.approx(sp["sl_net"] - 2 * th["_taker_fee"] * 100)
    o, h, l, c, t = _path([(100.5, 99.5), (100 + tp + 0.2, 100)])
    w = X.walk(o, h, l, c, t, 100.0, 1, tp, sl, sp["hold_min"])
    assert w["reason"] == "TP" and w["mfe_pct"] > tp
    o, h, l, c, t = _path([(100.5, 99.5), (100.2, 100 - sl - 0.01)])
    w = X.walk(o, h, l, c, t, 100.0, 1, tp, sl, sp["hold_min"])
    assert w["reason"] == "STOP_LOSS" and w["exit_px"] == pytest.approx(100 - sl)


def test_ht_not_priced_when_late_tp_on():
    th = dict(C.load_thresholds())
    th["frenzy_willy_ht_late_tp_enabled"] = True
    assert C.exit_specs(th)["FRENZY_WILLY_HT"] is None
    assert C.exit_specs(th)["FRENZY_WILLY"] is None


def test_flatbase_stop_counts_funding_paid_only():
    o, h, l, c, t = _path([(100.5, 99.5), (100.5, 99.5), (108.5, 100)])
    fund_paid = [(T0 + M, -0.02, 100.0)]                             # short pays 2 % (negative rate) at minute 2
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, 600, funding=fund_paid, stop_counts_funding=True, entry_ms=T0 - 1)
    assert w["reason"] == "STOP_LOSS" and w["exit_px"] == pytest.approx(108.0)   # stop moved to +8 % by the 2 % paid
    assert w["funding_pct"] == pytest.approx(-2.0)
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, 600, funding=fund_paid, stop_counts_funding=False, entry_ms=T0 - 1)
    assert w["reason"] == "CENSORED"
    fund_recv = [(T0 + M, +0.02, 100.0)]                             # received funding never moves the stop
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, 600, funding=fund_recv, stop_counts_funding=True, entry_ms=T0 - 1)
    assert w["reason"] == "CENSORED" and w["funding_pct"] == pytest.approx(2.0)
    fund_before = [(T0 - 5 * M, -0.5, 100.0)]                         # a settlement before the entry is not ours
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 10.0, 600, funding=fund_before, stop_counts_funding=True, entry_ms=T0 - 1)
    assert w["funding_pct"] == 0.0


def test_atr_levels_caps_from_live_config():
    sp = C.exit_specs(C.load_thresholds())
    e = sp["EXHAUSTION_SHORT"]
    tp, sl = C.levels_for(e, 3.0)
    assert tp == pytest.approx(min(e["tp_atr"] * 3, e["tp_cap"])) and sl == pytest.approx(min(e["sl_atr"] * 3, e["sl_cap"]))
    assert C.levels_for(e, float("nan")) == (None, None)
    d = sp["DISTRIBUTION_SHORT"]
    assert C.levels_for(d, 0.2)[0] == pytest.approx(4.0)                # 20 × 0.2 below the cap


def test_forward_returns_and_grid_signs():
    o, h, l, c, t = _path([(101, 99)] * 300, o=[100] * 300)
    fr = X.forward_returns(o, c, -1)
    assert fr["fwd_5m_pct"] == pytest.approx(0.0)
    g = X.grid(o, h, l, c, t, 1, tps=(0.5,), sls=(5,), hold_min=240)
    assert g["g_tp0.5_sl5_net"] == pytest.approx(0.5 - C.TAKER_PCT - C.MAKER_PCT)
    fr = X.forward_returns(o[:10], c[:10], 1)
    assert math.isnan(fr["fwd_15m_pct"])


# ───────────────────────────────────────────────────────── possibilities / episodes / journal mapping
def _sig(sleeve, pair, minute, exit_min=None, hold=60):
    return dict(sleeve=sleeve, pair=pair, signal_time=pd.Timestamp(T0 + minute * M, unit="ms", tz="UTC"),
                signal_id=f"{sleeve}|{pair}|{minute}", exit_ms=(T0 + exit_min * M) if exit_min is not None else np.nan, hold_min=hold)


def test_possibilities_dedup_by_holding_period():
    S = pd.DataFrame([_sig("EXH", "A", 0, exit_min=30), _sig("EXH", "A", 10, exit_min=50), _sig("EXH", "A", 30, exit_min=40),
                      _sig("DIST", "A", 5, exit_min=9), _sig("EXH", "B", 10, exit_min=20), _sig("FB", "C", 0, hold=20), _sig("FB", "C", 19),
                      _sig("FB", "C", 20)])
    S["status"] = "BLOCKED:X"
    P = SG.possibilities(S).set_index("signal_id")
    assert not P.loc["EXH|A|0", "poss_dup"] and P.loc["EXH|A|10", "poss_dup"] and P.loc["EXH|A|10", "poss_id"] == "EXH|A|0"
    assert not P.loc["EXH|A|30", "poss_dup"]                                 # at the exit minute → a new possibility
    assert not P.loc["DIST|A|5", "poss_dup"] and not P.loc["EXH|B|10", "poss_dup"]
    assert P.loc["FB|C|19", "poss_dup"] and not P.loc["FB|C|20", "poss_dup"]  # unpriced → the sleeve's hold
    assert P.loc["EXH|A|10", "window15"] == pd.Timestamp(T0 + 10 * M, unit="ms", tz="UTC").floor("15min").strftime("%Y-%m-%dT%H:%M")


def test_trigger_episodes_gap():
    ep = TR.episodes([("X", T0), ("X", T0 + 10 * M), ("X", T0 + 60 * M), ("Y", T0 + 5 * M), ("X", T0 + 100 * M)], gap_min=60)
    assert ep == [("X", T0), ("X", T0 + 60 * M), ("Y", T0 + 5 * M)]


def test_journal_mapping():
    base = {c: np.nan for c in SG_COLS}
    J = pd.DataFrame([
        {**base, "t": "2026-11-20T14:15:00", "e": "EXHAUSTION", "pair": "KUSDT", "dir": "SHORT", "gate": "OPENED", "exh_status": "OPENED",
         "exh_signal_at": "2026-11-20T14:15:00", "exh_decision_price": 0.055, "exh_atr_pct": 5.0},
        {**base, "t": "2026-11-20T15:00:00", "e": "EXHAUSTION", "pair": "RUSDT", "dir": "SHORT", "gate": "EXH_NO_SPOT", "exh_status": "BLOCKED",
         "exh_block": "EXH_NO_SPOT", "exh_signal_at": "2026-11-20T15:00:00", "exh_decision_price": 0.0125},
        {**base, "t": "2026-11-20T11:20:10", "e": "FLATBASE", "pair": "KUSDT", "dir": "SHORT", "gate": "TRACKING", "fb_status": "TRACKING",
         "fb_spike_at": "2026-11-20T11:20:00"},
        {**base, "t": "2026-11-20T20:00:35", "e": "FLATBASE", "pair": "KUSDT", "dir": "SHORT", "gate": "OPENED", "fb_status": "OPENED",
         "fb_spike_at": "2026-11-20T11:20:00", "price": 0.0442},
        {**base, "t": "2026-11-21T00:00:37", "e": "FLATBASE", "pair": "PUSDT", "dir": "SHORT", "gate": "SKIPPED", "fb_status": "SKIPPED",
         "fb_spike_at": "2026-11-20T17:40:00", "fb_entry_bar_at": "2026-11-20T19:55:00", "price": 4.5,
         "fb_reason": "no 4-h close below the S4 close 4.5 within 24 h → no trade (FB_CONFIRM)"},
        {**base, "t": "2026-11-21T00:10:00", "e": "FLATBASE", "pair": "QUSDT", "dir": "SHORT", "gate": "SKIPPED", "fb_status": "SKIPPED",
         "fb_spike_at": "2026-11-20T17:40:00", "price": 1.0, "fb_reason": "FUNDING_ABOVE_BASE: entry funding +0.01 % > +0.005 %"},
        {**base, "t": "2026-11-21T09:55:06", "e": "WILLY", "pair": "RUSDT", "gate": "FRENZY_WILLY_TURNOVER", "reason": "R=2.5 · A"},
        {**base, "t": "2026-11-21T10:00:00", "e": "BLOCK", "pair": "RUSDT", "dir": "LONG", "gate": "FRENZY_WILLY_PAIR_HELD"},
        {**base, "t": "2026-11-21T10:30:09", "e": "BLOCKED", "pair": "AUSDT", "dir": "LONG", "strategy": "FRENZY_WILLY_HT", "price": 0.01,
         "bl_sleeve": "FRENZY_WILLY_HT", "bl_lines": "WILLY_FAILED_SPIKE"},
        {**base, "t": "2026-11-21T11:00:00", "e": "BLOCK", "pair": "AUSDT", "dir": "LONG", "gate": "FRENZY_WIDE_OFF"},
        {**base, "t": "2026-11-21T11:05:00", "e": "BLOCK", "pair": "AUSDT", "dir": "SHORT", "gate": "EXH_WIDE_READ_CAP"},
    ])
    S = SG.build_signals(J, turnover_cap=1.0)
    st = {(r.sleeve, r.pair): r for r in S.itertuples()}
    assert st[("EXHAUSTION_SHORT", "KUSDT")].status == "OPENED" and st[("EXHAUSTION_SHORT", "KUSDT")].journal_atr_pct == 5.0
    assert st[("EXHAUSTION_SHORT", "RUSDT")].status == "BLOCKED:EXH_NO_SPOT"
    fb = S[S.sleeve == "FLATBASE_SHORT"].set_index("pair")
    assert fb.loc["KUSDT", "status"] == "OPENED" and len(S[(S.sleeve == "FLATBASE_SHORT") & (S.pair == "KUSDT")]) == 1   # TRACKING superseded
    assert fb.loc["PUSDT", "status"] == "REFUSED:SKIPPED_CONFIRM" and not fb.loc["PUSDT", "core_rule_met"]
    assert fb.loc["PUSDT", "signal_time"] == pd.Timestamp("2026-11-20T20:00:00", tz="UTC")          # the S4 bar close
    assert fb.loc["QUSDT", "status"] == "REFUSED:SKIPPED_FUNDING_ABOVE_BASE"
    assert st[("FRENZY_WILLY_HT", "RUSDT")].status == "BLOCKED:PAIR_HELD"                            # R 2.5 > cap 1 → HT
    assert st[("FRENZY_WILLY_HT", "AUSDT")].status == "BLOCKED:WILLY_FAILED_SPIKE"
    assert st[("FRENZY_WIDE", "AUSDT")].status == "REFUSED:SLEEVE_OFF" and not st[("FRENZY_WIDE", "AUSDT")].priceable
    assert not ((S.pair == "AUSDT") & (S.direction == "SHORT")).any()                              # the read-cap counter is no signal


SG_COLS = ["t", "e", "pair", "dir", "gate", "reason", "strategy", "price", "exh_status", "exh_block", "exh_reason", "exh_signal_at",
           "exh_decision_price", "exh_atr_pct", "exh_tp_pct", "exh_sl_pct", "exh_funding_rate", "exh_model_slip_pct", "dist_status",
           "dist_block", "dist_reason", "dist_signal_at", "dist_decision_price", "dist_atr_pct", "dist_tp_pct", "dist_sl_pct",
           "dist_funding_rate", "dist_vsw_mult", "dist_bar_ret_pct", "dist_vol24_usd", "dist_pos_chg_4h", "dist_rsi1h", "exh_vol24_usd", "exh_has_spot",
           "exh_vol_decay", "exh_spot_share_chg", "exh_oi_vol", "exh_universe", "fb_status", "fb_reason", "fb_spike_at", "fb_entry_bar_at", "fb_funding_rate", "bl_sleeve", "bl_lines", "bl_sole"]


# ───────────────────────────────────────────────────────── review round 1 (fixes 1–16)
def _day_frame(day_ms, n=1440):
    t = day_ms + np.arange(n) * M
    p = 100 + np.arange(n) * 0.01
    return pd.DataFrame(dict(open_ms=t, o=p, h=p + 0.05, l=p - 0.05, c=p, qv=1.0))


def test_klines_day_cache_slices_per_call(monkeypatch):
    calls = []
    d0 = int(pd.Timestamp("2026-11-20", tz="UTC").value // 1_000_000)

    def fake_day(sym, day, now_ms):
        calls.append((sym, day))
        return _day_frame(int(pd.Timestamp(day, tz="UTC").value // 1_000_000))
    monkeypatch.setattr(SRC, "klines_day", fake_day)
    SRC._DAY_CACHE.clear()
    now = d0 + 3 * 86400_000
    a = SRC.klines("XUSDT", d0 + 10 * M, d0 + 20 * M, now)
    b = SRC.klines("XUSDT", d0 + 100 * M, d0 + 700 * M, now)          # same day, a LATER and LONGER window
    assert len(a) == 10 and a.open_ms.iloc[0] == d0 + 10 * M
    assert len(b) == 600 and b.open_ms.iloc[0] == d0 + 100 * M and b.open_ms.iloc[-1] == d0 + 699 * M
    assert calls == [("XUSDT", "2026-11-20")]                            # the day was fetched once
    c = SRC.klines("XUSDT", d0 + 1400 * M, d0 + 1500 * M, now)           # spans two days
    assert len(c) == 100 and ("XUSDT", "2026-11-21") in calls
    SRC._DAY_CACHE.clear()


def test_carry_invalidation():
    th = C.load_thresholds()
    sp = C.exit_specs(th)["EXHAUSTION_SHORT"]
    k = BLD.carry_key(sp, "2026-11-20T14:15:30|KUSDT|SHORT", "OPENED")
    prev = dict(code_version=BLD.CODE_VERSION, carry_key=k, out_priced=True, out_final=True, out_funding_ok=True, out_lbl_done=True,
                out_exit_slip_src="route_b_book_at_exit", out_exit_ms=1.0, out_data_end_ms=2.0)
    assert BLD.can_carry(prev, k)
    sp2 = dict(sp, tp_cap=sp["tp_cap"] + 3)                                # exit spec changed → re-price
    assert not BLD.can_carry(prev, BLD.carry_key(sp2, "2026-11-20T14:15:30|KUSDT|SHORT", "OPENED"))
    assert not BLD.can_carry(prev, BLD.carry_key(sp, "2026-11-20T14:20:00|KUSDT|SHORT", "OPENED"))   # other fill linked
    assert not BLD.can_carry(prev, BLD.carry_key(sp, None, "OPENED_UNLINKED"))
    assert not BLD.can_carry(dict(prev, out_funding_ok=False), k)
    assert not BLD.can_carry(dict(prev, out_final=False), k)
    assert not BLD.can_carry(dict(prev, code_version=BLD.CODE_VERSION - 1), k)


def _sig_row(status="BLOCKED:EXH_VOL_DECAY", **kw):
    r = dict(sleeve="EXHAUSTION_SHORT", pair="KUSDT", direction="SHORT", signal_time=pd.Timestamp(T0 + 30 * M, unit="ms", tz="UTC"),
             status=status, priceable=True, journal_atr_pct=1.0, journal_model_slip_pct=math.nan, fill_entry_price=math.nan,
             fill_opened_at=None)
    r.update(kw)
    return pd.Series(r)


def test_funding_ok_gates_final_and_unlinked_never_final(monkeypatch):
    k = _day_frame(T0 - 40 * 60 * M, n=40 * 60 + 1700)                     # covers the 24-h entry-label hold too (build item 3)
    k["h"] = k.o + 0.01
    k.loc[k.open_ms >= T0 + 40 * M, "l"] = 1.0                             # crashes → short TP right after the entry
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], False))
    o = BLD.price_signal(_sig_row(), sp, k, None, T0 + 10_000 * M, 0)
    assert o["out_reason"] == "TP" and o["out_path_final"] and not o["out_final"] and "funding" in o["out_why"]
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    o = BLD.price_signal(_sig_row(), sp, k, None, T0 + 10_000 * M, 0)
    assert o["out_final"]
    o = BLD.price_signal(_sig_row(status="OPENED_UNLINKED"), sp, k, None, T0 + 10_000 * M, 0)
    assert o["out_path_final"] and not o["out_final"] and o["out_entry_src"] == "next_minute_open"


def test_stop_fill_worse_of_level_and_close():
    o, h, l, c, t = _path([(100.5, 99.5), (109, 100)], o=[100, 100.5])   # close (104.5)… below the 106 level → the level
    w = X.walk(o, h, l, c, t, 100.0, -1, 5.0, 6.0, 60)
    assert w["exit_px"] == pytest.approx(106.0)
    h2 = np.array([100.5, 109.0]); l2 = np.array([99.5, 100.0]); c2 = np.array([100.0, 108.0])
    w = X.walk(o, h2, l2, c2, t, 100.0, -1, 5.0, 6.0, 60)                 # a fast candle that closes through → the close
    assert w["exit_px"] == pytest.approx(108.0)
    w = X.walk(o, h2, l2, c2, t, 100.0, -1, 5.0, 6.0, 60, stop_fill="level")
    assert w["exit_px"] == pytest.approx(106.0)


def test_h_funding_kept_apart():
    o, h, l, c, t = _path([(100.2, 99.8)] * 300, o=[100] * 300)
    fund = [(t[0] + 30 * M, 0.01, 100.0)]                                  # +1 % rate: a short receives, a long pays
    fr = X.forward_returns(o, c, -1, t=t, funding=fund)
    assert math.isnan(fr["fwd_15m_funding_pct"]) is False and fr["fwd_15m_funding_pct"] == 0.0 and fr["fwd_60m_funding_pct"] == pytest.approx(1.0)
    assert X.funding_pct(fund, 1, 100.0, t[0], t[0] + 240 * M) == pytest.approx(-1.0)
    assert X.funding_pct([(t[0], 0.01, 100.0)], 1, 100.0, t[0], t[0] + 240 * M) == 0.0     # at the entry open: not ours


def _trig_book(n=400, sym="TTTUSDT", seed=3):
    rng = np.random.default_rng(seed)
    mins = T0 + np.arange(n) * M
    imb = rng.uniform(-0.3, 0.3, n)
    imb[150:154] = 0.7
    imb[260:264] = -0.7
    ask = rng.uniform(15_000, 20_000, n)
    ask[200] = 2_000
    buy = rng.uniform(100, 200, n)
    sell = rng.uniform(100, 200, n)
    buy[300:315] = 5_000
    return pd.DataFrame(dict(sym=sym, minute=mins, imb_05=imb, bid_usd_05=rng.uniform(15_000, 18_000, n), ask_usd_05=ask, buy_usd=buy,
                             sell_usd=sell))


def test_h3_h5_causal_on_shuffled_future():
    book = _trig_book()
    mark = pd.DataFrame(columns=["sym", "minute", "mark", "index", "funding"])
    base = TR.detect(book, mark, {})
    assert {"H3_imb_bid_heavy_thin", "H3_imb_ask_heavy_thin", "H4_ask_vacuum", "H5_taker_buy_climax"} <= set(base.trigger)
    for cut in (T0 + 155 * M, T0 + 205 * M, T0 + 316 * M):
        fut = book.copy()
        m = fut.minute > cut
        rng = np.random.default_rng(9)
        for col in ("imb_05", "bid_usd_05", "ask_usd_05", "buy_usd", "sell_usd"):
            v = fut.loc[m, col].to_numpy()
            fut.loc[m, col] = rng.permutation(v) * rng.uniform(0.01, 50, len(v))
        got = TR.detect(fut, mark, {})
        a = base[base.minute <= cut].sort_values(["trigger", "pair", "minute"]).reset_index(drop=True)
        b = got[got.minute <= cut].sort_values(["trigger", "pair", "minute"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["trigger", "pair", "minute"]], b[["trigger", "pair", "minute"]])


def test_atr_from_klines_causal():
    k = _day_frame(T0 - 40 * 60 * M, n=40 * 60 + 300)
    k["h"] = k.o * (1 + np.abs(np.sin(np.arange(len(k)))) / 100)
    T = T0 + 7 * M + 30_000
    base = BLD.atr_from_klines(k, T)
    assert base == base and base > 0
    k2 = k.copy()
    fut = k2.open_ms + M > T                                               # every minute not closed by T
    k2.loc[fut, ["h", "c"]] = k2.loc[fut, ["h", "c"]] * 3
    assert BLD.atr_from_klines(k2, T) == pytest.approx(base)


def test_gate_audit_multi_gate_and_unscored():
    th = C.load_thresholds()
    S = pd.DataFrame([
        dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_VOL_DECAY", core_rule_met=True, priceable=True, j_exh_vol24_usd=5e7,
             j_exh_has_spot=True, j_exh_vol_decay=0.1, j_exh_ssc=0.05, j_exh_oi_vol=0.01, j_exh_universe="FRENZY"),
        dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_VOL_DECAY", core_rule_met=True, priceable=True, j_exh_vol24_usd=5e7,
             j_exh_has_spot=True, j_exh_vol_decay=0.1, j_exh_ssc=-0.05, j_exh_oi_vol=0.01, j_exh_universe="FRENZY"),
        dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_NO_SPOT", core_rule_met=True, priceable=True),
        dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_PAIR_HELD", core_rule_met=True, priceable=True),
        dict(sleeve="DISTRIBUTION_SHORT", status="BLOCKED:DIST_RSI1H", core_rule_met=True, priceable=True, j_dist_vsw_mult=6.0,
             j_dist_bar_ret_pct=-1.0, j_dist_vol24_usd=3e7, j_dist_pos_chg_4h=-0.2, j_dist_rsi1h=50.0),
    ])
    A = SG.gate_audit(S, th)
    assert A.loc[0, "also_blocked_by"] == "EXH_SPOT_SHARE" and not A.loc[0, "sole_block"]
    assert A.loc[1, "also_blocked_by"] == "" and A.loc[1, "filters_unevaluated"] == "" and A.loc[1, "sole_block"]
    assert "EXH_VOL_DECAY" in A.loc[2, "filters_unevaluated"] and not A.loc[2, "sole_block"]
    assert A.loc[3, "block_class"] == "capacity" and not A.loc[3, "sole_block"]
    assert A.loc[4, "sole_block"] and A.loc[4, "block_class"] == "filter"


def test_pair_held_joins_the_holding_possibility():
    ts = lambda m: pd.Timestamp(T0 + m * M, unit="ms", tz="UTC")
    S = pd.DataFrame([
        dict(signal_id="EXH|K|0|OPENED", sleeve="EXHAUSTION_SHORT", pair="K", status="OPENED", signal_time=ts(0), out_exit_ms=T0 + 90 * M,
             poss_dup=False, poss_id="EXH|K|0|OPENED", poss_note=""),
        dict(signal_id="FB|K|60|HELD", sleeve="FLATBASE_SHORT", pair="K", status="REFUSED:PAIR_HELD", signal_time=ts(60), out_exit_ms=np.nan,
             poss_dup=False, poss_id="FB|K|60|HELD", poss_note=""),
        dict(signal_id="FB|K|120|HELD", sleeve="FLATBASE_SHORT", pair="K", status="REFUSED:PAIR_HELD", signal_time=ts(120), out_exit_ms=np.nan,
             poss_dup=False, poss_id="FB|K|120|HELD", poss_note=""),
    ])
    P = SG.pair_held_links(S).set_index("signal_id")
    assert P.loc["FB|K|60|HELD", "poss_dup"] and P.loc["FB|K|60|HELD", "poss_id"] == "EXH|K|0|OPENED"
    assert P.loc["FB|K|120|HELD", "poss_dup"] and P.loc["FB|K|120|HELD", "poss_id"] == "FB|K|120|HELD"    # holder gone: own id, still capacity


def test_exports_sorted_by_name_stamp_paper_and_live(tmp_path, monkeypatch):
    a, d = tmp_path / "a", tmp_path / "d"
    a.mkdir(); d.mkdir()
    for n in ("scalpars_orders_paper_2026-11-18_10-00-00.csv", "scalpars_orders_paper_2026-11-21_10-00-00.csv",
              "scalpars_orders_live_2026-11-21_12-00-00.csv", "scalpars_orders_live_2026-11-20_09-00-00.csv"):
        (a / n).write_text("x")
    monkeypatch.setattr(SRC, "ARCHIVE", str(a))
    monkeypatch.setattr(SRC, "DOWNLOADS", str(d))
    got = [os.path.basename(f) for f in SRC._export_files("orders", since=pd.Timestamp("2026-11-19", tz="UTC"))]
    assert got == ["scalpars_orders_live_2026-11-20_09-00-00.csv", "scalpars_orders_paper_2026-11-21_10-00-00.csv",
                   "scalpars_orders_live_2026-11-21_12-00-00.csv"]


def test_merge_spans_and_limiter_network_error(monkeypatch):
    assert SRC.merge_spans([(5, 9), (1, 3), (3, 4), (20, 30)]) == [(1, 4), (5, 9), (20, 30)]
    lim = SRC.Limiter()

    def boom(*a, **kw):
        raise SRC.requests.ConnectionError("blip")
    monkeypatch.setattr(lim.s, "get", boom)
    assert lim.get("/fapi/v1/klines", {}, 5) is None and lim.errors == 1 and not lim.stopped


def test_willy_turnover_read_is_causal():
    base = {c: np.nan for c in SG_COLS}
    J = pd.DataFrame([
        {**base, "t": "2026-11-21T10:00:00", "e": "BLOCK", "pair": "RUSDT", "dir": "LONG", "gate": "FRENZY_WILLY_PAIR_HELD"},
        {**base, "t": "2026-11-21T10:00:10", "e": "WILLY", "pair": "RUSDT", "gate": "FRENZY_WILLY_TURNOVER", "reason": "R=2.5 · A"},
    ])
    S = SG.build_signals(J, turnover_cap=1.0)
    assert S.iloc[0].sleeve == "FRENZY_WILLY"                              # the read 10 s AFTER the decision is not used


def test_fb_tracking_episodes_counts_last_status():
    J = pd.DataFrame([dict(t="1", e="FLATBASE", pair="A", fb_spike_at="s", fb_status="TRACKING"),
                      dict(t="2", e="FLATBASE", pair="A", fb_spike_at="s", fb_status="OPENED"),
                      dict(t="1", e="FLATBASE", pair="B", fb_spike_at="s", fb_status="TRACKING"),
                      dict(t="3", e="FLATBASE", pair="B", fb_spike_at="s", fb_status="TRACKING")])
    assert BLD.fb_tracking_episodes(J) == 1


# ───────────────────────────────────────────────────────── review round 2
def test_gate_classes_and_refused_status():
    th = C.load_thresholds()
    assert SG._status("REFUSED", np.nan, "REFUSED") == "REFUSED:OPEN_PATH"
    assert SG._status("REFUSED", "DIST_OPEN_REFUSED", "X") == "REFUSED:DIST_OPEN_REFUSED"
    assert SG._status("BLOCKED", "EXH_NO_DATA", "EXH_NO_DATA") == "BLOCKED:EXH_NO_DATA"
    cases = {"BLOCKED:EXH_NO_DATA": "data", "BLOCKED:EXH_NO_SIGNAL": "core_rule", "BLOCKED:EXH_SPOT_MISMATCH": "data",
             "BLOCKED:EXH_WIDE_READ_CAP": "capacity", "REFUSED:DIST_OPEN_REFUSED": "capacity", "BLOCKED:DIST_NO_DATA": "data",
             "BLOCKED:DIST_RSI1H_UNREAD": "data", "BLOCKED:DIST_BOOK_FULL": "capacity", "BLOCKED:EXH_LATE": "capacity",
             "REFUSED:OPEN_PATH": "capacity", "BLOCKED:WILLY_FAILED_SPIKE": "filter", "BLOCKED:EXH_FAILED": "data",
             "BLOCKED:WILLY_GAP+WILLY_FAILED_SPIKE": "filter", "BLOCKED:DIST_NOT_VSW": "core_rule", "BLOCKED:EXH_PAIR_HELD": "capacity",
             "BLOCKED:EXH_VOL_DECAY": "filter", "REFUSED:SKIPPED_FUNDING_ABOVE_BASE": "filter", "OPENED_UNLINKED": "taken"}
    for st, c in cases.items():
        assert SG.gate_class(st) == c, st
    rows = [dict(sleeve="EXHAUSTION_SHORT" if "EXH" in st else "DISTRIBUTION_SHORT", status=st, core_rule_met=True, priceable=True)
            for st in cases if "EXH" in st or "DIST" in st]
    A = SG.gate_audit(pd.DataFrame(rows), th)
    assert not A.sole_block.any()                                          # none of these can be a clean single-filter row


def test_sole_needs_a_known_scored_gate():
    th = C.load_thresholds()
    full = dict(j_exh_vol24_usd=5e7, j_exh_has_spot=True, j_exh_vol_decay=0.9, j_exh_ssc=-0.05, j_exh_oi_vol=0.01, j_exh_universe="FRENZY")
    S = pd.DataFrame([dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_SOMETHING_NEW", core_rule_met=True, priceable=True, **full),
                      dict(sleeve="EXHAUSTION_SHORT", status="BLOCKED:EXH_VOL_DECAY", core_rule_met=True, priceable=True,
                           **dict(full, j_exh_vol_decay=0.1, j_exh_oi_vol=np.nan))])
    A = SG.gate_audit(S, th)
    assert not A.loc[0, "sole_block"] and A.loc[0, "filters_unevaluated"] == "named gate not a scored filter"
    assert not A.loc[1, "sole_block"] and A.loc[1, "filters_not_reached"] == "EXH_SSC_OIVOL" and A.loc[1, "filters_unevaluated"] == ""


def test_dist_rsi_default_is_engine_default():
    th = {k: v for k, v in C.load_thresholds().items() if k != "distribution_rsi1h_min"}
    ev = SG.gate_eval(dict(sleeve="DISTRIBUTION_SHORT", j_dist_rsi1h=60.0), th)
    assert ev["DIST_RSI1H"] is True


def test_can_carry_nan_safe_and_permanent_only():
    k = BLD.carry_key(None, None, "BLOCKED:NOT_FLAT")
    base = dict(code_version=BLD.CODE_VERSION, carry_key=k)
    assert not BLD.can_carry(dict(base, out_priced=np.nan, out_final=np.nan, out_funding_ok=np.nan, out_why="no 1m klines"), k)
    assert not BLD.can_carry(dict(base, out_priced=False, out_why="entry minute not in klines yet"), k)
    assert not BLD.can_carry(dict(base, out_priced=False, out_why="levels undefined (ATR missing)"), k)
    assert BLD.can_carry(dict(base, out_priced=False, out_why="no entry decision (no entry bar journaled)"), k)
    assert BLD.can_carry(dict(base, out_priced=np.nan, out_why=C.NOT_PRICED_WHY["FRENZY_WIDE"]), k)
    pr = dict(base, out_priced=True, out_final=True, out_funding_ok=np.nan, out_exit_slip_src="maker_tp")
    assert not BLD.can_carry(pr, k)                                          # NaN funding_ok is not True
    pr = dict(base, out_priced=True, out_final=True, out_funding_ok=True, out_lbl_done=True, out_exit_slip_src="default", out_exit_ms=5.0, out_data_end_ms=5.0)
    assert not BLD.can_carry(pr, k)                                          # fallback slip with the exit at/after the then data end
    assert BLD.can_carry(dict(pr, out_exit_ms=4.0), k)


def _crash_klines():
    k = _day_frame(T0 - 40 * 60 * M, n=40 * 60 + 200)
    k["h"] = k.o + 0.01
    k.loc[k.open_ms >= T0 + 40 * M, "l"] = 1.0
    return k


def test_exit_after_data_end_with_fallback_slip_not_final(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = dict(C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"], sl_cap=0.05, sl_atr=0.05)   # a tiny stop → a STOP exit (taker leg)
    k = _crash_klines()
    k["h"] = k.o * 1.01
    now = T0 + 10_000 * M
    o = BLD.price_signal(_sig_row(), sp, k, None, now, T0 + 5 * M)          # Route B ends before the exit; no coin book → default slip
    assert o["out_reason"] == "STOP_LOSS" and o["out_exit_slip_src"] == "default" and not o["out_final"]
    o = BLD.price_signal(_sig_row(), sp, k, None, now, T0 + 9_000 * M)
    assert o["out_final"]


def test_funding_not_final_inside_publish_lag(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    k = _crash_klines()
    o = BLD.price_signal(_sig_row(), sp, k, None, T0 + 35 * M, 0)           # exit at ~T0+31 min, now T0+35 min (< 10-min lag)
    assert o["out_path_final"] and not o["out_final"] and not o["out_funding_ok"]


def test_funding_coverage_judged_up_to_now_minus_lag(tmp_path, monkeypatch):
    monkeypatch.setattr(SRC, "DATA", str(tmp_path))
    (tmp_path / "funding").mkdir()
    seen = {}

    class L:
        stopped = False

        def get(self, path, params, w):
            seen.update(params)
            return []
    monkeypatch.setattr(SRC, "limiter", lambda: L())
    now = T0 + 1000 * M
    _, ok = SRC.funding_history("XUSDT", T0, now + 60 * M, now)
    assert ok and seen["endTime"] == now - SRC.FUNDING_PUBLISH_LAG_MS
    spans = pd.read_pickle(tmp_path / "funding" / "XUSDT.pkl.gz")[1]
    assert spans[-1][1] == now - SRC.FUNDING_PUBLISH_LAG_MS


def test_taken_fill_minute_walked_pessimistically():
    k = _day_frame(T0, n=10)
    k.loc[0, ["o", "h", "l", "c"]] = [100.0, 100.5, 94.0, 100.0]           # the fill minute dipped to 94 (maybe before the fill)
    p = BLD.pessimistic_fill_minute(k, 0, 100.2, -1)                         # short filled at 100.2
    assert p.loc[0, "o"] == 100.2 and p.loc[0, "l"] == pytest.approx(100.0) and p.loc[0, "h"] == pytest.approx(100.5)
    p = BLD.pessimistic_fill_minute(k, 0, 100.2, 1)                          # long: the dip is kept (adverse), the high capped
    assert p.loc[0, "l"] == pytest.approx(94.0) and p.loc[0, "h"] == pytest.approx(100.2)


def test_taken_opens_own_possibility_and_unpriced_never_absorbs():
    S = pd.DataFrame([_sig("EXH", "A", 0, exit_min=100), _sig("EXH", "A", 20, exit_min=40), _sig("EXH", "A", 30, exit_min=60),
                      _sig("FB", "B", 0, hold=np.nan), _sig("FB", "B", 30, hold=np.nan), _sig("FB", "B", 61, hold=np.nan),
                      _sig("FB", "B", 10, exit_min=20)])
    S["status"] = ["BLOCKED:X", "OPENED", "BLOCKED:X", "REFUSED:SLEEVE_OFF", "REFUSED:SLEEVE_OFF", "REFUSED:SLEEVE_OFF", "BLOCKED:Y"]
    P = SG.possibilities(S).set_index("signal_id")
    assert not P.loc["EXH|A|20", "poss_dup"]                                 # taken inside a blocked row's hypothetical hold
    assert P.loc["EXH|A|30", "poss_dup"] and P.loc["EXH|A|30", "poss_id"] == "EXH|A|20"
    assert not P.loc["FB|B|10", "poss_dup"]                                  # a priced row is never absorbed by an unpriced one
    assert P.loc["FB|B|30", "poss_dup"] and not P.loc["FB|B|61", "poss_dup"]   # unpriced, no hold → the 60-min fallback window


def test_fb_signal_time_uses_entry_bar_for_every_final_status():
    base = {c: np.nan for c in SG_COLS}
    J = pd.DataFrame([{**base, "t": "2026-11-21T09:00:40", "e": "FLATBASE", "pair": "XUSDT", "dir": "SHORT", "gate": "SKIPPED",
                       "fb_status": "SKIPPED", "fb_spike_at": "s", "fb_entry_bar_at": "2026-11-21T07:55:00", "price": 1.0,
                       "fb_reason": "NEAR_BASE: the confirm close is +1.00 % above the base"},
                      {**base, "t": "2026-11-21T10:00:00", "e": "FLATBASE", "pair": "YUSDT", "dir": "SHORT", "gate": "EXPIRED",
                       "fb_status": "EXPIRED", "fb_spike_at": "s", "fb_entry_bar_at": "2026-11-21T03:55:00", "price": 2.0, "fb_reason": "x"}])
    S = SG.build_signals(J).set_index("pair")
    assert S.loc["XUSDT", "signal_time"] == pd.Timestamp("2026-11-21T08:00:00", tz="UTC")
    assert S.loc["YUSDT", "signal_time"] == pd.Timestamp("2026-11-21T04:00:00", tz="UTC")


def test_load_minutes_all_failed_exits_cleanly(monkeypatch):
    monkeypatch.setattr(SRC, "compact_hour", lambda k: None)
    with pytest.raises(SystemExit):
        SRC.load_minutes(["features/2026-11-21/00.csv.gz"])
