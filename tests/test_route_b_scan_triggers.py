"""Route B weekly-scan PREREG build item 5 — H7 (OI flush / build), the depth-conditioned variants, the matched random-minute controls. No network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import controls as CT    # noqa: E402
import triggers as TR    # noqa: E402

M = 60_000
T0 = 1_795_000_000_000 // (1440 * M) * (1440 * M)        # a UTC midnight


def _mark(n=60, ret_at=None, ret=0.0):
    mins = T0 + np.arange(n) * M
    px = np.ones(n)
    if ret_at is not None:
        px[ret_at:] = 1 + ret / 100
    return pd.DataFrame(dict(minute=mins, mark=px))


@pytest.mark.parametrize("d_oi,ret,key", [(-4, -2.5, "H7a_oi_flush_dump"), (-4, 2.5, "H7b_oi_flush_pump"), (6, 2.5, "H7c_oi_build_pump")])
def test_h7_keys_and_lag(d_oi, ret, key):
    mk = _mark(ret_at=30, ret=ret)                                  # the mark jumps at minute 30
    q = T0 + 32 * M                                                 # OI row stamped 32: return mark(32) / mark(27)
    oi = pd.DataFrame(dict(minute=[q], oi_chg_5m=[d_oi]))
    h = TR.h7_hits(oi, mk)
    assert h[key] == [q + M]                                        # trigger minute = row + 1 (the sweep-start lag rule)
    assert sum(len(v) for v in h.values()) == 1


def test_h7_mark_span_matches_the_oi_row():
    mk = _mark(ret_at=40, ret=-3)                                   # the drop happens AFTER the OI row's span
    oi = pd.DataFrame(dict(minute=[T0 + 35 * M], oi_chg_5m=[-5.0]))
    assert TR.h7_hits(oi, mk)["H7a_oi_flush_dump"] == []


def test_h7_stale_mark_unscored():
    mk = _mark(n=20)
    oi = pd.DataFrame(dict(minute=[T0 + 40 * M], oi_chg_5m=[-5.0]))  # no mark within 2 min of 40 / 35
    assert all(v == [] for v in TR.h7_hits(oi, mk).values())


def test_family_names():
    assert TR.family("H1_liq_long_cascade|c") == "H1" and TR.family("H7a_oi_flush_dump") == "H7a" and TR.family("H4_bid_vacuum") == "H4"


def test_depth_prior_median_is_causal():
    n = 200
    mins = T0 + np.arange(n) * M
    bid = np.r_[np.full(100, 10.0), np.full(100, 20.0)]
    fb = pd.DataFrame(dict(minute=mins, fb_bid_usd_05=bid, fb_ask_usd_05=bid))
    d = TR.Depth(fb)
    assert d.ok(int(mins[50]), "bid") == (False, None)              # < 60 prior rows → unscored
    assert d.ok(int(mins[99]), "bid") == (True, True)               # 10 ≥ median 10
    assert d.ok(int(mins[100]), "bid") == (True, True)              # the row itself is not in its own median
    fb2 = fb.copy()
    fb2.loc[150:, "fb_bid_usd_05"] = 1e9                            # the future never changes an earlier verdict
    assert TR.Depth(fb2).ok(int(mins[120]), "bid") == d.ok(int(mins[120]), "bid")
    assert d.ok(int(mins[-1]) + 5 * M, "bid") == (False, None)      # fb row too old


def _book(sym, mins):
    z = np.zeros(len(mins))
    return pd.DataFrame(dict(sym=sym, minute=mins, n_book=1.0, bid_usd_05=z + 1e5, ask_usd_05=z + 1e5, imb_05=z, buy_usd=z, sell_usd=z))


def test_detect_variants_subset_of_parent_and_flags():
    mins = T0 + np.arange(300) * M
    book = _book("AAAUSDT", mins)
    mark = pd.DataFrame(dict(sym="AAAUSDT", minute=mins, mark=1.0, index=1.0, funding=np.where(np.arange(300) >= 200, -0.006, 0.0)))
    fb = pd.DataFrame(dict(sym="AAAUSDT", minute=mins, fb_bid_usd_05=np.where(np.arange(300) >= 250, 2.0, 1.0), fb_ask_usd_05=1.0))
    T, hits = TR.detect(book, mark, {}, fb=fb, pos=None, return_hits=True)
    par = T[T.trigger == "H6_funding_le_-0.5pct"]
    var = T[T.trigger == "H6_funding_le_-0.5pct|c"]
    assert len(par) == 1 and int(par.minute.iloc[0]) == int(mins[200])
    assert bool(par.cond_depth_scored.iloc[0]) and par.cond_depth_ok.iloc[0] == True  # noqa: E712
    assert set(m for _, m in hits["H6_funding_le_-0.5pct|c"]) <= set(m for _, m in hits["H6_funding_le_-0.5pct"])
    assert len(var) == 1 and (var.family == "H6").all()


def test_controls_eligible_excludes_family_hits_and_other_days():
    bm = T0 + np.arange(-100, 1500) * M
    hits = np.array([T0 + 600 * M], np.int64)
    el = CT.eligible(bm, T0, hits, rec_start_ms=T0)
    assert el.min() >= T0 and el.max() < T0 + 1440 * M
    assert not ((el >= T0 + 540 * M) & (el <= T0 + 660 * M)).any()
    assert len(el) == 1440 - 121


def test_controls_draw_deterministic_and_sized():
    el = np.arange(1000, dtype=np.int64) * M
    a, b = CT.draw("H1|X|t", el), CT.draw("H1|X|t", el)
    assert a == b and len(a) == 5 and len(set(a)) == 5
    assert CT.draw("H1|X|u", el) != a
    assert CT.draw("H1|X|t", el[:3]) == [int(x) for x in el[:3]]


def _hours(day0, n_hours, first=-1):
    return {day0 + h * 3_600_000 for h in range(first, n_hours)}


NOW = T0 + 30 * 60 * M                                  # 30 h after the day start: no gap is permanent yet


def test_controls_for_rows_and_finality():
    mins = T0 + np.arange(1440) * M
    T = pd.DataFrame(dict(trigger=["H1_liq_long_cascade"], direction=["LONG"], pair=["AAAUSDT"], minute=[int(mins[600])],
                          trigger_id=["H1_liq_long_cascade|AAAUSDT|x"], family=["H1"]))
    hits = {"H1_liq_long_cascade": [("AAAUSDT", int(mins[600]))], "H1_liq_long_cascade|c": [("AAAUSDT", int(mins[900]))]}
    C = CT.controls_for(T, hits, {"AAAUSDT": mins}, TR.family, T0 - 5 * 3_600_000, _hours(T0, 25), NOW)
    assert len(C) == 5 and (C.direction == "LONG").all() and not C.ctrl_final.any()       # D+1 01:00 is missing (and recent)
    assert not ((C.minute - int(mins[900])).abs() <= 60 * M).any()                      # the variant's hits exclude too (same family)
    C2 = CT.controls_for(T, hits, {"AAAUSDT": mins}, TR.family, T0 - 5 * 3_600_000, _hours(T0, 26), NOW)
    assert C2.ctrl_final.all() and list(C2.minute) == list(C.minute) and (C2.ctrl_day_missing_hours == 0).all()
    no_prev = _hours(T0, 26, first=0)                                                    # D−1 23:00 missing
    assert not CT.controls_for(T, hits, {"AAAUSDT": mins}, TR.family, T0 - 5 * 3_600_000, no_prev, NOW).ctrl_final.any()
    hole = _hours(T0, 26) - {T0 + 7 * 3_600_000}
    assert not CT.controls_for(T, hits, {"AAAUSDT": mins}, TR.family, T0 - 5 * 3_600_000, hole, NOW).ctrl_final.any()
    late = CT.controls_for(T, hits, {"AAAUSDT": mins}, TR.family, T0 - 5 * 3_600_000, hole, T0 + 8 * 3_600_000 + CT.GAP_PERMANENT_MS)
    assert late.ctrl_final.all() and (late.ctrl_day_missing_hours == 1).all()          # a permanent gap: final, counted


def test_day_status_before_recording_counts_present():
    rec = T0 + 23 * 3_600_000 + 23 * M                                                   # the 11-19 23:23 case
    hours = {T0 + h * 3_600_000 for h in range(23, 26)}
    assert CT.day_status(T0, hours, rec, NOW) == (True, 0)


def test_controls_rec_start_and_zero_eligible_stub():
    mins = T0 + np.arange(1440) * M
    el = CT.eligible(mins, T0, None, rec_start_ms=T0 + 1000 * M)
    assert el.min() >= T0 + 1000 * M
    T = pd.DataFrame(dict(trigger=["H6_funding_ge_+0.1pct"], direction=["SHORT"], pair=["ZZZUSDT"], minute=[int(mins[10])],
                          trigger_id=["H6|ZZZ|x"], family=["H6"]))
    C = CT.controls_for(T, {}, {}, TR.family, T0, _hours(T0, 26), NOW)
    assert len(C) == 1 and C.ctrl_k.iloc[0] == -1 and C.n_eligible.iloc[0] == 0 and C.minute.isna().all()


def test_ctrl_id_embeds_the_drawn_minute():
    mins = T0 + np.arange(1440) * M
    T = pd.DataFrame(dict(trigger=["H1_liq_long_cascade"], direction=["LONG"], pair=["AAAUSDT"], minute=[int(mins[600])],
                          trigger_id=["H1|A|x"], family=["H1"]))
    a = CT.controls_for(T, {}, {"AAAUSDT": mins}, TR.family, T0, _hours(T0, 26), NOW)
    b = CT.controls_for(T, {}, {"AAAUSDT": mins[:700]}, TR.family, T0, _hours(T0, 26), NOW)        # another eligible set
    assert all(str(int(m)) in i for i, m in zip(a.ctrl_id, a.minute)) and set(a.ctrl_id) != set(b.ctrl_id)


def test_hour_starts_exact():
    import sources as SRC
    d = int(pd.Timestamp("2026-11-22", tz="UTC").value // 1_000_000)
    assert SRC.hour_starts(["features/2026-11-22/01.csv.gz", "features/2026-11-22/23.csv.gz", "x/features/2026-11-22/02.csv.gz",
                            "features/2026-11-22/24.csv.gz"]) == [d + 3_600_000, d + 23 * 3_600_000]


def test_controls_golden_draw():
    """pins numpy's stream: a numpy upgrade that changes default_rng(...).choice fails here instead of silently redrawing."""
    el = T0 + np.arange(1440, dtype=np.int64) * M
    got = [(m - T0) // M for m in CT.draw("H1_liq_long_cascade|AAAUSDT|2026-11-25T10:00", el)]
    assert got == GOLDEN


GOLDEN = [7, 102, 706, 773, 1163]   # numpy 2.4.1


def test_controls_exclusion_boundary():
    bm = T0 + np.arange(1440) * M
    el = CT.eligible(bm, T0, np.array([T0 + 600 * M], np.int64), T0)
    assert T0 + 540 * M not in el and T0 + 539 * M in el and T0 + 660 * M not in el and T0 + 661 * M in el


def test_h7_causal_on_the_trigger_minute_and_later():
    mk = _mark(ret_at=30, ret=-2.5)
    q = T0 + 32 * M
    oi = pd.DataFrame(dict(minute=[q], oi_chg_5m=[-4.0]))
    base = TR.h7_hits(oi, mk)
    mk2 = mk.copy()
    mk2.loc[mk2.minute >= q + M, "mark"] = 5.0                         # the trigger minute m = q + 1 and later: never read
    assert TR.h7_hits(oi, mk2) == base


@pytest.mark.parametrize("d_oi,ret,key", [(-3.0, -2.0, "H7a_oi_flush_dump"), (-3.0, 2.0, "H7b_oi_flush_pump"), (5.0, 2.0, "H7c_oi_build_pump")])
def test_h7_thresholds_inclusive(d_oi, ret, key):
    mk = _mark(ret_at=30, ret=ret)
    oi = pd.DataFrame(dict(minute=[T0 + 32 * M], oi_chg_5m=[d_oi]))
    assert TR.h7_hits(oi, mk)[key] == [T0 + 33 * M]
    oi2 = pd.DataFrame(dict(minute=[T0 + 32 * M], oi_chg_5m=[d_oi * 0.99]))
    assert TR.h7_hits(oi2, mk)[key] == []


def test_h7_mark_age_two_minutes():
    mins = T0 + np.r_[np.arange(0, 26), np.arange(30, 60)] * M              # marks missing 26–29
    mk = pd.DataFrame(dict(minute=mins, mark=np.where(mins >= T0 + 30 * M, 0.97, 1.0)))
    oi = pd.DataFrame(dict(minute=[T0 + 32 * M, T0 + 34 * M], oi_chg_5m=[-4.0, -4.0]))   # spans 27→32 (stale 2 → ok? 27−25 = 2) and 29→34
    h = TR.h7_hits(oi, mk)["H7a_oi_flush_dump"]
    assert h == [T0 + 33 * M]                                              # mark(27) from 25 (2 min old) ok; mark(29) from 25 (4 min) → unscored


def test_depth_side_own_row_and_future():
    n = 200
    mins = T0 + np.arange(n) * M
    bid = np.r_[np.full(30, 1.0), np.full(30, 3.0), [1.5], np.full(n - 61, 2.0)]
    fb = pd.DataFrame(dict(minute=mins, fb_bid_usd_05=bid, fb_ask_usd_05=np.full(n, 100.0)))
    d = TR.Depth(fb)
    assert d.ok(int(mins[60]), "bid") == (True, False)                     # median of the 60 prior rows = 2 → 1.5 < 2 (own row excluded)
    assert d.ok(int(mins[60]), "ask") == (True, True)                      # the ask side reads its own column
    fb2 = fb.copy()
    fb2.loc[61:, "fb_bid_usd_05"] = 0.0                                    # rows at m + 1, m + 2 … never read
    assert TR.Depth(fb2).ok(int(mins[60]), "bid") == d.ok(int(mins[60]), "bid")


def test_variants_need_the_condition_to_hold_not_just_scored():
    mins = T0 + np.arange(300) * M
    book = _book("AAAUSDT", mins)
    mark = pd.DataFrame(dict(sym="AAAUSDT", minute=mins, mark=1.0, index=1.0, funding=np.where(np.arange(300) >= 200, -0.006, 0.0)))
    fb = pd.DataFrame(dict(sym="AAAUSDT", minute=mins, fb_bid_usd_05=np.where(np.arange(300) >= 150, 0.5, 1.0), fb_ask_usd_05=1.0))
    T = TR.detect(book, mark, {}, fb=fb)
    par = T[T.trigger == "H6_funding_le_-0.5pct"]
    assert bool(par.cond_depth_scored.iloc[0]) and par.cond_depth_ok.iloc[0] == False   # noqa: E712 — scored, does not hold
    assert (T.trigger == "H6_funding_le_-0.5pct|c").sum() == 0


class _Mk:
    def state(self, T):
        return {"mk_has_row": False}


def _h_env(monkeypatch, n=2000, buy=(5.0, 9.0), sell=(5.0, 9.0), px=None):
    import build as BLD
    import sources as SRC
    import state as ST
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    t = T0 + np.arange(n) * M
    p = np.full(n, 100.0) if px is None else px
    k = pd.DataFrame(dict(open_ms=t, o=p, h=p * 1.005, l=p * 0.995, c=p, qv=1.0))
    calls = []
    monkeypatch.setattr(BLD, "klines_for", lambda sym, a, b, now: (calls.append(sym), k[(k.open_ms >= a) & (k.open_ms < b)].reset_index(drop=True))[1])
    book = pd.DataFrame(dict(sym="AAAUSDT", minute=t, n_book=1.0, buy_usd=0.0, sell_usd=0.0, cost_buy_2500=buy[0], cost_buy_10000=buy[1],
                             cost_sell_2500=sell[0], cost_sell_10000=sell[1], spread_bps=1.0, imb_05=0.0, reach_bid_pct=1.0, reach_ask_pct=1.0,
                             mid=100.0, bid_usd_02=1.0, ask_usd_02=1.0, bid_usd_05=1.0, ask_usd_05=1.0, bid_usd_1=1.0, ask_usd_1=1.0))
    coin = ST.Coin("AAAUSDT", book, book.iloc[:0].assign(mark=[], index=[], funding=[]),
                   pd.DataFrame(columns=["minute", "liq_long_usd", "liq_short_usd"]))
    return BLD, {"AAAUSDT": coin}, calls


def _rows(direction="LONG", minutes=(100,), ids=None, final=True):
    mins = [float(T0 + m * M) if m is not None else np.nan for m in minutes]
    Tt = pd.DataFrame(dict(trigger="H7a_oi_flush_dump", direction=direction, pair="AAAUSDT", minute=mins,
                           ctrl_id=ids or [f"id{i}" for i in range(len(mins))], ctrl_final=final))
    Tt["signal_time"] = pd.to_datetime(Tt.minute + M, unit="ms", utc=True)
    return Tt


def test_price_h_table_wiring(monkeypatch, tmp_path):
    """triggers / controls priced through one function: entry = the open of minute m + 2, a stub stays unpriced, final rows carried."""
    BLD, coins, calls = _h_env(monkeypatch)
    Tt = _rows(minutes=(100, None))
    out = BLD.price_h_table(Tt, "ctrl_id", None, None, coins, _Mk(), None, "hk", T0 + 5000 * M)
    assert out.h_final.iloc[0] and out.h_cost_src.iloc[0] == "route_b_book" and not out.h_final.iloc[1] and out.h_cost_src.iloc[1] == ""
    assert int(out.h_entry_ms.iloc[0]) == T0 + 102 * M
    assert "g_tp1_sl1_net" in out.columns and "btc_d_trend" in out.columns
    p = tmp_path / "prev.pkl.gz"
    out.to_pickle(p)
    n0 = len(calls)
    BLD.price_h_table(Tt, "ctrl_id", None, str(p), coins, _Mk(), None, "hk", T0 + 5000 * M, carry_if=lambda r: r.ctrl_final)
    assert len(calls) == n0                                               # carried, nothing re-fetched
    for bad in (dict(ctrl_final=False), dict(h_carry_key="other"), dict(code_version=BLD.CODE_VERSION - 1), dict(h_final=False)):
        out.assign(**bad).to_pickle(p)
        n1 = len(calls)
        BLD.price_h_table(Tt, "ctrl_id", None, str(p), coins, _Mk(), None, "hk", T0 + 5000 * M, carry_if=lambda r: r.ctrl_final)
        assert len(calls) > n1, bad                                       # re-priced
    out.assign(**{"fwd_5m_pct": 123.0, "h_slip_entry_pct": 9.0}).to_pickle(p)
    c = BLD.price_h_table(Tt, "ctrl_id", None, str(p), coins, _Mk(), None, "hk", T0 + 5000 * M, carry_if=lambda r: r.ctrl_final)
    assert c.fwd_5m_pct.iloc[0] == 123.0 and c.h_slip_entry_pct.iloc[0] == 9.0   # fwd_ / h_ columns carried too


def test_price_h_short_legs_and_sign(monkeypatch):
    px = np.r_[np.full(102, 100.0), np.linspace(100, 90, 1898)]           # falls after the entry: a SHORT gains
    BLD, coins, _ = _h_env(monkeypatch, buy=(4.0, 4.0), sell=(8.0, 8.0), px=px)
    out = BLD.price_h_table(_rows(direction="SHORT"), "ctrl_id", None, None, coins, _Mk(), None, "hk", T0 + 5000 * M)
    assert out.h_slip_entry_pct.iloc[0] == pytest.approx(0.08) and out.h_slip_exit_pct.iloc[0] == pytest.approx(0.04)
    assert out.fwd_240m_pct.iloc[0] > 0


def test_price_h_cost_sources_and_short_path(monkeypatch):
    BLD, coins, _ = _h_env(monkeypatch, sell=(5.0, np.nan))
    out = BLD.price_h_table(_rows(), "ctrl_id", None, None, coins, _Mk(), None, "hk", T0 + 5000 * M)
    assert out.h_cost_src.iloc[0] == "book_unwalkable" and out.h_slip_exit_pct.iloc[0] == pytest.approx(0.10)
    out2 = BLD.price_h_table(_rows(), "ctrl_id", None, None, {}, _Mk(), None, "hk", T0 + 5000 * M)
    assert out2.h_cost_src.iloc[0] == "no_book"
    BLD3, coins3, _ = _h_env(monkeypatch, n=300)
    out3 = BLD3.price_h_table(_rows(), "ctrl_id", None, None, coins3, _Mk(), None, "hk", T0 + 5000 * M)
    assert not out3.h_final.iloc[0]                                       # < 240 bars after the entry


def test_short_variant_holds_on_the_ask_side():
    n = 200
    mins = T0 + np.arange(n) * M
    fb = pd.DataFrame(dict(minute=mins, fb_bid_usd_05=np.r_[np.full(100, 10.0), np.full(100, 1.0)],
                           fb_ask_usd_05=np.r_[np.full(100, 1.0), np.full(100, 10.0)]))
    d = TR.Depth(fb)
    assert d.ok(int(mins[150]), "ask") == (True, True) and d.ok(int(mins[150]), "bid") == (True, False)


def test_h_cost_src_takes_only_the_three_values(monkeypatch):
    """PREREG Addendum 3: route_b_book · book_unwalkable · no_book (the interim 'default' label is gone; CODE_VERSION ≥ 7 re-prices)."""
    import build as BLD
    assert BLD.CODE_VERSION >= 7
    vals = set()
    for kw in (dict(), dict(sell=(5.0, np.nan))):
        B, coins, _ = _h_env(monkeypatch, **kw)
        vals |= set(B.price_h_table(_rows(), "ctrl_id", None, None, coins, _Mk(), None, "hk", T0 + 5000 * M).h_cost_src)
    B, _, _ = _h_env(monkeypatch)
    vals |= set(B.price_h_table(_rows(), "ctrl_id", None, None, {}, _Mk(), None, "hk", T0 + 5000 * M).h_cost_src)
    assert vals == {"route_b_book", "book_unwalkable", "no_book"}
