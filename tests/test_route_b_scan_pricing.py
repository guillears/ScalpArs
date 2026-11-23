"""Route B weekly-scan PREREG build item 2 — the sleeve NET uses the FULL-book cost ($6k, entry and exit) where the coin was in the full
book, else the 20-level cost flagged "lower-bound cost". No network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import build as BLD      # noqa: E402
import common as C       # noqa: E402
import sources as SRC    # noqa: E402
import state as ST       # noqa: E402
from tests.test_route_b_signals import M, T0, _crash_klines, _sig_row  # noqa: E402


def _coin_with(fb_buy=None, book_buy=None, ask5=1e6, fb_until=None):
    mins = T0 + np.arange(-60, 400) * M
    n = len(mins)
    book = pd.DataFrame(dict(sym="KUSDT", minute=mins, n_book=1.0, buy_usd=0.0, sell_usd=0.0,
                             cost_buy_2500=np.full(n, book_buy if book_buy is not None else np.nan),
                             cost_buy_10000=np.full(n, book_buy if book_buy is not None else np.nan),
                             cost_sell_2500=5.0, cost_sell_10000=5.0))
    mark = pd.DataFrame(dict(sym="KUSDT", minute=mins, mark=1.0, index=1.0, funding=0.0))
    liq = pd.DataFrame(dict(sym="KUSDT", minute=mins[:0], liq_long_usd=[], liq_short_usd=[]))
    fb = None
    if fb_buy is not None:
        fb = pd.DataFrame(dict(sym="KUSDT", minute=mins, n_fbook=10.0, fb_cost_buy_6000=fb_buy, fb_cost_sell_6000=3.0,
                               fb_ask_usd_5=ask5, fb_bid_usd_5=1e6))
    if fb is not None and fb_until is not None:
        fb = fb[fb.minute < fb_until]
    return ST.Coin("KUSDT", book, mark, liq, fb, None)


FIT = dict(fb_bid_usd_5=1e6, fb_ask_usd_5=1e6)


def _stop_spec():
    return dict(C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"], sl_cap=0.05, sl_atr=0.05)    # a STOP exit (a taker buy-back)


@pytest.fixture(autouse=True)
def _nofund(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))


def test_full_book_both_legs():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=20.0, book_buy=40.0)
    r = _sig_row(fb_cost_sell_6000=3.0, rb_cost6k_sell_bps=9.0, **FIT)
    o = BLD.price_signal(r, _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_reason"] == "STOP_LOSS"
    assert o["out_entry_slip_src"] == "full_book" and o["out_entry_slip_pct"] == pytest.approx(0.03)
    assert o["out_exit_slip_src"] == "full_book_at_exit" and o["out_exit_slip_pct"] == pytest.approx(0.20)
    assert o["out_cost_flag"] == "full_book"


def test_twenty_level_fallback_is_flagged():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=None, book_buy=40.0)
    o = BLD.price_signal(_sig_row(rb_cost6k_sell_bps=9.0), _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_entry_slip_src"] == "route_b_book" and o["out_exit_slip_src"] == "route_b_book_at_exit"
    assert o["out_cost_flag"] == "lower-bound cost"


def test_full_book_size_not_fitting_falls_back():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=np.nan, book_buy=40.0)                          # $6k never fitted within 5 % → NaN in the full book
    o = BLD.price_signal(_sig_row(fb_cost_sell_6000=np.nan, rb_cost6k_sell_bps=9.0, **FIT), _stop_spec(), k, coin, T0 + 10_000 * M,
                         T0 + 9_000 * M)
    assert o["out_exit_slip_src"] == "route_b_book_at_exit" and o["out_cost_flag"] == "lower-bound cost"


def test_maker_tp_with_full_book_entry_is_full_book():
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    o = BLD.price_signal(_sig_row(fb_cost_sell_6000=3.0, **FIT), sp, _crash_klines(), _coin_with(fb_buy=20.0), T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_reason"] == "TP" and o["out_exit_slip_src"] == "maker_tp" and o["out_cost_flag"] == "full_book"


def test_cost_flag_table():
    assert BLD.cost_flag("in_fill_price", "full_book_at_exit") == "full_book"
    assert BLD.cost_flag("full_book", "full_book_at_signal") == "full_book"
    assert BLD.cost_flag("full_book", "route_b_book_at_exit") == "lower-bound cost"
    assert BLD.cost_flag("route_b_book", "full_book_at_exit") == "lower-bound cost"
    assert BLD.cost_flag("default", "maker_tp") == "lower-bound cost"
    assert BLD.cost_flag("in_fill_price", "maker_tp") == "no_book_leg"
    assert BLD.cost_flag("full_book", "maker_tp", priced=False) == ""


def test_full_book_signal_fallback_after_data_end_not_final():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=None, book_buy=None)
    r = _sig_row(fb_cost_buy_6000=7.0, fb_cost_sell_6000=3.0, **FIT)
    o = BLD.price_signal(r, _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 5 * M)
    assert o["out_exit_slip_src"] == "full_book_at_signal" and o["out_exit_slip_pct"] == pytest.approx(0.07) and not o["out_final"]
    o2 = BLD.price_signal(r, _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o2["out_exit_slip_src"] == "full_book_at_signal" and o2["out_final"]


def test_exit_cost_read_before_data_end_never_final():
    """the exit minute is past the data end but a ≤ 2-min-old row exists → whatever the source, the row stays open (review I1)."""
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=20.0, book_buy=40.0)
    o = BLD.price_signal(_sig_row(**FIT), _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    xm = int(o["out_exit_ms"])
    o2 = BLD.price_signal(_sig_row(**FIT), _stop_spec(), k, coin, T0 + 10_000 * M, xm)
    assert o2["out_exit_slip_src"] == "full_book_at_exit" and not o2["out_final"]
    o3 = BLD.price_signal(_sig_row(has_fb=True, **FIT), _stop_spec(), k, _coin_with(fb_buy=20.0, book_buy=40.0, fb_until=xm - 5 * M),
                          T0 + 10_000 * M, T0 + 9_000 * M, fb_end_ms=xm - 5 * M)
    assert o3["out_exit_slip_src"] == "route_b_book_at_exit" and not o3["out_final"]          # the full book lags: wait for it


def test_stale_full_book_at_exit_falls_back():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    o = BLD.price_signal(_sig_row(**FIT), _stop_spec(), k, _coin_with(fb_buy=20.0, book_buy=40.0), T0 + 10_000 * M, T0 + 9_000 * M)
    xm = int(o["out_exit_ms"])
    coin = _coin_with(fb_buy=20.0, book_buy=40.0, fb_until=xm - 3 * M + 1)                       # newest fb row 3 min old
    o2 = BLD.price_signal(_sig_row(**FIT), _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o2["out_exit_slip_src"] == "route_b_book_at_exit"


def test_thin_full_book_side_falls_back():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    coin = _coin_with(fb_buy=20.0, book_buy=40.0, ask5=4000.0)                                    # asks within 5 % < $6k
    r = _sig_row(fb_cost_sell_6000=3.0, rb_cost6k_sell_bps=9.0, fb_bid_usd_5=5000.0, fb_ask_usd_5=1e6)
    o = BLD.price_signal(r, _stop_spec(), k, coin, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_entry_slip_src"] == "route_b_book" and o["out_exit_slip_src"] == "route_b_book_at_exit"
    assert o["out_cost_flag"] == "lower-bound cost"


def test_negative_full_book_cost_clamped_and_taken_fill_wins():
    k = _crash_klines()
    k["h"] = k.o * 1.01
    o = BLD.price_signal(_sig_row(fb_cost_sell_6000=-2.0, **FIT), _stop_spec(), k, _coin_with(fb_buy=-5.0), T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_entry_slip_pct"] == 0.0 and o["out_exit_slip_pct"] == 0.0
    fill = pd.Timestamp(T0 + 31 * M, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    r = _sig_row(status="OPENED", fill_entry_price=float(k.o[k.open_ms == T0 + 31 * M].iloc[0]), fill_opened_at=fill,
                 fb_cost_sell_6000=3.0, **FIT)
    o2 = BLD.price_signal(r, _stop_spec(), k, _coin_with(fb_buy=20.0), T0 + 10_000 * M, T0 + 9_000 * M)
    assert o2["out_entry_slip_src"] == "in_fill_price" and o2["out_entry_slip_pct"] == 0.0


def test_long_sides_mapping():
    """a LONG buys in (asks) and sells out (bids)."""
    k = _crash_klines()
    k["l"] = k.o * 0.99
    k["h"] = k.o * 1.0005
    sp = dict(C.exit_specs(C.load_thresholds())["FLATBASE_SHORT"], side=1, tp=50.0, sl=0.05)
    r = _sig_row(sleeve="FLATBASE_SHORT", direction="LONG", fb_cost_buy_6000=4.0, fb_cost_sell_6000=8.0, **FIT)
    o = BLD.price_signal(r, sp, k, _coin_with(fb_buy=20.0), T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_entry_slip_pct"] == pytest.approx(0.04)                                         # fb_cost_buy at the signal
    assert o["out_exit_slip_src"] == "full_book_at_exit" and o["out_exit_slip_pct"] == pytest.approx(0.03)   # fb_cost_sell at exit


def test_can_carry_refuses_any_taker_exit_at_or_past_data_end():
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    key = BLD.carry_key(sp, None, "BLOCKED:X")
    base = dict(code_version=BLD.CODE_VERSION, carry_key=key, out_priced=True, out_final=True, out_funding_ok=True)
    for src in BLD.TAKER_EXIT_SRCS:
        assert not BLD.can_carry(dict(base, out_exit_slip_src=src, out_exit_ms=3.0, out_data_end_ms=2.0), key), src
        assert BLD.can_carry(dict(base, out_exit_slip_src=src, out_exit_ms=1.0, out_data_end_ms=2.0), key), src
    assert BLD.can_carry(dict(base, out_exit_slip_src="maker_tp", out_exit_ms=3.0, out_data_end_ms=2.0), key)
    assert BLD.CODE_VERSION >= 4
