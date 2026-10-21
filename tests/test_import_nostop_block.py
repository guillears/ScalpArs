"""📥 / 🚫 / ⛔ (DECISION_LOG 343, operator 2026-10-21): the import ADOPTS the app's TP (cancel the app's TP / SL, re-create the TP as the bot's
own), the No-stop exit with its −15 % emergency stop, and "block bot trades while this is open". Real engine paths (import_binance_position,
import_preview, check_realtime_stop_loss, backstop_reprice_sweep, _rtp_guard_open_residual, open_position's block, set_exit_override,
lift endpoint, /api/performance's block state) on file SQLite with a stubbed binance_service — no network. UI / export wiring pinned
statically and run in node."""
import asyncio, datetime as dt, inspect, json, os, re, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import pytest

from test_manual_import import _harness, _pos, _kinds, _nothing_written   # the DL 328 import harness (real engine, stubbed exchange)
from services.position_import import classify_app_orders, import_tp_pct_from_price, describe_open_orders
from services.manual_nostop import (manual_emergency_pct, emergency_trigger_price, emergency_liq_refusal, emergency_clamp_to_liq,
                                    emergency_software_hit, is_nostop, MANUAL_NOSTOP_SOFT_MARGIN_PCT)
from services.exec_shadow import resting_tp_price

UI = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
MAIN = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
TF = 0.00045


def _algo(id_, typ, side, trig, close=True, qty=0.0, reduce=False):
    return dict(kind="algo", id=id_, type=typ, side=side, qty=qty, price=0.0, trigger_price=trig, reduce_only=reduce, close_position=close)


def _reg(id_, typ, side, price, qty, reduce=True, trig=0.0):
    return dict(kind="regular", id=id_, type=typ, side=side, qty=qty, price=price, trigger_price=trig, reduce_only=reduce, close_position=False)


APP_TP = _algo("tp-9", "TAKE_PROFIT_MARKET", "SELL", 2.2)
APP_SL = _algo("sl-9", "STOP_MARKET", "SELL", 1.8)
OUR_STOP = _algo("algo-1", "STOP_MARKET", "SELL", 1.95)


SEEN_ALL = ["algo:tp-9", "algo:sl-9", "regular:lim-1", "regular:old-77", "regular:5", "regular:6", "algo:t", "algo:tr"]   # what the pop-up showed


def _imp(exit_mode="FLOOR", sl=None, tp=None, **kw):
    kw.setdefault("app_orders_seen", SEEN_ALL)
    async def fn(eng, db):
        return await eng.import_binance_position(db, "FOOUSDT", kw.pop("direction", "LONG"), exit_mode, sl, tp, **kw)
    return fn


def _liq(p, liq):
    p = dict(p); p["liquidation_price"] = liq
    return p


# ── pure helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_classify_app_orders():
    c = classify_app_orders([APP_TP, APP_SL], "LONG", 2.0, 1000.0)
    assert [o["id"] for o in c["tp"]] == ["tp-9"] and [o["id"] for o in c["sl"]] == ["sl-9"] and c["other"] == [] and c["refusal"] is None
    assert c["tp_price"] == 2.2 and c["tp_qty"] == 1000.0 and c["tp_partial"] is False                    # close-all = the whole position
    part = classify_app_orders([_reg("5", "LIMIT", "SELL", 2.3, 400.0)], "LONG", 2.0, 1000.0)              # a reduce-only TP limit for 40 %
    assert part["tp_price"] == 2.3 and part["tp_qty"] == 400.0 and part["tp_partial"] is True
    sh = classify_app_orders([_algo("t", "TAKE_PROFIT_MARKET", "BUY", 1.6), _algo("s", "STOP_MARKET", "BUY", 2.4)], "SHORT", 2.0, 10.0)
    assert sh["tp_price"] == 1.6 and len(sh["sl"]) == 1 and sh["refusal"] is None
    for bad, why in ((_reg("1", "LIMIT", "BUY", 1.9, 5.0, reduce=False), "opening side"),                  # an entry / add limit
                     (_reg("2", "LIMIT", "SELL", 2.3, 5.0, reduce=False), "not reduce-only"),
                     (_algo("3", "TAKE_PROFIT_MARKET", "SELL", 1.9), "loss side"),                           # a "TP" below a LONG's entry
                     (_algo("4", "TRAILING_STOP_MARKET", "SELL", 1.9), "not a plain TP / SL"),
                     (_algo("5", "TAKE_PROFIT_MARKET", "SELL", 0.0), "could not be read")):
        r = classify_app_orders([APP_TP, bad], "LONG", 2.0, 1000.0)
        assert len(r["other"]) == 1 and why in r["refusal"] and "nothing changed" in r["refusal"], (bad, r["refusal"])
    two = classify_app_orders([APP_TP, _reg("6", "LIMIT", "SELL", 2.5, 1000.0)], "LONG", 2.0, 1000.0)
    assert len(two["tp"]) == 2 and two["tp_price"] is None and two["refusal"] is None
    assert describe_open_orders(None) == "the open orders could not be read" and describe_open_orders([]) == "no open orders"
    assert "TAKE_PROFIT_MARKET SELL @ 2.2 (close-all, algo #tp-9)" in describe_open_orders([APP_TP])


def test_tp_price_to_pct_round_trips_to_the_same_limit():
    """The adopted TP % (net of the taker entry fee estimate and a MAKER exit) re-creates the app's exact price as the bot's resting TP."""
    # (review #5) the % rounds UP: the bot's limit is never WORSE than the app's price (LONG never below, SHORT never above) — incl. a
    # high-priced pair with a tiny tick, where a 4-dp cut used to land several ticks short — and beyond it by ≤ 0.0001 % of price + 1 tick
    for d, e, px, tick in (("LONG", 2.0, 2.2, 0.0001), ("SHORT", 2.0, 1.6, 0.0001), ("LONG", 0.004211, 0.0047, 0.000001),
                           ("LONG", 60000.0, 61234.56, 0.01), ("SHORT", 60000.0, 58765.43, 0.01)):
        pct = import_tp_pct_from_price(d, e, px, TF * 100, 0.00018)
        assert pct and pct > 0
        lim = resting_tp_price(d, e, TF * 100, pct, 0.00018, tick)
        slack = px * 1.2e-6 + tick + 1e-12
        if d == "LONG":
            assert px - 1e-12 <= lim <= px + slack, (d, e, px, lim)
        else:
            assert px - slack <= lim <= px + 1e-12, (d, e, px, lim)
    assert import_tp_pct_from_price("LONG", 2.0, 2.0005, TF * 100, 0.00018) is None                       # no profit after fees


def test_emergency_helpers():
    class TH:
        manual_emergency_stop_pct = 15.0
    assert manual_emergency_pct(TH, "NOSTOP", True, None) == 15.0 and manual_emergency_pct(TH, "NOSTOP", True, "12.5") == 12.5
    assert manual_emergency_pct(TH, "NOSTOP", False, 15) is None and manual_emergency_pct(TH, "FLOOR", True, 15) is None
    for bad in ("x", 0, 95, 120):
        with pytest.raises(ValueError):
            manual_emergency_pct(TH, "NOSTOP", True, bad)
    assert abs(emergency_trigger_price(2.0, "LONG", 15) - 1.7) < 1e-12 and abs(emergency_trigger_price(2.0, "SHORT", 15) - 2.3) < 1e-12
    assert emergency_liq_refusal("P", "LONG", 1.7, 1.2) is None and "at or below the liquidation price" in emergency_liq_refusal("P", "LONG", 1.7, 1.7)
    assert emergency_liq_refusal("P", "LONG", 1.7, 0.0) is None                                        # cross margin: no liquidation price
    assert "did not report the liquidation price" in emergency_liq_refusal("P", "LONG", 1.7, None)
    assert "at or above the liquidation price" in emergency_liq_refusal("P", "SHORT", 2.3, 2.25)
    assert "did not report" in emergency_liq_refusal("P", "SHORT", 2.3, 0.0)
    assert emergency_clamp_to_liq(2.0, "LONG", 15, 1.8, 0.92) == round(0.1 * 100 * 0.92 - MANUAL_NOSTOP_SOFT_MARGIN_PCT, 4)   # 10 % → 9.2 − 0.3 (mirror inside too)
    assert emergency_clamp_to_liq(2.0, "LONG", 14.9, 1.7, 0.92) is not None                           # (review #8) the stop fits, its +0.3 mirror does not
    assert "software back-up line" in emergency_liq_refusal("P", "LONG", emergency_trigger_price(2.0, "LONG", 14.9), 1.7,
                                                             emergency_trigger_price(2.0, "LONG", 14.9 + MANUAL_NOSTOP_SOFT_MARGIN_PCT))
    assert emergency_clamp_to_liq(2.0, "LONG", 15, 1.2, 0.92) is None and emergency_clamp_to_liq(2.0, "LONG", 15, None, 0.92) is None
    assert emergency_software_hit("LONG", 2.0, 1.70, 15, True) and not emergency_software_hit("LONG", 2.0, 1.701, 15, True)
    assert not emergency_software_hit("LONG", 2.0, 1.70, 15, False)                                      # live: the exchange stop acts first
    assert emergency_software_hit("LONG", 2.0, 2.0 * (1 - (15 + MANUAL_NOSTOP_SOFT_MARGIN_PCT) / 100), 15, False)
    assert emergency_software_hit("SHORT", 2.0, 2.3, 15, True) and not emergency_software_hit("LONG", 2.0, 0.5, None, True)
    assert is_nostop("MANUAL", "NOSTOP") and not is_nostop("MOMENTUM", "NOSTOP") and not is_nostop("MANUAL", "FLOOR")


def test_row_backstop_pct_never_gives_a_nostop_row_a_stop():
    import services.trading_engine as T
    th = T.config.trading_config.thresholds
    base = dict(entry_strategy="MANUAL", leverage=5.0, pattern_fixed_sl_pct=None)
    assert T.row_backstop_pct(th, dict(base, manual_exit_mode="NOSTOP", manual_emergency_stop_pct=None)) is None
    assert T.row_backstop_pct(th, dict(base, manual_exit_mode="NOSTOP", manual_emergency_stop_pct=15.0)) == 15.0   # uncapped by the 92 % share
    assert T.row_backstop_pct(th, dict(base, manual_exit_mode="FLOOR", pattern_fixed_sl_pct=-3.0)) == T.order_backstop_pct(th, "MANUAL", "FLOOR", -3.0, 5.0)
    assert T.row_backstop_pct(th, dict(entry_strategy="MOMENTUM", leverage=20.0)) == T.order_backstop_pct(th, "MOMENTUM", None, None, 20.0)
    assert T.manual_exit_levels(th, "NOSTOP", 5, 4.0, 20) == (None, 4.0)                                 # no stop stored, an optional TP
    from services.position_add import add_exit_refusal
    assert "No-stop trade cannot be added to" in add_exit_refusal("MANUAL", "NOSTOP", dt.datetime.utcnow())   # even after an edit
    from services.resting_tp import rtp_row_target
    class R:
        entry_strategy = "MANUAL"; manual_exit_mode = "NOSTOP"; pattern_fixed_tp_pct = 3.0; exit_override_at = None
    class THR:
        resting_tp_manual_enabled = True
    assert rtp_row_target(THR, R()) == (3.0, "MANUAL_TP")                                                  # the TP rests like a FIXED one


# ── 📥 TP adoption (real import path) ─────────────────────────────────────────────────────────────────────────────────────────────
def test_adopts_the_app_tp_cancel_then_place(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP, APP_SL], [OUR_STOP]])
    o = asyncio.run(run(_imp("FLOOR", keep_app_tp=True, app_tp_price=2.2)))
    assert not isinstance(o, str), o
    assert _kinds(log) == ["positions", "open_orders", "backstop", "positions", "cancel_backstop", "cancel_backstop", "open_orders", "positions",
                           "commit", "rtp_arm", "cache", "ws_reset", "ws_sub"]                       # protection → cancel → re-read → book → TP
    assert [x[1] for x in log if x[0] == "cancel_backstop"] == ["tp-9", "sl-9"]
    o, _ = asyncio.run(read())
    want = import_tp_pct_from_price("LONG", 2.0, 2.2, TF * 100, eng._rtp_maker())
    assert o.pattern_fixed_tp_pct == want and o.manual_exit_mode == "FLOOR" and o.pattern_fixed_sl_pct == -3.0
    assert ("rtp_arm", o.id, want) in log and abs(resting_tp_price("LONG", 2.0, TF * 100, want, eng._rtp_maker(), 0.0001) - 2.2) < 1e-9
    assert o.backstop_algo_id == "algo-1" and o.manual_block_bot is False and o.manual_emergency_stop_pct is None


def test_partial_app_tp_limit_and_unticked_keep(monkeypatch, tmp_path):
    part = _reg("lim-1", "LIMIT", "SELL", 2.3, 400.0)
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[part], [OUR_STOP]])
    ret = asyncio.run(run(_imp("FIXED", sl=1.5, keep_app_tp=True)))
    assert not isinstance(ret, str), ret
    assert ret._import_tp_partial is True and ret._import_adopted_tp_price == 2.3                         # the endpoint warns: partial app TP
    o, _ = asyncio.run(read())
    assert ("cancel_regular", "lim-1") in log and o.pattern_fixed_tp_pct == import_tp_pct_from_price("LONG", 2.0, 2.3, TF * 100, eng._rtp_maker())
    assert o.quantity == 1000.0 and ("rtp_arm", o.id, o.pattern_fixed_tp_pct) in log                      # the bot's TP is armed on the whole row
    # unticked: the app TP is cancelled, no TP unless typed
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [OUR_STOP]])
    o = asyncio.run(run(_imp("FLOOR", keep_app_tp=False)))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert ("cancel_backstop", "tp-9") in log and o.pattern_fixed_tp_pct is None and ("rtp_arm", o.id, None) in log


def test_preview_shows_the_tp_and_the_partial_warning(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[_reg("lim-1", "LIMIT", "SELL", 2.3, 400.0), APP_SL])
    async def _pfs(symbol): return _liq(_pos(), 1.1)
    monkeypatch.setattr(T.binance_service, "get_position_for_symbol", _pfs)
    p = asyncio.run(eng.import_preview("FOOUSDT", "LONG"))
    assert p["error"] is None and p["tp_count"] == 1 and p["sl_count"] == 1 and p["tp_price"] == 2.3 and p["tp_partial"] is True
    assert p["tp_pct"] == import_tp_pct_from_price("LONG", 2.0, 2.3, TF * 100, eng._rtp_maker()) and p["liquidation_price"] == 1.1
    assert p["refusal"] is None and p["emergency_default_pct"] == float(T.config.trading_config.thresholds.manual_emergency_stop_pct)
    eng.is_paper_mode = True
    assert asyncio.run(eng.import_preview("FOOUSDT", "LONG"))["error"] == "import works in live mode only"


@pytest.mark.parametrize("kw,orders,expect", [
    (dict(exit_mode="MOMENTUM", keep_app_tp=True), [APP_TP], "can be kept only with Custom SL/TP, Floor stop or No stop"),
    (dict(exit_mode="FIXED", sl=1.5, tp=3.0, keep_app_tp=True), [APP_TP], "keep the app's take profit OR type a TP %"),
    (dict(exit_mode="FLOOR", keep_app_tp=True, app_tp_price=2.25), [APP_TP], "the app's take profit moved (2.25 in the pop-up, 2.2 on Binance now)"),
    (dict(exit_mode="FLOOR", keep_app_tp=True, app_tp_price=2.2), [], "no longer on Binance"),
    (dict(exit_mode="FLOOR", keep_app_tp=True), [APP_TP, _reg("6", "LIMIT", "SELL", 2.5, 1000.0)], "2 take-profit orders rest on FOOUSDT"),
    (dict(exit_mode="FLOOR", keep_app_tp=True), [_algo("t", "TAKE_PROFIT_MARKET", "SELL", 2.0005)], "books no profit after fees"),
])
def test_adoption_refusals_change_nothing(monkeypatch, tmp_path, kw, orders, expect):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=orders)
    m = kw.pop("exit_mode")
    r = asyncio.run(run(_imp(m, sl=kw.pop("sl", None), tp=kw.pop("tp", None), **kw)))
    assert isinstance(r, str) and r.startswith("ValueError") and expect in r, r
    assert [x for x in log if x[0] in ("backstop", "cancel_backstop", "cancel_regular", "commit")] == [] and _nothing_written(read)


def test_other_order_refuses_with_nothing_changed(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[APP_TP, _algo("tr", "TRAILING_STOP_MARKET", "SELL", 1.9)])
    r = asyncio.run(run(_imp("FLOOR")))
    assert "the bot cannot take over" in r and "TRAILING_STOP_MARKET" in r
    assert _kinds(log) == ["positions", "open_orders"] and _nothing_written(read)


def test_cancel_failure_reports_binance_and_keeps_the_stop(monkeypatch, tmp_path):
    """An app order that will not cancel: nothing booked, the text lists what rests on Binance, the bot's stop is KEPT (the app's SL may
    already be gone — the position is never left without the stop the operator picked)."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP, APP_SL], [OUR_STOP, APP_SL]], bk_cancel_ok=False)
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("RuntimeError") and "stopped while cancelling the app's TP/SL — nothing booked" in r
    assert "On Binance now: STOP_MARKET SELL @ 1.95 (close-all, algo #algo-1); STOP_MARKET SELL @ 1.8 (close-all, algo #sl-9)" in r
    assert "algoId=algo-1 at 1.93 was KEPT" in r                                                         # FLOOR: −3 % + 0.5 → 3.5 % below 2.0 and ("cancel_backstop", "algo-1") not in log
    assert "commit" not in _kinds(log) and _nothing_written(read)
    # the re-read itself fails → the same report, "could not be read"
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], "FAIL"])
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert "On Binance now: the open orders could not be read" in r and ("cancel_backstop", "algo-1") not in log


def test_position_closes_while_cancelling_withdraws_the_stop(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [OUR_STOP]], positions_seq=([_pos()], [_pos()], []))
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("RuntimeError") and "(now closed)" in r and "the app's TP/SL are cancelled" in r and "was withdrawn" in r
    assert ("cancel_backstop", "algo-1") in log and _nothing_written(read)
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [OUR_STOP]], positions_seq=([_pos()], [_pos()], [_pos(contracts=600.0)]))
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert "now LONG 600 @ 2" in r and "was KEPT" in r and ("cancel_backstop", "algo-1") not in log


def test_commit_failure_after_the_cancels_keeps_the_stop(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [OUR_STOP]], commit_fails=True)
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("RuntimeError") and "its app TP/SL were cancelled" in r and "was KEPT" in r and ("cancel_backstop", "algo-1") not in log


# ── 🚫 No stop (import) ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_nostop_without_emergency_places_nothing(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_SL], []])
    o = asyncio.run(run(_imp("NOSTOP", tp=4.0, emergency_stop=False, block_bot=True)))
    assert not isinstance(o, str), o
    assert [x for x in log if x[0] == "backstop"] == [] and ("cancel_backstop", "sl-9") in log                # the app SL is cancelled too
    o, _ = asyncio.run(read())
    assert o.manual_exit_mode == "NOSTOP" and o.pattern_fixed_sl_pct is None and o.pattern_fixed_tp_pct == 4.0
    assert o.manual_emergency_stop_pct is None and o.backstop_algo_id is None and o.backstop_pct is None and o.manual_block_bot is True


def test_nostop_emergency_at_15_and_liquidation_refusals(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_liq(_pos(lev=5), 1.2)],))
    o = asyncio.run(run(_imp("NOSTOP")))
    assert not isinstance(o, str), o
    assert [x for x in log if x[0] == "backstop"] == [("backstop", "FOOUSDT", "LONG", round(2.0 * 0.85, 8))]   # entry × (1 − 15 %)
    o, _ = asyncio.run(read())
    assert o.manual_emergency_stop_pct == 15.0 and o.backstop_pct == 15.0 and abs(o.backstop_trigger_price - 1.7) < 1e-12
    for liq, txt in ((1.7, "at or below the liquidation price 1.7"), (1.8, "at or below the liquidation price 1.8"),
                     (None, "did not report the liquidation price")):
        run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_liq(_pos(lev=5), liq)],))
        r = asyncio.run(run(_imp("NOSTOP")))
        assert r.startswith("ValueError") and txt in r, r
        assert [x for x in log if x[0] in ("backstop", "open_orders", "commit")] == [] and _nothing_written(read)
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_liq(_pos(side="SHORT", lev=5), 2.25)],))
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: type("K", (), {"last_price": 2.0})())
    assert "at or above the liquidation price 2.25" in asyncio.run(run(_imp("NOSTOP", direction="SHORT")))
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_liq(_pos(lev=5), 1.0)],), price=1.69)
    assert "already past its 15 % emergency level" in asyncio.run(run(_imp("NOSTOP")))


# ── 🚫 No stop: exits / sweeps never stop it out ──────────────────────────────────────────────────────────────────────────────────
def _drive_realtime(monkeypatch, row, prices, paper):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper
    closed = []

    class _Res:
        def __init__(self, o): self.o = o
        def scalar_one_or_none(self): return self.o

    class _DB:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *a, **k): return _Res(object())

    async def _close_(db, order, price, reason): closed.append(reason); return True
    monkeypatch.setattr(T, "AsyncSessionLocal", lambda: _DB())
    monkeypatch.setattr(eng, "close_position", _close_, raising=False)
    fee = float(getattr(T.config.trading_config, 'taker_fee', T.config.trading_config.trading_fee))
    out = []

    async def go():
        for px in prices:
            closed.clear(); T._rtp_rt.clear()
            T._open_orders_cache["ZZZUSDT"] = [dict({'id': 1, 'direction': "LONG", 'entry_price': 100.0, 'quantity': 10.0, 'entry_fee': 1000.0 * fee,
                                                     'confidence': 'STRONG_BUY', 'opened_at': T.datetime.utcnow(), 'entry_strategy': 'MANUAL',
                                                     'pattern_fixed_sl_pct': None, 'pattern_fixed_tp_pct': None, 'stop_loss': -0.7, 'peak_pnl': 0.0,
                                                     'trough_pnl': 0.0, 'leverage': 5.0, 'notional_value': 1000.0, 'investment': 200.0}, **row)]
            await eng.check_realtime_stop_loss("ZZZUSDT", px)
            out.append(list(closed))
        T._open_orders_cache.clear()
    asyncio.run(go())
    return out


def test_realtime_nostop_never_closes_on_a_loss_and_the_emergency_mirror(monkeypatch):
    # no emergency stop: −50 % never closes (no stop, no floor); the TP still does
    r = _drive_realtime(monkeypatch, dict(manual_exit_mode="NOSTOP", manual_emergency_stop_pct=None, pattern_fixed_tp_pct=3.0), [97.0, 80.0, 50.0, 104.0], True)
    assert r == [[], [], [], ["MANUAL_TP"]]
    # emergency 15 %: paper books it at the exchange trigger (85.0); live only 0.3 % beyond (the exchange stop acts first)
    r = _drive_realtime(monkeypatch, dict(manual_exit_mode="NOSTOP", manual_emergency_stop_pct=15.0), [86.0, 85.0], True)
    assert r == [[], ["MANUAL_SL_EMERGENCY"]]
    r = _drive_realtime(monkeypatch, dict(manual_exit_mode="NOSTOP", manual_emergency_stop_pct=15.0), [85.0, 84.8, 84.7], False)
    assert r == [[], [], ["MANUAL_SL_EMERGENCY"]]
    # a FLOOR row at the same depth closes at once (sanity: the intercept still stops the other modes)
    assert _drive_realtime(monkeypatch, dict(manual_exit_mode="FLOOR", pattern_fixed_sl_pct=-3.0), [96.5], True) == [["MANUAL_SL"]]


def test_reprice_sweep_and_residual_guard_skip_a_true_nostop_row(monkeypatch, tmp_path):
    """The re-price sweep (paper records / live re-places) and the residual guard's stop re-place never give a NOSTOP row without its
    emergency stop an exchange stop; a NOSTOP row WITH it is kept at its emergency distance (never re-centred to the bot's 2.5 %)."""
    import services.trading_engine as T, models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    e = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rp.db'}")
    S = async_sessionmaker(e, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)
    placed = []
    async def _place(pair, d, trig): placed.append((pair, trig)); return "NEW"
    monkeypatch.setattr(T.binance_service, "place_backstop_stop", _place)
    monkeypatch.setattr(T.config.trading_config.thresholds, "broker_backstop_enabled", True, raising=False)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True
    eng._record_filter_block = lambda *a, **k: None
    async def _io(coro, what): return await coro
    eng._rtp_io = _io

    async def go():
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as db:
            common = dict(direction="LONG", status="OPEN", entry_price=2.0, quantity=100.0, investment=40.0, leverage=5.0, notional_value=200.0,
                          confidence="STRONG_BUY", is_paper=True, entry_strategy="MANUAL", manual_exit_mode="NOSTOP")
            db.add(models.Order(pair="AAAUSDT", manual_emergency_stop_pct=None, **common))
            db.add(models.Order(pair="BBBUSDT", manual_emergency_stop_pct=15.0, backstop_pct=15.0, backstop_trigger_price=1.7, **common))
            await db.commit()
        out = await eng.backstop_reprice_sweep(now=1e12)
        async with S() as db:
            rows = {o.pair: o for o in (await db.execute(select(models.Order))).scalars().all()}
            eng.is_paper_mode = False
            await eng._rtp_guard_open_residual(db, rows["AAAUSDT"], "MANUAL")
        await e.dispose()
        return out, rows
    out, rows = asyncio.run(go())
    assert out == [] and rows["AAAUSDT"].backstop_pct is None and rows["BBBUSDT"].backstop_pct == 15.0
    assert placed == []


def test_every_stop_site_reads_row_backstop_pct():
    """Startup re-arm = the heal sweep (rows with no stop id), the re-price sweep, the close-failure re-place, the residual guard: every
    per-row site goes through row_backstop_pct and skips None — never order_backstop_pct on a row (a NOSTOP row would get 2.5 %)."""
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    for frag in ("_bk_pct = row_backstop_pct(config.trading_config.thresholds, _bko)",            # heal sweep (restart re-arm)
                 "continue   # 🖐 (343) a No-stop row with its emergency stop unticked carries NO exchange stop",
                 "want = row_backstop_pct(th, o)",                                                   # re-price sweep
                 "_bk_pct = row_backstop_pct(config.trading_config.thresholds, order)",              # close failure re-place
                 "_bk_pct = row_backstop_pct(th, order)"):                                           # residual guard
        assert frag in src, frag
    assert not re.search(r"order_backstop_pct\([^)]*_bko\.", src) and "order_backstop_pct(th, o.entry_strategy" not in src
    # max hold never closes a NOSTOP row; both exit intercepts take NOSTOP into the MANUAL branch
    assert "if max_hold > 0 and order.opened_at and not is_nostop(order.entry_strategy, getattr(order, 'manual_exit_mode', None)):" in src
    assert src.count('"BULLRUN_SL", MANUAL_NOSTOP_MODE)') == 1 and src.count("'BULLRUN_SL', MANUAL_NOSTOP_MODE)") == 1
    assert src.count('"MANUAL_SL_EMERGENCY"') == 2
    import services.trading_engine as T
    assert "manual_block_active" not in inspect.getsource(T.TradingEngine.open_manual_position)
    assert "manual_block_active" not in inspect.getsource(T.TradingEngine.import_binance_position)
    assert "manual_block_active" not in inspect.getsource(T.TradingEngine.add_to_position)


def test_target_only_edit_keeps_a_nostop_row_stopless(monkeypatch, tmp_path):
    import services.trading_engine as T, models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    e = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ed.db'}")
    S = async_sessionmaker(e, expire_on_commit=False)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True
    async def _none(*a, **k): return None
    eng.update_orders_cache = _none; eng._rtp_heal = _none; eng._rtp_withdraw = _none
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: type("K", (), {"last_price": 2.05})())

    async def go():
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as db:
            o = models.Order(pair="AAAUSDT", direction="LONG", status="OPEN", entry_price=2.0, quantity=100.0, investment=40.0, leverage=5.0,
                             notional_value=200.0, confidence="STRONG_BUY", is_paper=True, entry_strategy="MANUAL", manual_exit_mode="NOSTOP",
                             manual_emergency_stop_pct=15.0, entry_fee=0.09, closing_in_progress=False)
            db.add(o); await db.commit()
            await eng.set_exit_override(db, o.id, sl_pct=None, tp_pct=5.0)
            a = (o.manual_exit_mode, o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct, o.manual_emergency_stop_pct, o.exit_override_at)
            await eng.set_exit_override(db, o.id, sl_pct=-2.0, tp_pct=None)                                # a stop typed → FIXED
            b = (o.manual_exit_mode, o.pattern_fixed_sl_pct, o.manual_emergency_stop_pct)
        await e.dispose()
        return a, b
    a, b = asyncio.run(go())
    assert a == ("NOSTOP", None, 5.0, 15.0, None) and b == ("FIXED", -2.0, None)


# ── ⛔ block bot trades ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _block_env(monkeypatch, tmp_path, rows):
    import services.trading_engine as T, models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    e = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bl.db'}")
    S = async_sessionmaker(e, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True; eng.is_running = True
    blocks = []
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append((name, d))

    async def setup():
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as db:
            for r in rows:
                db.add(models.Order(**dict(dict(direction="LONG", status="OPEN", entry_price=1.0, quantity=10.0, investment=1.0, leverage=10.0,
                                                notional_value=10.0, confidence="STRONG_BUY", is_paper=True, entry_strategy="MANUAL"), **r)))
            await db.commit()
    return T, eng, S, e, blocks, setup


def test_bot_open_refused_while_a_manual_row_blocks_and_lifts_on_close(monkeypatch, tmp_path):
    import models
    from sqlalchemy import update
    T, eng, S, e, blocks, setup = _block_env(monkeypatch, tmp_path, [dict(pair="IOSTUSDT", manual_block_bot=True)])

    async def go():
        await setup()
        async with S() as db:
            r1 = await eng.open_position(db, "BTCUSDT", "LONG", "STRONG_BUY", 100.0)
            r1f = await eng.open_position(db, "XUSDT", "LONG", "STRONG_BUY", 1.0, frenzy_long=True)
        st = await eng.manual_block_active()
        async with S() as db:
            await db.execute(update(models.Order).values(status="CLOSED")); await db.commit()   # closed by any path → lifted
        st2 = await eng.manual_block_active()
        await e.dispose()
        return r1, r1f, st, st2
    r1, r1f, st, st2 = asyncio.run(go())
    assert r1 is None and r1f is None and blocks == [("MANUAL_BLOCK", "LONG"), ("MANUAL_BLOCK", "LONG")]
    assert "blocked by the manual IOSTUSDT position" in eng._frenzy_open_refusal
    assert st["pair"] == "IOSTUSDT" and st2 is None


def test_block_fails_closed_on_an_unreadable_flag(monkeypatch, tmp_path):
    T, eng, S, e, blocks, setup = _block_env(monkeypatch, tmp_path, [])
    class _Broken:
        def __call__(self):
            raise RuntimeError("db gone")
    monkeypatch.setattr(T, "AsyncSessionLocal", _Broken())
    st = asyncio.run(eng.manual_block_active())
    assert st["pair"] == "?" and "db gone" in st["error"]
    assert asyncio.run(eng.open_position(None, "BTCUSDT", "SHORT", "STRONG_BUY", 100.0)) is None and blocks == [("MANUAL_BLOCK_READ_FAILED", "SHORT")]
    assert eng._manual_block_read_failed_at is not None                                                # /api/status surfaces it


def test_import_and_manual_opens_are_never_blocked(monkeypatch, tmp_path):
    """The real import path books while another OPEN row blocks bot trades (it never consults the block)."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, rows=[dict(pair="IOSTUSDT", entry_strategy="MANUAL", manual_block_bot=True)])
    async def _boom():
        raise AssertionError("the import must never read the bot block")
    eng.manual_block_active = _boom
    o = asyncio.run(run(_imp("FIXED", sl=1.5, tp=3.0, block_bot=True)))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert o.manual_block_bot is True


def test_lift_endpoint_and_performance_block_state(monkeypatch, tmp_path):
    import main
    from fastapi import HTTPException
    T, eng, S, e, blocks, setup = _block_env(monkeypatch, tmp_path, [dict(pair="IOSTUSDT", manual_block_bot=True),
                                                                      dict(pair="AAAUSDT", status="CLOSED", manual_block_bot=True)])
    monkeypatch.setattr(main, "trading_engine", eng)

    async def go():
        await setup()
        async with S() as db:
            st = await eng.manual_block_state(db)
            ok = await main.manual_lift_block(main.LiftBlockRequest(order_id=1), db=db)
            st2 = await eng.manual_block_state(db)
            errs = []
            for oid in (1, 2, 99):
                try:
                    await main.manual_lift_block(main.LiftBlockRequest(order_id=oid), db=db)
                except HTTPException as ex:
                    errs.append((ex.status_code, ex.detail))
        after = await eng.manual_block_active()
        await e.dispose()
        return st, ok, st2, errs, after
    st, ok, st2, errs, after = asyncio.run(go())
    assert [x["pair"] for x in st] == ["IOSTUSDT"] and ok == {"ok": True, "order_id": 1, "pair": "IOSTUSDT", "manual_block_bot": False}
    assert st2 == [] and after is None
    assert errs == [(400, "IOSTUSDT does not block bot trades"), (400, "that position is closed — its block already lifted"), (400, "no such position")]
    assert '@app.post("/api/manual/lift_block")' in MAIN and '@app.get("/api/manual/import_orders")' in MAIN
    perf = MAIN[MAIN.index('@app.get("/api/performance")'):MAIN.index("def _compute_entry_type_stats(")]
    assert 'out["manual_block"] = await trading_engine.manual_block_state(db)' in perf and 'out["manual_block"] = None' in perf


# ── schema / config / UI / exports ───────────────────────────────────────────────────────────────────────────────────────────────
def test_schema_config_and_payloads():
    import models, config
    cols = [c.name for c in models.Order.__table__.columns]
    assert "manual_emergency_stop_pct" in cols and "manual_block_bot" in cols                       # D12: the orders CSV dumps every column
    db_src = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "(('manual_emergency_stop_pct', 'FLOAT'), ('manual_block_bot', 'BOOLEAN'))" in db_src
    assert config.ThresholdConfig().manual_emergency_stop_pct == 15.0 if hasattr(config, "ThresholdConfig") else True
    js = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
    th = js.get("thresholds", js)
    assert th.get("manual_emergency_stop_pct") == 15.0
    assert MAIN.count('"manual_emergency_stop_pct": getattr(o, \'manual_emergency_stop_pct\', None), "manual_block_bot": bool(') == 2   # open + closed payloads


def test_ui_ids_load_save_and_both_exports():
    for i in ("manual-emergency-wrap", "manual-emergency-on", "manual-emergency-pct", "manual-block-bot", "import-emergency-wrap", "import-emergency-on",
              "import-emergency-pct", "import-block-bot", "import-app-orders", "import-keep-tp", "import-keep-tp-wrap", "manual-block-banner",
              "config-manual-emergency-stop-pct"):
        assert UI.count(f'id="{i}"') == 1, i
    sel = lambda i: re.search(rf'<select id="{i}"[^>]*>(.*?)</select>', UI, re.S).group(1)
    for i in ("manual-exit-mode", "import-exit-mode"):
        assert '<option value="NOSTOP">🚫 No stop (+ opt. TP)</option>' in sel(i)
    assert "document.getElementById('config-manual-emergency-stop-pct'); if(_e && config.thresholds.manual_emergency_stop_pct !== undefined)" in UI   # load
    assert "manual_emergency_stop_pct: (() => { const _v = document.getElementById('config-manual-emergency-stop-pct')?.value;" in UI           # save
    assert UI.count("lines.push(...manualBlockReportLines(perf, hr2));") == 2 and UI.count("function manualBlockReportLines(perf, hr2)") == 1   # D12
    assert "`${API_BASE}/api/manual/lift_block`" in UI and "/api/manual/import_orders?pair=" in UI and "renderManualBlockBanner(data || [])" in UI
    assert "Refused if it has its own TP / SL orders on Binance" not in UI and "cancel them in the app first.`" not in UI
    assert "TP on Binance${p.tp_maybe_leftover ? ' (could be a bot leftover)' : ''} found: <b>" in UI and "Keep it (the bot takes it over)" in UI and "Bot trades blocked by <b>" in UI and ">Lift block</button>" in UI
    sub = UI[UI.index("async function submitImportPosition()"):UI.index("// ➕ (DECISION_LOG 324) ADD TO POSITION — the ➕ button")]
    assert "keep_app_tp: importKeepTpEffective()" in sub and "block_bot: !!g('import-block-bot')?.checked" in sub and "body.emergency_stop_pct = _em" in sub
    mo = UI[UI.index("async function openManualPosition()"):]
    mo = mo[:mo.index("JSON.stringify(body)")]
    assert "if (g('manual-block-bot')?.checked) body.block_bot = true;" in mo and "body.emergency_stop = _em != null; body.emergency_stop_pct = _em;" in mo


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_risk_lines_and_explain_in_node():
    a = UI.index("// ⇄ SLTP-UNITS BEGIN"); b = UI.index("// 🖐 manual entry: SL / TP typed in % or $")
    c = UI.index("// 📥 IMPORT FROM BINANCE (DECISION_LOG 328"); d = UI.index("function openImportModal(")
    x0 = UI.index("        const MANUAL_EXIT_FIELD_WHY = {"); x1 = UI.index("        // paints the line")
    prog = """
const els = {};
function mk(id, v) { els[id] = { id, value: v || '', disabled: false, checked: false, dataset: {}, validity: { badInput: false }, textContent: '', innerHTML: '',
  classList: { _s: new Set(), toggle(k, on) { on ? this._s.add(k) : this._s.delete(k); }, contains(k) { return this._s.has(k); }, add(k) { this._s.add(k); }, remove(k) { this._s.delete(k); } } }; return els[id]; }
global.document = { getElementById: id => els[id] || null, addEventListener() {} };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.window = { _cachedConfig: { thresholds: { manual_floor_sl_pct: -3, manual_emergency_stop_pct: 15 }, investment: { max_holding_time_minutes: 1200 } }, _lastTotalPortfolio: 3000 };
global.showToast = () => {}; global._setHTMLIfChanged = () => {}; global.formatTime = x => x; global.API_BASE = '';
const MANUAL_MAINT_MARGIN_PCT_UI = 0.5, MANUAL_BACKSTOP_LIQ_SHARE_UI = 0.92, MANUAL_BACKSTOP_MARGIN_PCT_UI = 0.5, MANUAL_BACKSTOP_MIN_GAP_PCT_UI = 0.3;
global.manualStopForLeverage = (lev, f) => ({ stop: Math.max(f, -(100 / lev - 0.5) * 0.8) });
['import-risk-line', 'import-what-line', 'import-sl', 'import-tp', 'import-sl-conv', 'import-tp-conv', 'import-emergency-pct', 'import-block-bot'].forEach(i => mk(i));
mk('import-exit-mode', 'NOSTOP'); mk('import-emergency-on').checked = true; mk('import-keep-tp').checked = true; els['import-emergency-pct'].value = '15';
""" + UI[a:b] + UI[x0:x1] + UI[c:d] + """
_importCands = [{ pair: 'IOSTUSDT', side: 'LONG', qty: 2400000, entry_price: 0.0045, leverage: 10, importable: true }];
_importModalPair = 'IOSTUSDT';
_importPreview = { liquidation_price: 0.0031, tp_count: 1, sl_count: 0, tp_price: 0.0052, tp_pct: 15.2 };
importRiskLine(); const em = els['import-risk-line'].textContent, what = els['import-what-line'].textContent;
els['import-emergency-on'].checked = false; importRiskLine(); const none = els['import-risk-line'].textContent;
els['import-emergency-on'].checked = true; els['import-emergency-pct'].value = '40'; importRiskLine(); const beyond = els['import-risk-line'].textContent;
const ex1 = manualExitExplainText('NOSTOP', { cfg: window._cachedConfig, size: 1080, lev: 10, dir: 'LONG', unit: 'pct', sl: null, tp: 3, emergency: 15 }).text;
const ex2 = manualExitExplainText('NOSTOP', { cfg: window._cachedConfig, size: 1080, lev: 10, dir: 'LONG', unit: 'pct', sl: null, tp: null, emergency: null }).text;
const nr = noStopRiskText(10800, 15, 'x').text;
console.log(JSON.stringify({ em, what, none, beyond, ex1, ex2, nr }));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["nr"].startswith("Position $10,800 · at −15 % emergency ≈ −$1,620") and "(54 % of the account)" in r["nr"]
    assert "at −15 % emergency ≈ −$1,620" in r["em"] and "refused" not in r["em"]
    assert r["none"].startswith("Position $10,800 · no stop — loss unlimited until liquidation at 0.00310000")
    assert "is at / beyond the liquidation price" in r["beyond"] and "the import will be refused" in r["beyond"]
    assert "ONE emergency stop is placed on Binance at −15 % of price" in r["what"] and "own take profit at the app's price 0.00520000" in r["what"]
    assert "ONE emergency stop on Binance at −15.00 % of price" in r["ex1"] and "no max hold" in r["ex1"] and "max hold 20" not in r["ex1"]
    assert "No stop at all" in r["ex2"] and "loss unlimited until liquidation" in r["ex2"]


# ── (343) Banner fix: size / P&L at the mark (Binance's convention), no ticking age ──────────────────────────────────────────────
def test_candidates_value_the_position_at_the_mark_with_binance_roi(monkeypatch):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    monkeypatch.setattr(T.config.trading_config.thresholds, "broker_backstop_enabled", True, raising=False)
    eng._untracked_positions = {"IOSTUSDT": {"pair": "IOSTUSDT", "side": "LONG", "qty": 7458347.0, "entry_price": 0.0014734, "reason": "NO_ROW"}}
    p = _pos(symbol="IOST/USDT:USDT", contracts=7458347.0, entry=0.0014734, lev=10, mark=0.00145031, upnl=-172.51)
    p["margin"] = 1081.685
    eng._untracked_snapshot = (dt.datetime.utcnow() - dt.timedelta(seconds=15), {"IOSTUSDT": p})
    c = eng.import_candidates()[0]
    assert c["notional"] == round(7458347.0 * 0.00145031, 2) and c["notional_basis"] == "mark" and c["investment"] == 1081.68
    assert c["roi_pct"] == round(-172.51 / 1081.685 * 100, 2) and c["snapshot_stale"] is False
    p2 = dict(p, mark_price=0.0, margin=0.0)                                                   # no mark → entry, labelled
    eng._untracked_snapshot = (dt.datetime.utcnow() - dt.timedelta(seconds=150), {"IOSTUSDT": p2})
    c = eng.import_candidates()[0]
    assert c["notional_basis"] == "entry" and c["notional"] == round(7458347.0 * 0.0014734, 2) and c["investment"] == round(c["notional"] / 10, 2)
    assert c["snapshot_stale"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_banner_text_at_mark_and_stale_warning_only():
    a = UI.index("function importSizeText(c)"); b = UI.index("function renderImportBanner()")
    prog = """
var _impUsd = v => '$' + Number(v).toLocaleString('en-US', { maximumFractionDigits: Math.abs(Number(v)) >= 100 ? 0 : 2 });
""" + UI[a:b] + """
const c = { notional: 10816.85, notional_basis: 'mark', unrealized_pnl: -172.51, roi_pct: -15.94, snapshot_age_s: 15, snapshot_stale: false };
console.log(JSON.stringify({ size: importSizeText(c), fresh: importStaleNote(c),
  stale: importStaleNote(Object.assign({}, c, { snapshot_age_s: 130, snapshot_stale: true })),
  entry: importSizeText({ notional: 10989, notional_basis: 'entry', unrealized_pnl: null }) }));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["size"] == "$10,817 at mark · P&L −$172.51 (−15.9 % ROI)" and r["fresh"] == "" and r["stale"] == " · ⚠ figures 2 min old"
    assert r["entry"] == "$10,989 at entry"
    js = UI[UI.index("function renderImportBanner()"):UI.index("function importBadgeHtml(o)")]
    assert "snapshot_age_s" not in js and "importStaleNote(c)" in js and "importSizeText(c)" in js
    head = UI[UI.index("function openImportModal("):UI.index("function closeImportModal()")]
    assert "s old — re-read" not in head and "(re-read from Binance at the import)' + importStaleNote(c)" in head and "% ROI" in head


# ── review fixes (DL 343 deep + caveman reviews) ─────────────────────────────────────────────────────────────────────────────────
def test_paper_row_never_blocks_live_and_vice_versa(monkeypatch, tmp_path):
    T, eng, S, e, blocks, setup = _block_env(monkeypatch, tmp_path, [dict(pair="IOSTUSDT", manual_block_bot=True, is_paper=True)])
    async def go():
        await setup()
        eng.is_paper_mode = False
        live = await eng.manual_block_active()
        eng.is_paper_mode = True
        paper = await eng.manual_block_active()
        await e.dispose()
        return live, paper
    live, paper = asyncio.run(go())
    assert live is None and paper["pair"] == "IOSTUSDT"
    import services.trading_engine as T
    assert "Order.is_paper == bool(self.is_paper_mode)" in inspect.getsource(T.TradingEngine.manual_block_active)
    assert "Order.is_paper == bool(self.is_paper_mode)" in inspect.getsource(T.TradingEngine.manual_block_state)


def test_race_guard_1_after_a_4130_wipe_keeps_the_stop(monkeypatch, tmp_path):
    """(review #1) the app's algo TP / SL were wiped by our placement (-4130) and the position then changed (still open) / the re-read
    failed → our stop is KEPT (never withdrawn into a naked position), RuntimeError naming it and what rests."""
    for again, txt in (([_pos(contracts=900.0)], "the position changed on Binance (now LONG 900 @ 2)"), ("FAIL", "Binance could not be re-read")):
        run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP, APP_SL], [OUR_STOP]], positions_seq=([_pos()], again))
        r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
        assert r.startswith("RuntimeError") and txt in r and "The app's TP/SL were removed" in r and "algoId=algo-1 at 1.93 was KEPT" in r, r
        assert "On Binance now: STOP_MARKET SELL @ 1.95 (close-all, algo #algo-1)" in r
        assert ("cancel_backstop", "algo-1") not in log and _nothing_written(read)
    # the app's orders still rest (no wipe) → the DL 328 behaviour: withdraw ours, ValueError
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [APP_TP, OUR_STOP]], positions_seq=([_pos()], [_pos(contracts=900.0)]))
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("ValueError") and ("cancel_backstop", "algo-1") in log
    # confirmed flat → withdrawn too
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP], [OUR_STOP]], positions_seq=([_pos()], []))
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("ValueError") and "(now closed)" in r and ("cancel_backstop", "algo-1") in log


def test_4130_wipe_with_failed_placement_retries_then_says_no_stop(monkeypatch, tmp_path):
    """(review #2) the -4130 clear removed the app's TP / SL and our placement failed: one retry; still failing → "⚠ NO STOP on Binance"."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, bk_id=None, open_orders_seq=[[APP_TP, APP_SL], []])
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("RuntimeError: ⚠ NO STOP on Binance — the app's TP/SL were removed") and "set a stop in the app NOW" in r
    assert [x[0] for x in log].count("backstop") == 2 and _nothing_written(read)
    # the retry lands → the import goes on
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP, APP_SL], [], [OUR_STOP]])
    seq = iter([None, "algo-1"])
    async def _place(pair, d, trig):
        log.append(("backstop", pair, d, round(trig, 8))); return next(seq)
    monkeypatch.setattr(T.binance_service, "place_backstop_stop", _place)
    o = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert o.backstop_algo_id == "algo-1" and [x[0] for x in log].count("backstop") == 2


def test_unseen_app_order_refuses_and_preview_failure_blocks_submit(monkeypatch, tmp_path):
    """(review #3) the click carries what the pop-up showed: a TP / SL placed after it opened → refused, nothing changed."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[APP_TP, APP_SL])
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True, app_orders_seen=["algo:tp-9"])))
    assert "appeared after the pop-up read Binance (STOP_MARKET SELL @ 1.8" in r and _kinds(log) == ["positions", "open_orders"]
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[APP_TP, APP_SL], [OUR_STOP]])
    assert not isinstance(asyncio.run(run(_imp("FLOOR", keep_app_tp=True, app_orders_seen=["algo:tp-9", "algo:sl-9"]))), str)
    sub = UI[UI.index("async function submitImportPosition()"):UI.index("// ➕ (DECISION_LOG 324) ADD TO POSITION — the ➕ button")]
    assert "if (_importPreview.error) {" in sub and "label: 'Retry', onClick: () => loadImportPreview(c.pair, c.side)" in sub
    assert sub.index("if (_importPreview.error) {") < sub.index("JSON.stringify(body)") and "body.app_orders_seen = " in sub
    assert '"app_orders_seen": ' not in MAIN and "app_orders_seen=request.app_orders_seen" in MAIN


def test_bot_leftover_limit_is_labelled_and_never_kept(monkeypatch, tmp_path):
    """(review #7) a reduce-only limit the bot's own close left (its id on a closed row / the pair flagged dirty) is a LEFTOVER: cancelled at
    the import, never offered as "your TP"."""
    lim = _reg("old-77", "LIMIT", "SELL", 2.3, 1000.0)
    c = classify_app_orders([lim, APP_TP], "LONG", 2.0, 1000.0, bot_leftover_ids={"old-77"})
    assert [o["id"] for o in c["leftover"]] == ["old-77"] and c["tp_price"] == 2.2 and c["refusal"] is None
    # (round 2 #1) a dirty pair with an UNKNOWN id: still a TP (keepable), flagged maybe_leftover — only an id match is a leftover
    sus = classify_app_orders([lim], "LONG", 2.0, 1000.0, leftover_suspect=True)
    assert sus["leftover"] == [] and sus["tp_price"] == 2.3 and sus["tp"][0]["maybe_leftover"] is True
    assert classify_app_orders([APP_TP], "LONG", 2.0, 1000.0, leftover_suspect=True)["tp"][0]["maybe_leftover"] is False   # algo = the app's
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[lim, APP_TP], [OUR_STOP]],
                                          rows=[dict(pair="FOOUSDT", status="CLOSED", protective_tp_order_id="old-77")])
    o = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert ("cancel_regular", "old-77") in log and o.pattern_fixed_tp_pct == import_tp_pct_from_price("LONG", 2.0, 2.2, TF * 100, eng._rtp_maker())
    assert "Leftover from the bot: " in UI and "never kept" in UI and "(could be a bot leftover)" in UI
    # dirty pair + unknown-id limit: kept as the TP; once step 3 confirms only our stop rests, the pair is no longer dirty
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders_seq=[[_reg("new-5", "LIMIT", "SELL", 2.3, 1000.0)], [OUR_STOP]])
    eng._rtp_dirty_pairs = {"FOOUSDT"}
    o = asyncio.run(run(_imp("FLOOR", keep_app_tp=True, app_orders_seen=["regular:new-5"])))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert o.pattern_fixed_tp_pct == import_tp_pct_from_price("LONG", 2.0, 2.3, TF * 100, eng._rtp_maker()) and "FOOUSDT" not in eng._rtp_dirty_pairs


def test_old_page_without_app_orders_seen_is_refused(monkeypatch, tmp_path):
    """(round 2 #3) a cached page that does not send app_orders_seen → refused (reload), never a skipped check."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[APP_TP])
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True, app_orders_seen=None)))
    assert "reload the page (Cmd+Shift+R) and try again" in r and _kinds(log) == ["positions", "open_orders"] and _nothing_written(read)


def test_landed_stop_after_a_wipe_is_kept_not_cancelled(monkeypatch, tmp_path):
    """(round 2 #2) the call 'failed' but our stop landed and the app's TP/SL are gone → kept as the only protection (never "cancel it")."""
    landed = _algo("999", "STOP_MARKET", "SELL", 1.93)
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, bk_id=None, open_orders_seq=[[APP_TP, APP_SL], [landed]])
    r = asyncio.run(run(_imp("FLOOR", keep_app_tp=True)))
    assert r.startswith("RuntimeError") and "KEPT as the position's only protection" in r and "replaces it" in r and "cancel it before retrying" not in r


def test_manual_nostop_open_places_the_emergency_first_then_clamps():
    """(review #6) the 🖐 NOSTOP open: the emergency stop is placed at the REQUESTED % first (no window without it), the liquidation read
    and the move inside it come after."""
    import services.trading_engine as T
    src = inspect.getsource(T.TradingEngine.open_manual_position)
    i_place = src.index("_bk_algo_id = await binance_service.place_backstop_stop(pair, direction, _bk_trigger)")
    i_liq = src.index("_lp = await asyncio.wait_for(binance_service.get_position_for_symbol(symbol), 4.0)")
    i_move = src.index("await self._backstop_replace_one(pair, direction, _bk_algo_id, _clamped, _ctrig)")
    assert i_place < i_liq < i_move
