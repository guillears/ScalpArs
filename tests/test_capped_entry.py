"""📖🧢 (DECISION_LOG 553) capped entry — the ceiling price, the fee split, the exchange order life-cycle (fake exchange), wiring."""
import asyncio
import json
import pathlib

import pytest

from services import entry_depth as D

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cap_price_rounds_on_the_safe_side():
    assert D.cap_price(100.0, "LONG", 0.15, 0.01) == pytest.approx(100.15)
    assert D.cap_price(0.1263, "LONG", 0.15, 0.0001) == pytest.approx(0.1264)          # 0.12648… rounds DOWN, never above the ceiling
    assert D.cap_price(0.1263, "SHORT", 0.15, 0.0001) == pytest.approx(0.1262)         # 0.12611… rounds UP, never below
    assert D.cap_price(0.01875, "LONG", 0.15, 0.00001) == pytest.approx(0.01877)
    assert D.cap_price(1.0, "LONG", 0.15, 0.01) == pytest.approx(1.0)                  # coarse tick: never below the best ask
    assert D.cap_price(1.0, "SHORT", 0.15, 0.01) == pytest.approx(1.0)
    assert D.cap_price(0, "LONG", 0.15, 0.01) is None and D.cap_price("x", "LONG", 0.15, 0.01) is None


def test_capped_fee_split():
    assert D.capped_fill_fee(100, 100, 2.0, 0.0005, 0.0002) == pytest.approx(0.1)
    assert D.capped_fill_fee(60, 100, 2.0, 0.0005, 0.0002) == pytest.approx(60 * 2 * 0.0005 + 40 * 2 * 0.0002)
    assert D.capped_fill_fee(150, 100, 2.0, 0.0005, 0.0002) == pytest.approx(0.1)      # at-once never above the total


class _Ex:
    """fake ccxt exchange: create_order ACK (or raise), then fetch_order returns the scripted states in turn (the last repeats); cancel /
    cancel-all recorded; cancel_all can turn the order terminal."""
    def __init__(self, ack, states, cancel_raises=False, fetch_fails=False, send_raises=False, cancel_all_ends=None):
        self.ack, self.states, self.cancel_raises, self.fetch_fails = ack, list(states), cancel_raises, fetch_fails
        self.send_raises, self.cancel_all_ends = send_raises, cancel_all_ends
        self.sent, self.cancelled, self.cancel_all = [], [], []

    def amount_to_precision(self, s, a):
        return str(round(a, 0))

    def price_to_precision(self, s, p):
        return str(p)

    def market(self, s):
        return {'limits': {'amount': {'min': 1}, 'cost': {'min': 5}}}

    async def create_order(self, **kw):
        self.sent.append(kw)
        if self.send_raises:
            import ccxt.async_support as _c
            raise (self.send_raises if isinstance(self.send_raises, Exception) else _c.RequestTimeout("timeout"))
        return dict(self.ack)

    async def fetch_order(self, oid, sym, params=None):
        if self.fetch_fails:
            raise (self.fetch_fails if isinstance(self.fetch_fails, Exception) else RuntimeError("down"))
        if self.cancel_all and self.cancel_all_ends:
            return dict(self.cancel_all_ends)
        return dict(self.states.pop(0) if len(self.states) > 1 else self.states[0])

    async def cancel_order(self, oid, sym, params=None):
        self.cancelled.append(oid or (params or {}).get('origClientOrderId'))
        if self.cancel_raises:
            raise RuntimeError("Unknown order sent")

    async def cancel_all_orders(self, sym):
        self.cancel_all.append(sym)


def _bs(monkeypatch, ex, pos=None):
    from services import binance_service as B
    bs = B.binance_service

    async def _none(*a, **k):
        return None

    async def _pos(sym):
        if isinstance(pos, Exception):
            raise pos
        return pos
    monkeypatch.setattr(bs, "exchange", ex, raising=False)
    monkeypatch.setattr(bs, "_check_ban", _none)
    monkeypatch.setattr(bs, "load_markets", _none)
    monkeypatch.setattr(bs, "get_position_for_symbol", _pos)
    return bs


def _run(monkeypatch, ex, amount=1000, px=1.0, wait=0.05, pos=None):
    bs = _bs(monkeypatch, ex, pos)
    st = {}
    r = asyncio.run(bs.create_capped_entry("X/USDT:USDT", "buy", amount, px, wait_s=wait, poll_s=0.01, status=st))
    return r, st


def test_full_fill_at_once(monkeypatch):
    ex = _Ex({'id': '1', 'status': 'closed', 'filled': 1000, 'average': 0.999}, [{'id': '1', 'status': 'closed', 'filled': 1000, 'average': 0.999}])
    r, st = _run(monkeypatch, ex)
    assert st['sent'] is True and ex.sent[0]['type'] == 'limit' and ex.sent[0]['price'] == 1.0
    assert ex.sent[0]['params']['timeInForce'] == 'GTC' and ex.sent[0]['params']['newClientOrderId'].startswith('cap')
    assert r['amount'] == 1000 and r['price'] == 0.999 and r['immediate'] == 1000 and ex.cancelled == [] and r['unconfirmed'] is False


def test_partial_then_rest_fills_while_waiting(monkeypatch):
    ex = _Ex({'id': '2', 'status': 'open', 'filled': 600, 'average': 0.998},
             [{'id': '2', 'status': 'open', 'filled': 600, 'average': 0.998}, {'id': '2', 'status': 'closed', 'filled': 1000, 'average': 0.999}])
    r, _ = _run(monkeypatch, ex, wait=5)
    assert r['amount'] == 1000 and r['immediate'] == 600 and ex.cancelled == []


def test_timeout_cancels_and_keeps_the_partial(monkeypatch):
    ex = _Ex({'id': '3', 'status': 'open', 'filled': 400, 'average': 0.997},
             [{'id': '3', 'status': 'open', 'filled': 400, 'average': 0.997}, {'id': '3', 'status': 'open', 'filled': 400, 'average': 0.997},
              {'id': '3', 'status': 'canceled', 'filled': 450, 'average': 0.9975}])
    r, _ = _run(monkeypatch, ex, wait=0.03)
    assert ex.cancelled[:1] == ['3'] and r['amount'] == 450 and r['price'] == 0.9975 and r['immediate'] == 400 and not r['unconfirmed']


def test_nothing_filled_is_zero(monkeypatch):
    ex = _Ex({'id': '4', 'status': 'open', 'filled': 0, 'average': None},
             [{'id': '4', 'status': 'open', 'filled': 0}, {'id': '4', 'status': 'canceled', 'filled': 0}])
    r, _ = _run(monkeypatch, ex, wait=0.03)
    assert r['amount'] == 0 and ex.cancelled[:1] == ['4'] and not r['unconfirmed']


def test_cancel_keeps_failing_order_still_open(monkeypatch):
    """(B2) the cancel is re-sent on every non-terminal read, then cancel-all; still open → unconfirmed + the live position size."""
    ex = _Ex({'id': '5', 'status': 'open', 'filled': 300, 'average': 0.99}, [{'id': '5', 'status': 'open', 'filled': 300, 'average': 0.99}],
             cancel_raises=True)
    r, _ = _run(monkeypatch, ex, wait=0.03, pos={'contracts': 700, 'entry_price': 0.995})
    assert len(ex.cancelled) == 5 and ex.cancel_all == ["X/USDT:USDT"]
    assert r['unconfirmed'] is True and r['amount'] == 700 and r['price'] == 0.995
    ex2 = _Ex({'id': '6', 'status': 'open', 'filled': 300, 'average': 0.99}, [{'id': '6', 'status': 'open', 'filled': 300, 'average': 0.99}],
              cancel_raises=True, cancel_all_ends={'id': '6', 'status': 'canceled', 'filled': 320, 'average': 0.991})
    r2, _ = _run(monkeypatch, ex2, wait=0.03, pos=RuntimeError("never read"))
    assert r2['unconfirmed'] is False and r2['amount'] == 320                 # cancel-all ended it: confirmed
    ex3 = _Ex({'id': '7', 'status': 'open', 'filled': 300, 'average': 0.99}, [{}], fetch_fails=True)
    r3, _ = _run(monkeypatch, ex3, wait=0.03, pos=RuntimeError("x"))
    assert r3['unconfirmed'] is True and r3['amount'] == 300                  # last resort: the at-once fill


def test_send_raised_is_recovered_by_client_id(monkeypatch):
    """(M1) the request left but the reply was lost: the order is found by its client id and settled."""
    ex = _Ex({}, [{'id': '9', 'status': 'canceled', 'filled': 250, 'average': 0.996}], send_raises=True)
    r, st = _run(monkeypatch, ex, wait=0.03)
    assert ex.cancelled and ex.cancelled[0].startswith('cap') and r['amount'] == 250 and r['price'] == 0.996
    ex0 = _Ex({}, [{'id': '9', 'status': 'canceled', 'filled': 0}], send_raises=True)
    r0, st0 = _run(monkeypatch, ex0, wait=0.03)
    assert r0 is None and st0['sent'] is True


def test_rejection_and_never_placed_are_not_alarms(monkeypatch):
    """(re-review N1) a definite rejection → None at once (no settle, no cancel-all); a lost reply whose client id is NOT found twice →
    never placed: 0 filled, confirmed (no untracked mark)."""
    import ccxt.async_support as _c
    ex = _Ex({}, [{}], send_raises=_c.InsufficientFunds("-2019 margin"))
    r, st = _run(monkeypatch, ex, wait=0.03)
    assert r is None and ex.cancelled == [] and ex.cancel_all == []
    ex2 = _Ex({}, [{}], send_raises=True, fetch_fails=_c.OrderNotFound("-2013"))
    r2, _ = _run(monkeypatch, ex2, wait=0.03)
    assert r2 is None and ex2.cancel_all == []


def test_task_cancelled_mid_wait_settles_and_reports_the_fill(monkeypatch):
    """(B1) a deploy restart cancels the open task mid-wait: the order is cancelled under shield and status carries the fill for the flatten."""
    ex = _Ex({'id': '8', 'status': 'open', 'filled': 400, 'average': 0.997},
             [{'id': '8', 'status': 'open', 'filled': 400, 'average': 0.997}] * 3 + [{'id': '8', 'status': 'canceled', 'filled': 400, 'average': 0.997}])
    bs = _bs(monkeypatch, ex)
    st = {}

    async def go():
        t = asyncio.create_task(bs.create_capped_entry("X/USDT:USDT", "buy", 1000, 1.0, wait_s=30, poll_s=0.01, status=st))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.sleep(1.5)                                                # the shielded settle finishes
    asyncio.run(go())
    assert ex.cancelled and ex.cancelled[0] == '8' and st['filled'] == 400 and st['average'] == 0.997


def test_below_minimum_never_sent(monkeypatch):
    ex = _Ex({'id': '7'}, [{}])
    r, st = _run(monkeypatch, ex, amount=3, px=1.0)
    assert r is None and st == {} and ex.sent == []


def test_engine_wiring():
    src = (ROOT / "services/trading_engine.py").read_text()
    i_lev = src.index("_cap_lv = await asyncio.wait_for(binance_service.set_leverage(_cap_sym, int(leverage)), 5.0)")
    i_book = src.index("_sg_ob = await asyncio.wait_for(binance_service.fetch_orderbook(")
    assert i_lev < i_book                                                         # leverage set BEFORE the book read
    i_cap = src.index("result = await binance_service.create_capped_entry(symbol, side, quantity, _cap_px")
    i_mkt = src.index("result = await binance_service.create_market_order(\n                    symbol=symbol, side=side, amount=quantity, leverage=int(leverage)")
    assert i_cap < i_mkt
    assert 'self._record_filter_block("DEPTH_CAP_NOFILL", direction)' in src
    assert "_cap_on = bool(not self.is_paper_mode and _dg_on" in src            # live only
    assert "entry_cap_immediate_frac=_cap_stamp.get('entry_cap_immediate_frac')" in src
    i_try = src.index("result = await binance_service.create_capped_entry(")
    blk = src[i_try:i_try + 900]
    assert "except BaseException:" in blk and "self._note_live_fill(pair, symbol, direction, float(_cap_st['filled'])" in blk and "raise" in blk
    assert 'self._mark_untracked(pair, direction, 0.0, _cap_px, "CAP_ORDER_UNCONFIRMED", "CAPPED_ENTRY")' in src
    assert 'self._blocked_record(_wh_es, direction, pair, current_price, ["DEPTH_CAP_NOFILL"], _cn_ef)' in src


def test_config_ui_and_stamps():
    import config
    import models
    from services.trading_engine import TradingEngine
    inv = config.InvestmentConfig()
    assert inv.entry_depth_cap_enabled is False and inv.entry_depth_cap_wait_s == 15.0
    j = json.loads((ROOT / "trading_config.json").read_text())["investment"]
    assert j["entry_depth_cap_enabled"] is True and j["entry_depth_cap_wait_s"] == 15.0
    html = (ROOT / "templates/index.html").read_text()
    for el in ("config-depth-cap-enabled", "config-depth-cap-wait"):
        assert html.count(el) == 3
    cols = {c.name for c in models.Order.__table__.columns}
    for c in ("entry_cap_price", "entry_cap_fill_frac", "entry_cap_immediate_frac"):
        assert c in cols and c in TradingEngine.STAMP_SLEEVE_ONLY
