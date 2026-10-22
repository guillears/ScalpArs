"""🏦 (DECISION_LOG 347) Pre-trade deposits fold into the live start balance.

Operator bug: BNB ($104.35) arrived first, 3,100 USDT 9 s later; the baseline was fixed at the first poll ($104.35) and the USDT read as
a deposit ("$104.35 start + $3,100.00 deposits"). Rule: until the BOT's first live trade, every deposit / withdrawal IS the start.
Pinned (in-memory SQLite, stubbed binance_service, no network):
- the exact server state (104.35 + 3,100 stored flows, no orders) → start 3,204.35, flows 0, one write;
- a bot order on file → no fold (deposits stay deposits); 📱 app / 📥 import / RECOVERED rows do NOT end the pre-trade window;
- a failed transfer read → no fold, no write; flows ≈ 0 → no write; a withdrawal folds negatively;
- a still-young row is fenced by the cursor (never lost, never double counted) and folds on a later read.
"""
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

import main
import models
from tests.test_live_balance_equity import _raw, _svc

BNB_PX = 600.0


@pytest.fixture
def live(monkeypatch):
    state = {"bal": None, "rows": [], "fail": False, "commits": 0}

    async def _init(_db):
        return None
    monkeypatch.setattr(main.trading_engine, "initialize", _init)
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False, raising=False)

    async def _gb():
        return state["bal"]

    async def _px():
        return BNB_PX

    async def _none(*a, **k):
        return None

    async def _rows(start_ms, max_age_s=None):
        if state["fail"]:
            raise RuntimeError("transfer read down")
        return [(ts, a) for ts, a in state["rows"] if ts >= start_ms]   # Binance startTime is inclusive

    async def _cap(start_ms):
        return []
    monkeypatch.setattr(main.binance_service, "get_balance", _gb)
    monkeypatch.setattr(main.binance_service, "get_bnb_price", _px)
    monkeypatch.setattr(main.binance_service, "get_spot_balance_usd", _none)
    monkeypatch.setattr(main.binance_service, "get_transfer_rows", _rows, raising=False)
    monkeypatch.setattr(main.binance_service, "get_capital_flows", _cap, raising=False)
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)
    _real_commit = main.locked_commit

    async def _count(db):
        state["commits"] += 1
        await _real_commit(db)
    monkeypatch.setattr(main, "locked_commit", _count)
    main._LIVE_FLOW_MEMO.clear()
    yield state
    main._LIVE_FLOW_MEMO.clear()


def _ms(dt):
    return int((dt - datetime(1970, 1, 1)).total_seconds() * 1000)


async def _state(db):
    return (await db.execute(select(models.BotState))).scalar_one()


async def _seed(db, base, base_at, stored=0.0, cursor=None):
    db.add(models.BotState(live_initial_total_usd=base, live_baseline_at=base_at, live_net_flows_usd=stored, live_flows_cursor_ms=cursor))
    await db.flush()
    await db.commit()


async def _order(db, entry_strategy=None, confidence="HIGH", import_source=None, status="CLOSED"):
    db.add(models.Order(pair="BTCUSDT", direction="LONG", status=status, is_paper=False, entry_price=100.0, current_price=100.0,
                        quantity=1.0, investment=10.0, leverage=10, notional_value=100.0, confidence=confidence,
                        entry_strategy=entry_strategy, import_source=import_source, opened_at=datetime(2026, 10, 22, 12, 0), entry_fee=0.0))
    await db.flush()
    await db.commit()


def _operator_case():
    """the server today: baseline $104.35 (the BNB) at the first poll, USDT +3,100 9 s later — settled & stored long ago."""
    at = datetime.utcnow() - timedelta(hours=2)
    usdt_ts = _ms(at) + 9_000
    return at, usdt_ts


async def _bal_case(live):
    live["bal"] = await _svc(_raw(3100.0, 0.0, 0.0, bnb=104.35 / BNB_PX))       # total = 3,100 USDT + 104.35 BNB-USD


async def test_operator_case_folds_once(db, live):
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    live["rows"] = [(usdt_ts, 3100.0)]
    await _bal_case(live)
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 3204.35 and st.live_net_flows_usd == 0.0
    assert st.live_baseline_at > at and st.live_flows_cursor_ms == usdt_ts       # fenced at the folded row
    assert r["initial_balance"] == 3204.35 and r["net_flows"] == 0.0
    assert r["invested"] == 3204.35 and abs(r["pnl_vs_start"]) < 0.01           # P&L / invested unchanged by the fold
    assert main._LIVE_FLOW_MEMO["key"] == st.live_baseline_at.isoformat() and main._LIVE_FLOW_MEMO["value"] == 0.0
    commits = live["commits"]
    main._LIVE_FLOW_MEMO.clear()                                                # the next fresh read: nothing more to fold
    r2 = await main.get_balance(db)
    assert r2["initial_balance"] == 3204.35 and r2["net_flows"] == 0.0 and live["commits"] == commits


async def test_operator_case_unsettled_rows_fold_too(db, live):
    """same case before the 15-min settle (the rows still pending, cursor None) — both legs fold, nothing double counted."""
    at = datetime.utcnow() - timedelta(minutes=5)
    live["rows"] = [(_ms(at) + 9_000, 3100.0)]
    await _seed(db, 104.35, at)
    await _bal_case(live)
    r = await main.get_balance(db)
    assert r["initial_balance"] == 3204.35 and r["net_flows"] == 0.0
    main._LIVE_FLOW_MEMO.clear()
    r2 = await main.get_balance(db)
    assert r2["initial_balance"] == 3204.35 and r2["net_flows"] == 0.0


async def test_no_fold_after_a_bot_order(db, live):
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    await _order(db)                                                            # the bot's own trade
    live["rows"] = [(usdt_ts, 3100.0)]
    await _bal_case(live)
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 104.35 and st.live_baseline_at == at and st.live_net_flows_usd == 3100.0
    assert r["initial_balance"] == 104.35 and r["net_flows"] == 3100.0 and live["commits"] == 0


@pytest.mark.parametrize("kw", [
    dict(entry_strategy="MANUAL", import_source="BINANCE_APP_HISTORY"),        # 📱 auto-recorded app trade
    dict(entry_strategy="MANUAL", import_source="BINANCE_APP", status="OPEN"),  # 📥 imported app position
    dict(entry_strategy="MANUAL"),                                              # 🖐 pop-up manual
    dict(entry_strategy=None, confidence="RECOVERED", status="OPEN"),           # adopted app position
])
async def test_app_or_manual_trades_do_not_end_the_pretrade_window(db, live, kw):
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    await _order(db, **kw)
    live["rows"] = [(usdt_ts, 3100.0)]
    await _bal_case(live)
    r = await main.get_balance(db)
    assert r["initial_balance"] == 3204.35 and r["net_flows"] == 0.0


async def test_failed_flows_read_never_folds(db, live):
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    live["fail"] = True
    await _bal_case(live)
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 104.35 and st.live_baseline_at == at and st.live_net_flows_usd == 3100.0
    assert r["initial_balance"] is None and r["net_flows"] is None and live["commits"] == 0


async def test_no_write_when_flows_are_zero(db, live):
    at = datetime.utcnow() - timedelta(hours=2)
    await _seed(db, 3204.35, at, stored=0.004)
    live["bal"] = await _svc(_raw(3204.35, 0.0, 0.0))
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 3204.35 and st.live_baseline_at == at and st.live_flows_cursor_ms is None
    assert r["initial_balance"] == 3204.35 and live["commits"] == 0


async def test_withdrawal_before_first_trade_folds_negatively(db, live):
    at = datetime.utcnow() - timedelta(hours=2)
    w_ts = _ms(at) + 60_000
    await _seed(db, 1000.0, at, stored=-300.0, cursor=w_ts)
    live["rows"] = [(w_ts, -300.0)]
    live["bal"] = await _svc(_raw(700.0, 0.0, 0.0))
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 700.0 and st.live_net_flows_usd == 0.0
    assert r["initial_balance"] == 700.0 and r["net_flows"] == 0.0 and r["pnl_vs_start"] == 0.0


async def test_young_row_stays_a_flow_then_folds_once(db, live, monkeypatch):
    at = datetime.utcnow() - timedelta(hours=2)
    old_ts = _ms(at) + 9_000
    young_ts = int(time.time() * 1000) - 5_000                                  # 5 s old: the transfer cache may not hold its neighbours yet
    await _seed(db, 100.0, at)
    live["rows"] = [(old_ts, 3100.0), (young_ts, 50.0)]
    live["bal"] = await _svc(_raw(3250.0, 0.0, 0.0))
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 3200.0 and st.live_flows_cursor_ms == old_ts
    assert r["initial_balance"] == 3200.0 and r["net_flows"] == 50.0           # the young row is still counted — never lost
    monkeypatch.setattr(main, "LIVE_PRETRADE_FOLD_MIN_AGE_MS", 0)               # … it ages
    main._LIVE_FLOW_MEMO.clear()
    r2 = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 3250.0 and st.live_flows_cursor_ms == young_ts
    assert r2["initial_balance"] == 3250.0 and r2["net_flows"] == 0.0 and r2["pnl_vs_start"] == 0.0
    main._LIVE_FLOW_MEMO.clear()
    r3 = await main.get_balance(db)
    assert r3["initial_balance"] == 3250.0 and r3["net_flows"] == 0.0           # never double counted


async def test_bot_order_helper_matches_origin_view(db):
    from services.origin_view import order_origin
    cases = [(None, "HIGH"), ("MOMENTUM", "HIGH"), ("MANUAL", "HIGH"), (" manual ", "HIGH"), ("FRENZY_LONG", "RECOVERED")]
    for es, conf in cases:
        await db.execute(models.Order.__table__.delete())
        db.add(models.Order(pair="X", direction="LONG", status="CLOSED", is_paper=False, entry_price=1.0, quantity=1.0, investment=1.0,
                            leverage=1, notional_value=1.0, confidence=conf, entry_strategy=es, opened_at=datetime(2026, 10, 22)))
        await db.flush()
        assert await main._live_bot_order_exists(db) == (order_origin(es, conf) == "BOT"), (es, conf)
    await db.execute(models.Order.__table__.delete())
    db.add(models.Order(pair="X", direction="LONG", status="CLOSED", is_paper=True, entry_price=1.0, quantity=1.0, investment=1.0,
                        leverage=1, notional_value=1.0, confidence="HIGH", opened_at=datetime(2026, 10, 22)))
    await db.flush()
    assert await main._live_bot_order_exists(db) is False                       # paper rows never count


async def test_unfilled_bot_rows_do_not_end_the_pretrade_window(db, live):
    """a live SIGNAL_EXPIRED / aborted maker row never filled — the bot has not traded yet."""
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    await _order(db, entry_strategy="MOMENTUM", status="SIGNAL_EXPIRED")
    live["rows"] = [(usdt_ts, 3100.0)]
    await _bal_case(live)
    r = await main.get_balance(db)
    assert r["initial_balance"] == 3204.35 and r["net_flows"] == 0.0


async def test_fold_reads_the_cursor_written_in_the_same_session(db, live):
    """_live_net_flows stores with synchronize_session=False in the SAME session; the fold must not use the identity map's stale copy."""
    from sqlalchemy import update
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at)                                                  # loaded: cursor None, stored 0
    st = await _state(db)
    assert st.live_flows_cursor_ms is None
    await db.execute(update(models.BotState).where(models.BotState.id == st.id)
                     .values(live_net_flows_usd=3100.0, live_flows_cursor_ms=usdt_ts).execution_options(synchronize_session=False))
    live["rows"] = [(usdt_ts, 3100.0)]
    b, b_at, fl = await main._live_pretrade_fold(db, 104.35, at, 3100.0)
    assert b == 3204.35 and fl == 0.0 and b_at > at                             # stale copy (cursor None, stored 0) → CAS miss, no fold
    st = await _state(db)
    assert st.live_initial_total_usd == 3204.35 and st.live_flows_cursor_ms == usdt_ts and st.live_net_flows_usd == 0.0


async def test_no_fold_to_an_empty_start(db, live):
    at = datetime.utcnow() - timedelta(hours=2)
    w_ts = _ms(at) + 60_000
    await _seed(db, 500.0, at, stored=-499.5, cursor=w_ts)                      # everything withdrawn before the first trade
    live["rows"] = [(w_ts, -499.5)]
    live["bal"] = await _svc(_raw(0.5, 0.0, 0.0))
    r = await main.get_balance(db)
    st = await _state(db)
    assert st.live_initial_total_usd == 500.0 and st.live_baseline_at == at and live["commits"] == 0
    assert r["initial_balance"] == 500.0 and r["net_flows"] == -499.5


async def test_fold_exception_rolls_back_and_the_card_still_serves(db, live, monkeypatch):
    at, usdt_ts = _operator_case()
    await _seed(db, 104.35, at, stored=3100.0, cursor=usdt_ts)
    db.add(models.Order(pair="ETHUSDT", direction="LONG", status="OPEN", is_paper=False, entry_price=100.0, current_price=100.0, quantity=1.0,
                        investment=10.0, leverage=10, notional_value=100.0, confidence="HIGH", entry_strategy="MANUAL",
                        opened_at=datetime(2026, 10, 22, 12, 0), entry_fee=0.0))
    await db.commit()
    live["rows"] = [(usdt_ts, 3100.0)]

    async def _boom(*a, **k):
        raise RuntimeError("db locked")
    monkeypatch.setattr(main, "_live_pretrade_fold", _boom)
    rolled = {"n": 0}
    _rb = db.rollback

    async def _spy():
        rolled["n"] += 1
        await _rb()
    monkeypatch.setattr(db, "rollback", _spy)
    await _bal_case(live)
    r = await main.get_balance(db)
    assert rolled["n"] == 1
    assert r["initial_balance"] == 104.35 and r["net_flows"] == 3100.0          # unfolded numbers, card lines still shown
    assert r["manual_open_count"] == 1 and r["open_orders_count"] == 0          # the expired open rows were re-loaded (no lazy-load crash)
