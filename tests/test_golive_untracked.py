"""🛡 Go-live ② (DECISION_LOG 292): a LIVE Binance position with no OPEN DB row is detected by the reconciler (2 consecutive passes),
logged CRITICAL, counted, surfaced in the status payload and BLOCKS new entries on that symbol. Default action "alert" never touches it;
"flatten" closes it reduce-only. All exchange calls mocked."""
import asyncio, os, sys, json
from datetime import datetime, timedelta
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import find_untracked_positions, untracked_action, UNTRACKED_CONFIRM_PASSES


def _pos(sym, side, qty, px=10.0):
    return {"symbol": sym, "side": side, "contracts": qty, "entry_price": px}


def test_find_untracked_pure():
    pos = [_pos("FOO/USDT:USDT", "LONG", 5), _pos("BAR/USDT:USDT", "SHORT", 2), _pos("BAZ/USDT:USDT", "LONG", 0), _pos("QUX/USDT:USDT", "LONG", 1)]
    rows = [("BARUSDT", "SHORT"), ("QUXUSDT", "SHORT")]
    out = {u["pair"]: u for u in find_untracked_positions(pos, rows)}
    assert set(out) == {"FOOUSDT", "QUXUSDT"}                                    # BAR tracked · BAZ zero size ignored
    assert out["FOOUSDT"]["reason"] == "NO_ROW" and out["FOOUSDT"]["qty"] == 5.0 and out["FOOUSDT"]["side"] == "LONG"
    assert out["QUXUSDT"]["reason"] == "SIDE_MISMATCH"
    assert find_untracked_positions([], rows) == [] and find_untracked_positions(None, None) == []
    # dust below 5 USDT notional and non-USDT-settled symbols are not positions the bot could own
    assert find_untracked_positions([_pos("FOO/USDT:USDT", "LONG", 0.4, 10.0), _pos("FOO/USDC:USDC", "LONG", 5)], []) == []
    assert [u["pair"] for u in find_untracked_positions([_pos("FOO/USDT:USDT", "LONG", 0.5, 10.0)], [])] == ["FOOUSDT"]


def test_action_reads_alert_unless_exactly_flatten():
    class T: pass
    t = T()
    assert untracked_action(t) == "alert"
    for v, want in (("flatten", "flatten"), (" FLATTEN ", "flatten"), ("close", "alert"), ("", "alert"), (None, "alert")):
        t.untracked_position_action = v
        assert untracked_action(t) == want


def test_d11_bundle():
    import config
    assert config.SignalThresholds.model_fields["untracked_position_action"].default == "alert"
    th = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))["thresholds"]
    assert th["untracked_position_action"] == "alert"
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count("config-untracked-position-action") == 3                    # select + load + save
    assert "untracked_position_action: (document.getElementById('config-untracked-position-action')?.value === 'flatten') ? 'flatten' : 'alert'" in html
    assert "Untracked Binance position (DECISION_LOG 292)" in html and html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2   # both exports
    assert 'id="untracked-alert"' in html and "data.untracked_positions" in html
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert '"untracked_positions": [dict(v)' in eng and '"untracked_position_count"' in eng
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "await trading_engine.check_untracked_positions(db, binance_positions)" in main


def _harness(monkeypatch, action="alert", rows=(), close_ok=True):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    S = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(T.config.trading_config.thresholds, "untracked_position_action", action, raising=False)
    log, blocks = [], []
    async def _close(symbol, side, qty):
        log.append(("close", symbol, side, qty)); return {"id": "c"} if close_ok else None
    async def _cancel_all(pair):
        log.append(("cancel_all", pair)); return True
    monkeypatch.setattr(T.binance_service, "close_position", _close)
    monkeypatch.setattr(T.binance_service, "cancel_all_backstops", _cancel_all)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append((name, d))

    async def seed():
        async with engine.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s:
            for pair, d in rows:
                s.add(models.Order(pair=pair, direction=d, status="OPEN", entry_price=1.0, current_price=1.0, investment=10.0, leverage=1.0,
                                   notional_value=10.0, quantity=10.0, confidence="STRONG_BUY", is_paper=False, opened_at=datetime.utcnow()))
            await s.commit()
    return T, eng, engine, S, seed, log, blocks


def test_two_passes_confirm_block_and_resolve(monkeypatch):
    T, eng, engine, S, seed, log, blocks = _harness(monkeypatch, rows=[("BARUSDT", "SHORT")])
    snap = [_pos("FOO/USDT:USDT", "LONG", 5, 2.0), _pos("BAR/USDT:USDT", "SHORT", 2)]
    async def go():
        await seed()
        async with S() as db:
            n1 = await eng.check_untracked_positions(db, snap)
            b1 = eng.untracked_block("FOOUSDT")
            n2 = await eng.check_untracked_positions(db, snap)
            b2 = eng.untracked_block("FOOUSDT")
            n3 = await eng.check_untracked_positions(db, snap)                 # still there: no second alert
            n4 = await eng.check_untracked_positions(db, [_pos("BAR/USDT:USDT", "SHORT", 2)])   # closed / imported → resolved
            b4 = eng.untracked_block("FOOUSDT")
        await engine.dispose()
        return n1, b1, n2, b2, n3, n4, b4
    n1, b1, n2, b2, n3, n4, b4 = asyncio.run(go())
    assert UNTRACKED_CONFIRM_PASSES == 2 and n1 == [] and b1 is None                  # one sighting = maybe a row still being written
    assert len(n2) == 1 and b2["side"] == "LONG" and b2["qty"] == 5.0 and b2["entry_price"] == 2.0 and b2["source"] == "RECONCILE"
    assert n3 == [] and n4 == [] and b4 is None
    assert blocks == [("UNTRACKED_POSITION", "LONG")] and log == []                   # alert: never touched
    assert eng.__dict__["_untracked_detect_count"] == 1


def test_open_in_flight_is_never_flagged(monkeypatch):
    T, eng, engine, S, seed, log, blocks = _harness(monkeypatch)
    snap = [_pos("FOO/USDT:USDT", "LONG", 5)]
    async def go():
        await seed()
        lk = asyncio.Lock(); await lk.acquire()
        eng._pair_open_locks = (asyncio.get_running_loop(), {"FOOUSDT": lk}, {})
        async with S() as db:
            for _ in range(3):
                await eng.check_untracked_positions(db, snap)
            b = eng.untracked_block("FOOUSDT")
        await engine.dispose(); return b
    assert asyncio.run(go()) is None and blocks == []


def test_flatten_action_closes_no_row_positions_only(monkeypatch):
    T, eng, engine, S, seed, log, blocks = _harness(monkeypatch, action="flatten", rows=[("QUXUSDT", "SHORT")])
    snap = [_pos("FOO/USDT:USDT", "LONG", 5), _pos("QUX/USDT:USDT", "LONG", 1)]
    async def go():
        await seed()
        async with S() as db:
            await eng.check_untracked_positions(db, snap)
            await eng.check_untracked_positions(db, snap)
            await eng.check_untracked_positions(db, [snap[1]])                      # FOO gone after the flatten: not closed twice
        await engine.dispose()
    asyncio.run(go())
    assert log == [("close", "FOO/USDT:USDT", "LONG", 5.0), ("cancel_all", "FOOUSDT")]   # the SIDE_MISMATCH (has a DB row) is alert-only
    assert ("UNTRACKED_FLATTENED", "LONG") in blocks and eng.untracked_block("QUXUSDT")["reason"] == "SIDE_MISMATCH"


def test_untracked_symbol_blocks_bot_entries(monkeypatch):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False; eng.is_running = True
    blocks = []
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append((name, d))
    eng._mark_untracked("FOOUSDT", "LONG", 5.0, 2.0, "NO_ROW", "RECONCILE")
    async def go():
        return await eng.open_position(db=None, pair="FOOUSDT", direction="LONG", confidence="STRONG_BUY", current_price=2.0)
    assert asyncio.run(go()) is None and blocks == [("UNTRACKED_POSITION_BLOCK", "LONG")]


def test_reconciler_runs_the_inverse_check_only_on_a_real_snapshot(monkeypatch):
    import main
    calls = []
    async def _chk(db, positions): calls.append(positions); return []
    async def _orph(db, pairs): return []
    monkeypatch.setattr(main.trading_engine, "check_untracked_positions", _chk)
    monkeypatch.setattr(main, "_close_orphan_orders", _orph)
    async def _ok(): return [_pos("FOO/USDT:USDT", "LONG", 5)]
    async def _fail(): return None
    monkeypatch.setattr(main.binance_service, "get_open_positions", _ok)
    asyncio.run(main._reconcile_open_orders(None))
    monkeypatch.setattr(main.binance_service, "get_open_positions", _fail)
    monkeypatch.setattr(main, "_reconcile_skip_counter", 0)
    asyncio.run(main._reconcile_open_orders(None))
    assert len(calls) == 1 and calls[0][0]["symbol"] == "FOO/USDT:USDT"


def test_flatten_retries_a_position_still_shown_a_pass_later(monkeypatch):
    T, eng, engine, S, seed, log, blocks = _harness(monkeypatch, action="flatten")
    snap = [_pos("FOO/USDT:USDT", "LONG", 5)]
    async def go():
        await seed()
        async with S() as db:
            for _ in range(3):
                await eng.check_untracked_positions(db, snap)                       # pass 3: a residual is still there → retried
        await engine.dispose()
    asyncio.run(go())
    assert [x[0] for x in log].count("close") == 2
