"""✎ Exit override (DECISION_LOG 182): the operator replaces an open position's exit with his own stop / target — manual AND bot trades."""
import asyncio, datetime as dt, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _setup(monkeypatch, *, paper=True, price=1.0, **row):
    """An engine + one OPEN order on in-memory SQLite. Returns run(fn) where fn(engine, db, order_id) is awaited."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper
    async def _noop(*a, **k): return None
    eng.update_orders_cache = _noop
    class _Trk: last_price = price
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    base = dict(pair="FOOUSDT", direction="LONG", status="OPEN", entry_price=1.0, quantity=1000.0, investment=50.0, leverage=20.0, notional_value=1000.0,
                confidence="STRONG_BUY", entry_fee=0.45, is_paper=paper, entry_strategy="MOMENTUM", opened_at=dt.datetime.utcnow() - dt.timedelta(minutes=5))
    base.update(row)

    async def run(fn):
        e = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        try:
            async with async_sessionmaker(e, expire_on_commit=False)() as db:
                o = models.Order(**base); db.add(o); await db.commit(); await db.refresh(o)
                try:
                    return await fn(eng, db, o.id)
                except ValueError as err:
                    return "ValueError: " + str(err)
        finally:
            await e.dispose()
    return run, T


def test_bot_trade_gets_the_operators_stop_and_target(monkeypatch):
    run, T = _setup(monkeypatch)                                        # a momentum long sitting at ≈ −0.09 % (fees)
    async def go(eng, db, oid):
        o = await eng.set_exit_override(db, oid, sl_pct=-1.5, tp_pct=2.0)
        return o.entry_strategy, o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct, o.exit_override_at is not None, o.manual_exit_mode, o.exit_override_prev
    strat, sl, tp, flagged, mode, prev = asyncio.run(run(go))
    assert (strat, sl, tp, flagged, mode) == ("MOMENTUM", -1.5, 2.0, True, None) and prev.startswith("BOT")   # label kept, exit replaced
    async def lock(eng, db, oid):                                       # a positive stop locks a profit — the trade must already be above it
        return await eng.set_exit_override(db, oid, sl_pct=0.5)
    assert "would close it at once" in asyncio.run(run(lock))
    run2, _ = _setup(monkeypatch, price=1.02)                           # now at ≈ +1.9 %
    async def lock2(eng, db, oid):
        o = await eng.set_exit_override(db, oid, sl_pct=0.5, tp_pct=4.0); return o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct
    assert asyncio.run(run2(lock2)) == (0.5, 4.0)


def test_manual_trade_becomes_custom_and_keeps_a_stop(monkeypatch):
    run, T = _setup(monkeypatch, entry_strategy="MANUAL", manual_exit_mode="MOMENTUM")
    async def only_tp(eng, db, oid):
        o = await eng.set_exit_override(db, oid, tp_pct=3.0)
        return o.manual_exit_mode, o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct, o.entry_strategy
    assert "type a stop too" in asyncio.run(run(only_tp))                                     # no fixed stop to keep → a stop is REQUIRED (never invented)
    async def both(eng, db, oid):
        o = await eng.set_exit_override(db, oid, sl_pct=-1.0, tp_pct=3.0)
        return o.manual_exit_mode, o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct, o.entry_strategy
    assert asyncio.run(run(both)) == ("FIXED", -1.0, 3.0, "MANUAL")
    run, _ = _setup(monkeypatch, entry_strategy="MANUAL", manual_exit_mode="FLOOR", pattern_fixed_sl_pct=-2.0)
    async def keep(eng, db, oid):
        o = await eng.set_exit_override(db, oid, tp_pct=1.0); return o.pattern_fixed_sl_pct, o.pattern_fixed_tp_pct
    assert asyncio.run(run(keep)) == (-2.0, 1.0)                                                # an existing fixed stop is kept


def test_refusals(monkeypatch):
    run, T = _setup(monkeypatch)
    call = lambda **kw: asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, **kw)))
    assert "give a stop, a target, or both" in call()
    assert "wider than the widest stop allowed" in call(sl_pct=-9.0)
    assert "target must be above the stop" in call(sl_pct=-1.0, tp_pct=-2.0)
    assert "already reached" in call(sl_pct=-2.0, tp_pct=-0.5)
    assert "type a stop too" in call(tp_pct=2.0)                       # a bot trade has no fixed stop to keep: a blank stop is refused, never −3 %
    assert "must be a number" in call(sl_pct="abc")
    assert "not open" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid + 99, sl_pct=-1)))
    run, _ = _setup(monkeypatch, paper=True, is_paper=False)
    assert "other mode" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, sl_pct=-1)))
    # LIVE: the stop must stay inside the resting exchange safety stop (bot trade: 2.5 % → at most −2.2)
    run, T2 = _setup(monkeypatch, paper=False)
    th = T2.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    assert "too close to it" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, sl_pct=-2.5)))
    async def ok(eng, db, oid):
        return (await eng.set_exit_override(db, oid, sl_pct=-2.0)).pattern_fixed_sl_pct
    assert asyncio.run(run(ok)) == -2.0


def test_wiring_everywhere():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_ovr_rt = order_info.get('exit_override_at') is not None" in eng and "_ovr_m = getattr(order, 'exit_override_at', None) is not None" in eng
    assert '_mn_lbl + "_SL"' in eng and '_mf_lbl + "_SL"' in eng and eng.count('"MANUAL_", "OVERRIDE_",') == 2          # both paths; both urgent lists (taker)
    assert "'exit_override_at': getattr(order, 'exit_override_at', None)," in eng                                        # survives the cache rebuild
    import models as M
    assert {"exit_override_at", "exit_override_prev"} <= {c.name for c in M.Order.__table__.columns}
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "('exit_override_at', 'DATETIME')" in db and "('exit_override_prev', 'VARCHAR(40)')" in db
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '@app.post("/api/orders/exit_override")' in main and main.count('"exit_override_at": (o.exit_override_at.isoformat()') == 2
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "function closeChoiceNear(" in html and "/api/orders/exit_override" in html and "o.pnl_percentage.toFixed(2) : 'null'})" in html


def test_refused_while_closing_and_unknown_price(monkeypatch):
    run, T = _setup(monkeypatch, closing_in_progress=True)
    assert "being closed right now" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, sl_pct=-1)))
    run, T = _setup(monkeypatch)
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: None)          # no live price and no stored price
    assert "not available" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, sl_pct=-1)))
    run, T = _setup(monkeypatch, direction="SHORT", price=0.98)                        # a SHORT ≈ +1.9 %: profit lock at +1 is accepted, +2.5 is not
    async def lock(eng, db, oid):
        return (await eng.set_exit_override(db, oid, sl_pct=1.0)).pattern_fixed_sl_pct
    assert asyncio.run(run(lock)) == 1.0
    assert "would close it at once" in asyncio.run(run(lambda eng, db, oid: eng.set_exit_override(db, oid, sl_pct=2.5)))


def test_live_edits_cannot_walk_past_the_resting_safety_stop(monkeypatch):
    """Review: the row no longer tells where the exchange stop was PLACED once it was edited — a second edit is judged against the
    bot's distance (2.5 % → at most −2.2), so repeated edits can never step beyond the real stop."""
    run, T = _setup(monkeypatch, paper=False, entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_sl_pct=-1.0)
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    async def twice(eng, db, oid):
        a = (await eng.set_exit_override(db, oid, sl_pct=-2.2)).pattern_fixed_sl_pct
        try:
            await eng.set_exit_override(db, oid, sl_pct=-2.4); b = "accepted"
        except ValueError as e:
            b = str(e)
        return a, b
    a, b = asyncio.run(run(twice))
    assert a == -2.2 and "too close to it" in b
    assert T.manual_own_stop_pct(th, "FIXED", 4.0) is None and T.order_backstop_pct(th, "MANUAL", "FIXED", 4.0, 20) == 2.5   # a profit lock is not a 4 % loss stop


def test_overridden_bot_trade_is_closed_only_by_the_operators_levels(monkeypatch):
    """Behavioural: a MOMENTUM long (own stop −0.7) with an override −1.5 / +3 stays open at −1.0 and +2, closes OVERRIDE_SL / OVERRIDE_TP
    exactly at the operator's levels (LONG and SHORT, and a profit-lock stop), through the real realtime path."""
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE)
    closed = []

    class _Res:
        def __init__(self, o): self.o = o
        def scalar_one_or_none(self): return self.o

    class _DB:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *a, **k): return _Res(object())

    async def _close(db, order, price, reason): closed.append(reason); return True
    monkeypatch.setattr(T, "AsyncSessionLocal", lambda: _DB())
    monkeypatch.setattr(eng, "close_position", _close, raising=False)
    fee = float(getattr(T.config.trading_config, 'taker_fee', T.config.trading_config.trading_fee))

    def entry(direction, strategy, sl, tp, peak=0.0):
        return {'id': 1, 'direction': direction, 'entry_price': 100.0, 'quantity': 10.0, 'entry_fee': 1000.0 * fee, 'confidence': 'STRONG_BUY',
                'opened_at': T.datetime.utcnow(), 'entry_strategy': strategy, 'manual_exit_mode': None, 'pattern_fixed_sl_pct': sl, 'pattern_fixed_tp_pct': tp,
                'exit_override_at': T.datetime.utcnow(), 'stop_loss': -0.7, 'peak_pnl': peak, 'trough_pnl': 0.0, 'leverage': 20.0, 'notional_value': 1000.0,
                'investment': 50.0, 'entry_atr_pct': 2.0}

    async def drive(direction, strategy, sl, tp, price, peak=0.0):
        closed.clear()
        T._open_orders_cache["ZZZUSDT"] = [entry(direction, strategy, sl, tp, peak)]
        await eng.check_realtime_stop_loss("ZZZUSDT", price)
        return list(closed)
    try:
        for strat in ("MOMENTUM", "FRENZY_LONG", "SURGE_LONG", "BULLRUN_LONG"):
            assert asyncio.run(drive("LONG", strat, -1.5, 3.0, 99.0)) == [], strat          # −1.1 %: past the momentum stop, inside the operator's
            assert asyncio.run(drive("LONG", strat, -1.5, 3.0, 102.0)) == [], strat
            assert asyncio.run(drive("LONG", strat, -1.5, 3.0, 98.0)) == ["OVERRIDE_SL"], strat
            assert asyncio.run(drive("LONG", strat, -1.5, 3.0, 104.0)) == ["OVERRIDE_TP"], strat
        for strat in ("MOMENTUM", "SURGE_SHORT"):
            assert asyncio.run(drive("SHORT", strat, -1.5, 3.0, 101.0)) == [], strat
            assert asyncio.run(drive("SHORT", strat, -1.5, 3.0, 102.0)) == ["OVERRIDE_SL"], strat
            assert asyncio.run(drive("SHORT", strat, -1.5, 3.0, 96.0)) == ["OVERRIDE_TP"], strat
        assert asyncio.run(drive("LONG", "MOMENTUM", 1.0, None, 102.0, peak=2.5)) == []       # profit lock at +1: holds at +1.9
        assert asyncio.run(drive("LONG", "MOMENTUM", 1.0, None, 100.9, peak=2.5)) == ["OVERRIDE_SL"]
    finally:
        T._open_orders_cache.pop("ZZZUSDT", None)
