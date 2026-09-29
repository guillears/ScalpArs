"""🖐 Sep-29 MANUAL sleeve — input validation, exit-mode semantics, exclusion from systematic reads, D11/D12 wiring parity."""
import json, os, sys, asyncio, inspect
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _engine():
    from services.trading_engine import TradingEngine
    return TradingEngine


def _run(coro):
    return asyncio.run(coro)


def test_validation_rejects_bad_input_before_touching_db_or_exchange():
    TE = _engine(); eng = TE.__new__(TE); eng.is_paper_mode = True
    for kw in [dict(direction="UP"), dict(exit_mode="NONE"), dict(investment=0), dict(leverage=0), dict(leverage=200),
               dict(exit_mode="FIXED", sl_pct=None), dict(exit_mode="FIXED", sl_pct=9.0)]:
        args = dict(pair="DOGEUSDT", direction="LONG", investment=100, leverage=20, exit_mode="FIXED", sl_pct=1.5); args.update(kw)
        try:
            _run(TE.open_manual_position(eng, None, **args)); raise AssertionError(f"accepted {kw}")
        except ValueError:
            pass


def test_floor_and_fixed_semantics_from_config():
    import config as C
    assert C.SignalThresholds.model_fields["manual_floor_sl_pct"].default == -3.0
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["manual_floor_sl_pct"] == -3.0
    src = inspect.getsource(_engine().open_manual_position)
    assert 'exit_mode == "FLOOR"' in src and "sl = floor" in src and 'sl = -abs(float(sl_pct))' in src   # sign ignored, floor enforced


def test_exits_never_touch_a_fixed_manual_trade_and_cache_carries_the_mode():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("(order_info.get('entry_strategy') or '') == 'MANUAL'") == 1          # realtime intercept, before BULLRUN
    assert eng.index("== 'MANUAL'\n                    and (order_info.get('manual_exit_mode')") < eng.index("if (order_info.get('entry_strategy') or '') == 'BULLRUN_LONG':")
    assert '(order.entry_strategy or "") == "MANUAL" and (getattr(order, \'manual_exit_mode\', None) or "FIXED") in ("FIXED", "FLOOR")' in eng   # candle loop skip
    assert "'manual_exit_mode': getattr(order, 'manual_exit_mode', None)" in eng             # cache rebuild
    assert eng.count('"MANUAL_TP"') == 1 and eng.count('"MANUAL_SL"') == 1


def test_manual_is_excluded_from_ledger_and_pool_and_has_own_row():
    from scripts.build_master_pool import STACK_VERSION  # importable
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert "if strat == 'MANUAL':\n            k, why = False, 'MANUAL_EXEMPT'" in bld and "if not r.is_probe and strat != 'MANUAL':" in bld
    led = open(os.path.join(ROOT, "scripts", "current_stack_ledger.py")).read()
    assert 'eq("MANUAL")' in led and "MANUAL fills excluded" in led
    main = open(os.path.join(ROOT, "main.py")).read()
    assert "return 'Manual'" in main and main.count('"manual_exit_mode": getattr(') == 2 and '@app.post("/api/orders/manual_open")' in main
    assert "class ManualOpenRequest(BaseModel)" in main


def test_ui_wiring_parity():
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    for i in ("manual-pair", "manual-direction", "manual-investment", "manual-leverage", "manual-exit-mode", "manual-sl", "manual-tp", "manual-note", "manual-open-btn"):
        assert ui.count(f"id=\"{i}\"") == 1, i
    assert "async function openManualPosition()" in ui and "/api/orders/manual_open" in ui
    assert ui.count("config-manual-floor-sl-pct") == 3 and "Manual sleeve (Sep 29)" in ui
    models = open(os.path.join(ROOT, "models.py")).read(); db = open(os.path.join(ROOT, "database.py")).read()
    assert "manual_exit_mode = Column(String(12)" in models and "manual_note = Column(String(200)" in models
    assert "('manual_exit_mode', 'VARCHAR(12)'), ('manual_note', 'VARCHAR(200)')" in db


def test_realtime_intercept_closes_on_sl_and_tp_and_passes_momentum_through(monkeypatch):
    """Behavioural: drive check_realtime_stop_loss on a fake cache entry — MANUAL_SL at pnl ≤ sl, MANUAL_TP at pnl ≥ tp (LONG and
    SHORT signs), and a MOMENTUM-mode manual order is NOT intercepted (falls through to the stack)."""
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

    def entry(direction, mode, sl, tp, entry_price=100.0):
        qty = 1000.0 / entry_price
        return {'id': 1, 'direction': direction, 'entry_price': entry_price, 'quantity': qty, 'entry_fee': 1000.0 * fee, 'confidence': 'STRONG_BUY',
                'opened_at': T.datetime.utcnow(), 'entry_strategy': 'MANUAL', 'manual_exit_mode': mode, 'pattern_fixed_sl_pct': sl, 'pattern_fixed_tp_pct': tp,
                'stop_loss': -0.7, 'peak_pnl': 0.0, 'trough_pnl': 0.0, 'leverage': 20.0, 'notional_value': 1000.0, 'investment': 50.0}

    async def drive(direction, mode, sl, tp, price):
        closed.clear()
        T._open_orders_cache["ZZZUSDT"] = [entry(direction, mode, sl, tp)]
        await eng.check_realtime_stop_loss("ZZZUSDT", price)
        return list(closed)

    # LONG FIXED: −1.5 / +3.0 (net of fees → price moves of ≈ −1.4 / +3.1 clear them decisively at −2 / +4)
    assert asyncio.run(drive("LONG", "FIXED", -1.5, 3.0, 98.0)) == ["MANUAL_SL"]
    assert asyncio.run(drive("LONG", "FIXED", -1.5, 3.0, 104.0)) == ["MANUAL_TP"]
    assert asyncio.run(drive("LONG", "FIXED", -1.5, 3.0, 100.5)) == []            # between SL and TP: nothing closes
    # SHORT signs
    assert asyncio.run(drive("SHORT", "FIXED", -1.5, 3.0, 102.0)) == ["MANUAL_SL"]
    assert asyncio.run(drive("SHORT", "FIXED", -1.5, 3.0, 96.0)) == ["MANUAL_TP"]
    # FLOOR: no TP ever
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, None, 110.0)) == []
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, None, 96.0)) == ["MANUAL_SL"]
    # MOMENTUM mode is not intercepted by the manual block (whatever else fires, it is not MANUAL_*)
    assert not any(r.startswith("MANUAL_") for r in asyncio.run(drive("LONG", "MOMENTUM", None, None, 98.0)))
    T._open_orders_cache.pop("ZZZUSDT", None)


def test_manual_slot_lane_wiring():
    """Manual positions never consume a bot slot (open_position count excludes MANUAL) and are capped by their own field."""
    import config as C
    assert C.InvestmentConfig.model_fields["manual_max_open_positions"].default == 8
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["investment"]["manual_max_open_positions"] == 8
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert 'or_(Order.entry_strategy.is_(None), Order.entry_strategy != "MANUAL")' in eng            # bot slot count
    assert 'Order.entry_strategy == "MANUAL")))).scalar() or 0' in eng and "'manual_max_open_positions', 8" in eng
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-manual-max-open-positions") >= 3 and "own slot lane, cap" in ui


def test_manual_cap_zero_means_off_and_bot_count_excludes_manual():
    import services.trading_engine as T
    from sqlalchemy import and_, select, func
    from models import Order
    sql = str(select(func.count(Order.id)).where(and_(Order.status == "OPEN", T._bot_open_filter())).compile(compile_kwargs={"literal_binds": True}))
    assert "entry_strategy IS NULL OR orders.entry_strategy != 'MANUAL'" in sql.replace("orders.entry_strategy IS NULL", "entry_strategy IS NULL")
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True
    old = T.config.trading_config.investment.manual_max_open_positions
    try:
        T.config.trading_config.investment.manual_max_open_positions = 0

        class _Res:
            def scalar(self): return 0

        class _DB:
            async def execute(self, *a, **k): return _Res()
        try:
            asyncio.run(TE.open_manual_position(eng, _DB(), pair="DOGEUSDT", direction="LONG", investment=100, leverage=20, exit_mode="FLOOR"))
            raise AssertionError("cap 0 did not refuse")
        except ValueError as e:
            assert "manual entry is off" in str(e)
        T.config.trading_config.investment.manual_max_open_positions = 2

        class _Res2:
            def scalar(self): return 2          # dup query → treated as "2 open on pair"? no: first query is the dup check

        class _DB2:
            def __init__(self): self.n = 0
            async def execute(self, *a, **k):
                self.n += 1
                r = _Res(); r.scalar = (lambda: 0) if self.n == 1 else (lambda: 2)   # dup check 0, manual count 2 ≥ cap 2
                return r
        try:
            asyncio.run(TE.open_manual_position(eng, _DB2(), pair="DOGEUSDT", direction="LONG", investment=100, leverage=20, exit_mode="FLOOR"))
            raise AssertionError("cap reached did not refuse")
        except ValueError as e:
            assert "manual positions cap reached (2/2" in str(e)
    finally:
        T.config.trading_config.investment.manual_max_open_positions = old
