"""🖐 Sep-29 MANUAL sleeve — input validation, exit-mode semantics, exclusion from systematic reads, D11/D12 wiring parity."""
import json, os, sys, asyncio, inspect
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _synthetic_ohlcv(n, seed=7):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h = c * (1 + rng.uniform(0, 0.002, n)); l = c * (1 - rng.uniform(0, 0.002, n)); o = np.r_[c[0], c[:-1]]
    return [[1_700_000_000_000 + i * 300_000, float(o[i]), float(h[i]), float(l[i]), float(c[i]), float(1000 + 50 * i)] for i in range(n)]


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
    assert "dynamic_tp_target=(float(getattr(config.trading_config.confidence_levels.get(\"STRONG_BUY\"), 'tp_min', 0.4)" in src and 'current_tp_level=1' in src   # UI arm badge seed + cache level never None


def test_exits_never_touch_a_fixed_manual_trade_and_cache_carries_the_mode():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("(order_info.get('entry_strategy') or '') == 'MANUAL'") == 1          # realtime intercept, before BULLRUN
    assert eng.index("== 'MANUAL'\n                    and (order_info.get('manual_exit_mode')") < eng.index("if (order_info.get('entry_strategy') or '') in ('BULLRUN_LONG', 'SURGE_LONG', 'SURGE_SHORT'):")
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
    # FLOOR: no TP unless given (Sep-30: optional TP on top of the floor stop)
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, None, 110.0)) == []
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, None, 96.0)) == ["MANUAL_SL"]
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, 5.0, 106.0)) == ["MANUAL_TP"]
    assert asyncio.run(drive("SHORT", "FLOOR", -3.0, 5.0, 94.0)) == ["MANUAL_TP"]
    assert asyncio.run(drive("LONG", "FLOOR", -3.0, 5.0, 103.0)) == []            # between the floor stop and the TP: holds
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


def test_momentum_mode_refuses_only_what_the_ema13_exit_would_close_at_once(monkeypatch):
    """QNT (Sep-29): a SHORT above EMA13 in Momentum mode was closed by EMA13_CROSS_EXIT at the first tick — refuse it up front."""
    import services.trading_engine as T, datetime as _dt
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True

    async def _bal(db): return 10_000.0
    eng.get_available_balance = _bal
    async def _bnb(db): return 1_000.0                                 # BNB reserve covers the fee (the fee-aware balance checks)
    eng._recalculate_paper_bnb = _bnb

    class _Trk: last_price = 268.0
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)
    _bars = _synthetic_ohlcv(260)                                                  # 📐 the entry stamps fetch klines: never the network in tests

    async def _ohlcv(symbol, tf, n): return _bars[-n:]
    async def _fr(symbol): return 0.0001
    monkeypatch.setattr(T.binance_service, "get_ohlcv", _ohlcv)
    monkeypatch.setattr(T.binance_service, "fetch_funding_rate", _fr)

    class _Accepted(Exception): pass
    captured = []

    def db_with(ema13, age_s, ema5=269.0, ema8=268.5):
        class _Row: pass
        row = None
        if ema13 is not None:
            row = _Row(); row.ema13 = ema13; row.ema5 = ema5; row.ema8 = ema8; row.updated_at = _dt.datetime.utcnow() - _dt.timedelta(seconds=age_s)
            row.signal = "NO_TRADE"; row.confidence = "STRONG_BUY"; row.price = 268.0; row.ema20 = ema13; row.rsi = 50.0; row.adx = 20.0

        class _Res:
            def __init__(self, first=None, scalar=0): self._f, self._s = first, scalar
            def scalar(self): return self._s
            def first(self): return self._f

        class _DB:
            def __init__(self): self.n = 0
            async def execute(self, *a, **k):
                self.n += 1
                return _Res(first=row) if self.n == 3 else _Res()      # 1 dup check, 2 manual count, 3 PairData
            def add(self, o): captured.append(o); raise _Accepted()
        return _DB()

    def attempt(direction, ema13, age_s=10, leverage=20, exit_mode="MOMENTUM", tp_pct=None, **kw):
        try:
            asyncio.run(TE.open_manual_position(eng, db_with(ema13, age_s, **kw), pair="QNTUSDT", direction=direction, investment=100, leverage=leverage, exit_mode=exit_mode, tp_pct=tp_pct))
        except ValueError as e:
            return str(e)
        except _Accepted:                                              # reached db.add = every guard passed
            return "ACCEPTED"
        raise AssertionError("stub DB should have raised _Accepted")
    th = T.config.trading_config.thresholds
    old = (th.ema13_cross_exit_enabled, th.ema13_cross_exit_long_enabled, th.ema13_cross_exit_short_enabled, th.ema13_cross_requires_stack_flip)
    try:
        th.ema13_cross_exit_enabled = True; th.ema13_cross_exit_long_enabled = True; th.ema13_cross_exit_short_enabled = True; th.ema13_cross_requires_stack_flip = True
        # QNT: SHORT at 268 above EMA13 266 with the stack long (EMA5 269 > EMA8 268.5) → the exit fires at the first tick
        assert "would close this SHORT at the first tick" in attempt("SHORT", 266.0)
        # strict mode: same SHORT but the stack is NOT against it (EMA5 < EMA8) → the exit would hold → accepted
        assert attempt("SHORT", 266.0, ema5=267.0, ema8=268.0) == "ACCEPTED"
        assert "would close this LONG at the first tick" in attempt("LONG", 270.0, ema5=267.0, ema8=268.0)
        assert attempt("LONG", 266.0) == "ACCEPTED" and attempt("SHORT", 270.0) == "ACCEPTED"          # with-trend
        assert attempt("LONG", None) == "ACCEPTED"                                                        # pair not scanned: no EMA exits
        # a row that exists is judged even when stale (deep review: the first refreshed tick closed a stale-admitted short)
        _m = attempt("SHORT", 266.0, age_s=1200); assert "at the first tick" in _m and "min old" in _m
        assert attempt("SHORT", 270.0, age_s=1200) == "ACCEPTED"
        # leverage: the stack's widest stop (signal_active_sl) must sit inside the floor
        _sb = T.config.trading_config.confidence_levels.get("STRONG_BUY"); _widest = min(_sb.stop_loss, _sb.signal_active_sl)
        _hi = next(l for l in range(20, 126) if T.manual_floor_for_leverage(th, l) > _widest)             # first leverage whose floor is tighter than the stop
        assert "momentum stack's stop" in attempt("LONG", 266.0, leverage=_hi) and attempt("LONG", 266.0, leverage=_hi - 1) == "ACCEPTED"
        captured.clear(); assert attempt("LONG", 266.0, leverage=50, exit_mode="FLOOR") == "ACCEPTED"
        assert captured[-1].pattern_fixed_sl_pct == T.manual_floor_for_leverage(th, 50) == -1.2 and captured[-1].pattern_fixed_tp_pct is None
        captured.clear(); assert attempt("LONG", 266.0, leverage=20, exit_mode="FLOOR", tp_pct=8) == "ACCEPTED"   # FLOOR + optional TP
        assert captured[-1].pattern_fixed_tp_pct == 8.0 and captured[-1].pattern_fixed_sl_pct == T.manual_floor_for_leverage(th, 20)
        assert "take profit must be > 0" in attempt("LONG", 266.0, exit_mode="FLOOR", tp_pct=0)
        assert "take profit must be > 0" in attempt("LONG", 266.0, exit_mode="FLOOR", tp_pct=float("nan"))
        captured.clear(); assert attempt("LONG", 266.0, exit_mode="FLOOR", tp_pct="") == "ACCEPTED"
        assert captured[-1].pattern_fixed_tp_pct is None
        captured.clear(); assert attempt("LONG", 266.0, exit_mode="MOMENTUM", tp_pct=8) == "ACCEPTED"          # MOMENTUM ignores a TP
        assert captured[-1].pattern_fixed_tp_pct is None
        # the gate context lands on the Order itself (behavioural, not a source grep): stub row = NO_TRADE / STRONG_BUY, fan long
        _o = captured[-1]
        assert _o.manual_block_reason == T.PAIR_REASON_AWAITING and _o.manual_setup_rating == "STRONG_BUY" and _o.manual_setup_side == "LONG"
        assert _o.manual_pair_rsi == 50.0 and _o.manual_pair_adx == 20.0 and _o.manual_gap_5_20 is not None and _o.manual_gap_5_20 >= 0
        # 📐 Sep-29: a manual fill now records the bot's entry_* stamps too (operator request); entry_strategy = MANUAL keeps
        # it out of every systematic read (test_pair_block_reason pins that with every entry_* column populated)
        assert _o.entry_strategy == "MANUAL" and _o.entry_rsi is not None and _o.entry_adx is not None and _o.entry_gap is not None
        th.ema13_cross_exit_long_enabled = False                                                           # live config today
        assert attempt("LONG", 270.0, ema5=267.0, ema8=268.0) == "ACCEPTED"                                # LONG side disabled → phantom only
    finally:
        (th.ema13_cross_exit_enabled, th.ema13_cross_exit_long_enabled, th.ema13_cross_exit_short_enabled, th.ema13_cross_requires_stack_flip) = old


def test_manual_floor_is_leverage_aware():
    import services.trading_engine as T
    from types import SimpleNamespace as NS
    th = NS(manual_floor_sl_pct=-3.0)
    f = T.manual_floor_for_leverage
    assert f(th, 20) == -3.0 and f(th, 10) == -3.0                 # config floor binds (liquidation ≈ −4.5 % / −9.5 %)
    assert abs(f(th, 30) - (-2.2667)) < 1e-3 and f(th, 50) == -1.2 and f(th, 100) == -0.4      # 0.8 × (100/lev − 0.5 maintenance margin)
    assert f(NS(manual_floor_sl_pct=-1.0), 50) == -1.0 and f(NS(manual_floor_sl_pct=-1.0), 100) == -0.4
    assert T.manual_liquidation_distance_pct(50) == 1.5 and T.manual_liquidation_distance_pct(250) == 0.0
    assert f(th, None) == -3.0 and f(th, "x") == -3.0 and f(th, 0) == -3.0
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True
    for lev, sl, ok in [(50, 1.1, True), (50, 1.5, False), (20, 2.5, True), (100, 1.0, False), (30, 2.2, True), (30, 2.4, False)]:
        try:
            asyncio.run(TE.open_manual_position(eng, None, pair="DOGEUSDT", direction="LONG", investment=100, leverage=lev, exit_mode="FIXED", sl_pct=sl))
            raise AssertionError("no DB → must raise")
        except ValueError as e:
            assert ("wider than the widest stop allowed" in str(e)) == (not ok), (lev, sl, str(e))
        except Exception:
            assert ok, (lev, sl)                                    # got past the floor check (then fails on the None db)


def test_gate_context_is_stamped_automatically():
    """Operator request: the gate shown on the Top Pairs row at the click is recorded on the manual order (no hand-typed note)."""
    import services.trading_engine as T, datetime as _dt
    from types import SimpleNamespace as NS
    now = _dt.datetime.utcnow()
    st = T.PairReasonStash(); st.cur_seq = 3
    row = lambda **k: NS(**{**dict(signal="NO_TRADE", confidence="STRONG_BUY", price=268.0, ema5=268.6, ema8=268.4, ema13=268.2, ema20=267.7, rsi=54.2, adx=16.8, updated_at=now), **k})
    st["QNTUSDT"] = "ATR_GAP_LONG"; st.mark_verdict("QNTUSDT")
    g = T.manual_gate_context(st, "QNTUSDT", row(), now=now)
    assert g["block_reason"] == "ATR_GAP_LONG" and g["rating"] == "STRONG_BUY" and g["side"] == "LONG" and g["rsi"] == 54.2 and g["adx"] == 16.8
    assert g["gap_5_8"] > 0 and g["gap_5_20"] > 0 and g["px_vs_ema5"] < 0
    # enterable pair: signal LONG and nothing refused this verdict → the explicit word NONE (the CSV writes None as an empty cell)
    st.mark_verdict("SOONUSDT")
    assert T.manual_gate_context(st, "SOONUSDT", row(signal="LONG", confidence="VERY_STRONG"), now=now)["block_reason"] == T.MANUAL_NO_GATE == "NONE"
    # rated LONG but a late gate refused the same verdict
    st["DOGEUSDT"] = "LONG_HEAT_BLOCK"; st.mark_verdict("DOGEUSDT")
    assert T.manual_gate_context(st, "DOGEUSDT", row(signal="LONG"), now=now)["block_reason"] == "LONG_HEAT_BLOCK"
    # bearish fan → setup side SHORT, gaps stay ABSOLUTE like every systematic stamp; nothing stamped → what the row shows
    gb = T.manual_gate_context(st, "BTCUSDT", row(ema5=100.0, ema8=101.0, ema13=102.0, ema20=103.0, price=99.5, confidence="NO_TRADE"), now=now, live_price=99.0)
    assert gb["side"] == "SHORT" and gb["gap_5_20"] > 0 and gb["gap_5_8"] > 0 and gb["gap_8_13"] > 0
    assert gb["px_vs_ema5"] == round((99.0 - 100.0) / 100.0 * 100, 4)                      # the FILL price, signed
    assert gb["block_reason"] == T.PAIR_REASON_AWAITING
    # no row / stale row → everything None (the pair is not in the scanned list)
    assert T.manual_gate_context(st, "XUSDT", None)["block_reason"] is None
    assert T.manual_gate_context(st, "QNTUSDT", row(updated_at=now - _dt.timedelta(minutes=30)), now=now)["block_reason"] is None
    assert T.manual_gate_context(None, "QNTUSDT", row(), now=now)["block_reason"] == T.PAIR_REASON_AWAITING      # no stash: never raises
    # wiring
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "manual_block_reason=(None if _gc['block_reason'] is None else str(_gc['block_reason'])[:60])" in eng
    models = open(os.path.join(ROOT, "models.py"), encoding="utf-8").read(); db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    for c in ("manual_block_reason", "manual_setup_rating", "manual_setup_side"):
        assert f"{c} = Column(String" in models and f"'{c}'" in db
    for c in ("manual_pair_rsi", "manual_pair_adx", "manual_gap_5_20", "manual_gap_5_8", "manual_gap_8_13", "manual_px_vs_ema5"):
        assert f"{c} = Column(Float" in models and f"('{c}', 'FLOAT')" in db
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read(); assert main.count('"manual_block_reason": getattr(') == 2
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "function manualPairHint()" in ui and 'id="manual-pair-hint"' in ui and "gate at entry:" in ui


def test_manual_panel_always_starts_closed():
    """Operator, Sep-29: the manual entry panel is closed on every page load — hidden in the markup, folded by the init call,
    and its state is not remembered per browser."""
    import os
    ui = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()
    assert '<div id="manual-entry-body" class="hidden">' in ui
    assert 'id="manual-entry-chevron" class="text-gray-500">▸<' in ui
    assert "try { toggleManualPanel(true); } catch (e) {}" in ui
    assert "manualPanelCollapsed" not in ui


def test_manual_trades_never_enter_a_bot_cell_baseline():
    """Sep-30 (operator): the pattern-cell report graded bot cells against a per-direction baseline that counted closed MANUAL
    trades (no cell tag → 'baseline'). Since Sep-29 the performance endpoint already passes MANUAL-free orders; this pins every
    cell/pattern report to drop them itself too (defense in depth for any other caller)."""
    import types
    import main as M
    class _O(types.SimpleNamespace):
        def __getattr__(self, k):                                            # every other Order column reads as NULL
            return None
    mk = lambda pct, strat="MOMENTUM", src=None, mult=1.0: _O(
        status="CLOSED", pnl=pct, pnl_percentage=pct, direction="LONG", entry_strategy=strat, pattern_cell_source=src,
        cell_multiplier=mult, cell_lev_multiplier=1.0, investment=100.0, leverage=20.0, opened_at=None, closed_at=None)
    base = [mk(0.2), mk(0.4), mk(0.3, src="W1")]
    with_manual = base + [mk(11.0, strat="MANUAL")]
    r1 = M._compute_pattern_cell_performance(base)
    r2 = M._compute_pattern_cell_performance(with_manual)
    b = lambda r: [x.get("baseline_avg_pct") for x in r["rules"] if x.get("baseline_avg_pct") is not None]
    assert b(r1) == b(r2) and b(r1) and abs(b(r1)[0] - 0.3) < 1e-9          # the manual +11 % does not move the ruler
    for fn in (M._compute_multiplier_cell_performance, M._compute_extension_multiplier_performance,
               M._compute_btc_1h_slope_btc_adx_multiplier_performance, M._compute_pattern_4cohort_coverage,
               M._compute_pattern_combo_tracker):
        assert repr(fn(base)) == repr(fn(with_manual)), fn.__name__          # a MANUAL row changes nothing in any cell report
