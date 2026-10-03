"""🌊 Manual entry: Bull-Run exit (BULLRUN) and Bull-Run TP + custom SL (BULLRUN_SL) — operator 2026-10-03."""
import asyncio
import datetime as _dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import services.trading_engine as T  # noqa: E402

TH = T.config.trading_config.thresholds
F = T.manual_bullrun_exit_for


def test_widest_stop_and_own_stop():
    assert T.manual_bullrun_worst_stop_pct(TH) == min(TH.bullrun_base_sl_pct, TH.sl_atr_widen_floor_pct) == -1.2
    assert T.manual_own_stop_pct(TH, "BULLRUN", None) == 1.2
    assert T.manual_own_stop_pct(TH, "BULLRUN_SL", -2.5) == 2.5 and T.manual_own_stop_pct(TH, "BULLRUN_SL", 0.5) is None   # a lock is no loss stop

    class _T: bullrun_base_sl_pct = -0.7; sl_atr_widen_floor_pct = 0.0; sl_atr_multiplier = 1.5; manual_floor_sl_pct = -3.0
    assert T.manual_bullrun_worst_stop_pct(_T) is None and T.manual_own_stop_pct(_T, "BULLRUN", None) == 3.0                # uncapped → the config floor
    _T.sl_atr_multiplier = 0.0
    assert T.manual_bullrun_worst_stop_pct(_T) == -0.7


def test_the_sleeve_exit_on_a_manual_trade():
    # stop = min(−0.7, max(−1.5 × ATR, −1.2))
    assert F(-0.69, 0.0, 0.4, TH)[:2] == (False, None)                    # ATR 0.4 → stop −0.7 (fires at ≤): −0.69 holds
    assert F(-0.71, 0.0, 0.4, TH)[:2] == (True, "MANUAL_SL")
    assert F(-1.1, 0.0, 2.0, TH)[0] is False and F(-1.21, 0.0, 2.0, TH)[:2] == (True, "MANUAL_SL")   # ATR 2 → stop capped at −1.2
    # armed at +1: trail peak − 2 × ATR, never below the +0.2 lock
    c, why, line = F(0.25, 1.1, 0.5, TH); assert not c and abs(line - 0.2) < 1e-9
    assert F(0.19, 1.1, 0.5, TH)[:2] == (True, "MANUAL_TRAIL")
    assert abs(F(2.0, 3.0, 0.5, TH)[2] - 2.0) < 1e-9                             # 3.0 − 2 × 0.5
    # the ladder: peak 5.2 → floor 4.5 even with a wide ATR trail
    c, why, line = F(4.6, 5.2, 2.0, TH); assert not c and line == 4.5
    assert F(4.4, 5.2, 2.0, TH)[:2] == (True, "MANUAL_TRAIL")


def test_custom_stop_until_the_arm_then_the_bullrun_profit_side():
    assert F(-2.0, 0.5, 0.4, TH, custom_sl=-2.5)[0] is False                     # the Bull-Run stop (−0.7) is NOT used
    assert F(-2.51, 0.5, 0.4, TH, custom_sl=-2.5)[:2] == (True, "MANUAL_SL")
    assert F(0.6, 1.5, 0.5, TH, custom_sl=-2.5)[0] is False and F(0.45, 1.5, 0.5, TH, custom_sl=-2.5)[:2] == (True, "MANUAL_TRAIL")   # trail 1.5 − 2 × 0.5 = +0.5
    assert F(4.4, 5.2, 2.0, TH, custom_sl=-2.5)[:2] == (True, "MANUAL_TRAIL")    # ladder unchanged
    assert F(-0.71, 0.0, 0.4, TH, custom_sl=None)[0] is True                     # no custom stop = the plain Bull-Run exit


def test_leverage_floor_clamps_the_stop():
    fl = T.manual_floor_for_leverage(TH, 75)
    assert -1.2 < fl < -0.5
    assert F(fl - 0.01, 0.0, 2.0, TH, leverage=75)[:3] == (True, "MANUAL_SL", fl)  # the −1.2 stop would sit past the floor → the floor fires
    assert F(fl + 0.01, 0.0, 2.0, TH, leverage=75)[0] is False
    assert F("x", 0.0, 2.0, TH, leverage=20)[1] == "MANUAL_SL"                     # never raises


def test_live_backstop_floor_follows_the_custom_stop(monkeypatch):
    """Caveman review: in live the clamp must use THIS row's safety stop (built from its custom SL), not the plain Bull-Run one."""
    monkeypatch.setattr(TH, "broker_backstop_enabled", True, raising=False)
    bk = T.manual_backstop_stop_floor(False, TH, "BULLRUN_SL", -2.5, 20)
    assert bk < -2.5                                                             # the safety stop sits beyond the operator's −2.5
    assert F(-2.3, 0.0, 0.4, TH, leverage=20, is_paper=False, custom_sl=-2.5)[0] is False   # holds (was cut at −2.2 before the fix)
    assert F(-2.51, 0.0, 0.4, TH, leverage=20, is_paper=False, custom_sl=-2.5)[:2] == (True, "MANUAL_SL")
    assert F(-1.21, 0.0, 2.0, TH, leverage=20, is_paper=False)[:2] == (True, "MANUAL_SL")      # plain BULLRUN unchanged


def _engine():
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = True
    return TE, eng


def test_realtime_path_runs_both_modes(monkeypatch):
    TE, eng = _engine(); closed = []

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

    def entry(direction, mode, sl=None, peak=0.0, atr=0.5):
        return {'id': 1, 'direction': direction, 'entry_price': 100.0, 'quantity': 10.0, 'entry_fee': 1000.0 * fee, 'confidence': 'STRONG_BUY',
                'opened_at': T.datetime.utcnow(), 'entry_strategy': 'MANUAL', 'manual_exit_mode': mode, 'pattern_fixed_sl_pct': sl, 'pattern_fixed_tp_pct': None,
                'stop_loss': -0.7, 'peak_pnl': peak, 'trough_pnl': 0.0, 'leverage': 20.0, 'notional_value': 1000.0, 'investment': 50.0, 'entry_atr_pct': atr}

    async def drive(e, price):
        closed.clear(); T._open_orders_cache["ZZZUSDT"] = [e]
        await eng.check_realtime_stop_loss("ZZZUSDT", price)
        return list(closed)

    try:
        assert asyncio.run(drive(entry("LONG", "BULLRUN"), 99.0)) == ["MANUAL_SL"]           # −1 % < −0.75 stop (ATR 0.5)
        assert asyncio.run(drive(entry("LONG", "BULLRUN"), 99.5)) == []
        assert asyncio.run(drive(entry("LONG", "BULLRUN", peak=5.3, atr=2.0), 104.5)) == ["MANUAL_TRAIL"]   # ladder 4.5 (net ≈ 4.4)
        assert asyncio.run(drive(entry("SHORT", "BULLRUN"), 101.0)) == ["MANUAL_SL"]
        assert asyncio.run(drive(entry("LONG", "BULLRUN_SL", sl=-2.5), 98.0)) == []          # custom −2.5: −2 holds
        assert asyncio.run(drive(entry("LONG", "BULLRUN_SL", sl=-2.5), 97.3)) == ["MANUAL_SL"]
        assert asyncio.run(drive(entry("LONG", "BULLRUN_SL", sl=-2.5, peak=1.5), 100.15)) == ["MANUAL_TRAIL"]   # armed → lock +0.2
        assert asyncio.run(drive(entry("LONG", "BULLRUN_SL", sl=-2.5), 110.0)) == []          # no TP: a winner keeps running
    finally:
        T._open_orders_cache.pop("ZZZUSDT", None)


def test_open_checks(monkeypatch):
    TE, eng = _engine()

    async def _bal(db): return 10_000.0
    async def _bnb(db): return 1_000.0
    eng.get_available_balance = _bal; eng._recalculate_paper_bnb = _bnb

    class _Trk: last_price = 1.0
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)
    atr = {"v": 0.8}

    async def _stamps(pair, symbol, direction, price): return {"entry_atr_pct": atr["v"]}
    eng._manual_entry_stamps = _stamps

    async def _brk(): return {}
    monkeypatch.setattr(T.binance_service, "get_leverage_brackets", _brk)

    class _Accepted(Exception): pass
    captured = []

    class _Res:
        def scalar(self): return 0
        def first(self): return None
        def scalar_one_or_none(self): return None

    class _DB:
        async def execute(self, *a, **k): return _Res()
        def add(self, o): captured.append(o); raise _Accepted()

    def attempt(**kw):
        args = dict(pair="SANDUSDT", direction="LONG", investment=100, leverage=20); args.update(kw)
        try:
            asyncio.run(TE.open_manual_position(eng, _DB(), **args))
        except ValueError as e:
            return str(e)
        except _Accepted:
            return "ACCEPTED"
        return "?"

    assert attempt(exit_mode="BULLRUN") == "ACCEPTED" and captured[-1].manual_exit_mode == "BULLRUN"
    assert captured[-1].pattern_fixed_sl_pct is None and captured[-1].pattern_fixed_tp_pct is None
    assert attempt(exit_mode="BULLRUN", sl_pct=2.0, tp_pct=3.0) == "ACCEPTED" and captured[-1].pattern_fixed_sl_pct is None   # nothing typed is used
    assert "needs a stop loss" in attempt(exit_mode="BULLRUN_SL")
    assert attempt(exit_mode="BULLRUN_SL", sl_pct=2.5) == "ACCEPTED" and captured[-1].pattern_fixed_sl_pct == -2.5 and captured[-1].manual_exit_mode == "BULLRUN_SL"
    assert captured[-1].pattern_fixed_tp_pct is None
    assert attempt(exit_mode="BULLRUN_SL", sl_pct=2.5, tp_pct=3.0) == "ACCEPTED" and captured[-1].pattern_fixed_tp_pct is None   # no TP in this mode
    assert "wider than the widest stop" in attempt(exit_mode="BULLRUN_SL", sl_pct=9.0)
    assert "stop loss must be > 0" in attempt(exit_mode="BULLRUN_SL", sl_pct=0) and "stop loss must be > 0" in attempt(exit_mode="FIXED", sl_pct=0)
    assert "Bull-Run stop can reach" in attempt(exit_mode="BULLRUN", leverage=75)
    atr["v"] = None
    assert "no ATR reading" in attempt(exit_mode="BULLRUN") and "no ATR reading" in attempt(exit_mode="BULLRUN_SL", sl_pct=2.0)
    assert "exit mode must be" in attempt(exit_mode="BULL")


def test_wiring_both_paths_ui_and_override():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert eng.count('in ("FIXED", "FLOOR", "FRENZY", "BULLRUN", "BULLRUN_SL")') == 1 and eng.count("in ('FIXED', 'FLOOR', 'FRENZY', 'BULLRUN', 'BULLRUN_SL')") == 1
    assert eng.count("manual_bullrun_exit_for(") == 3                                   # def + monitor + realtime
    assert '(order.manual_exit_mode or "") == "BULLRUN_SL" and _br_unarmed' in eng            # an edit keeps the custom stop only before the arm
    assert "['MOMENTUM', 'FRENZY', 'BULLRUN', 'BULLRUN_SL'].includes" in ui and "['BULLRUN', 'BULLRUN_SL'].includes(o.manual_exit_mode || '')" in ui
    assert '<option value="BULLRUN">🌊 Bull-Run exit</option><option value="BULLRUN_SL">🌊 Bull-Run TP + custom SL</option>' in ui
    assert "sl.disabled = (m !== 'FIXED' && m !== 'BULLRUN_SL')" in ui and "m === 'BULLRUN' || m === 'BULLRUN_SL');" in ui
    assert "m === 'BULLRUN_SL' ? `exit: BULL-RUN TP + CUSTOM SL" in ui and "body.exit_mode === 'BULLRUN_SL' && !(body.sl_pct > 0)" in ui
