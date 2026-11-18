"""🎯 (DECISION_LOG 563) FAN flip FIXED TP — a FLIP:FAN_RATIO_GATE SHORT stamped with flip_fan_tp_pct takes profit at that NET % (a resting maker
limit when resting_tp_flip_fan_enabled, else the software TP at the touch) instead of the strpk runner trail + the short HARD_TP ladder; the
stop / backstop / NO_EXPANSION / MAX_HOLD are unchanged; a row without the stamp (pre-563, or the setting 0) keeps the trail."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import resting_tp as R  # noqa: E402
from tests.test_frenzy_willy import _hold_db  # noqa: E402

FAN = "FLIP:FAN_RATIO_GATE"
PAIR = "FANUSDT"


def _row(**kw):
    d = dict(entry_strategy=FAN, direction="SHORT", flip_fan_tp_pct=0.5, exit_override_at=None, manual_exit_mode=None)
    d.update(kw)
    return NS(**d)


def test_row_target_rule():
    on, off = NS(resting_tp_flip_fan_enabled=True), NS(resting_tp_flip_fan_enabled=False)
    assert R.rtp_row_target(on, _row()) == (0.5, "FLIP_FAN_TP")
    assert R.rtp_row_target(off, _row()) is None                                  # switch off → no limit (the software TP runs)
    assert R.rtp_row_target(on, _row(flip_fan_tp_pct=None)) is None              # pre-563 row → the trail, no limit
    assert R.rtp_row_target(on, _row(flip_fan_tp_pct=0)) is None
    assert R.rtp_row_target(on, _row(direction="LONG")) is None
    assert R.rtp_row_target(on, _row(entry_strategy="FLIP:PAIR_RSI_OB")) is None
    assert R.rtp_target(on, FAN) is None                                          # row-derived only
    assert R.rtp_reason_for_limit(on, FAN, "SHORT", 1.0, 0.045, 0.99, 0.00018) == "FLIP_FAN_TP"
    assert R.flip_fan_row_tp(_row(flip_fan_tp_pct="x")) is None
    assert FAN in R.RTP_SLEEVES and FAN in R.RTP_ROW_STRATEGIES and "resting_tp_flip_fan_enabled" in R.RTP_SWITCHES
    assert R.stop_class("FLIP_STOP_LOSS L1") and not R.stop_class("FLIP_FAN_TP")


def test_wiring():
    import config
    d = config.SignalThresholds()
    assert d.flip_fan_fixed_tp_pct == 0.0 and d.resting_tp_flip_fan_enabled is False
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["flip_fan_fixed_tp_pct"] == 0.5 and th["resting_tp_flip_fan_enabled"] is True
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count("config-flip-fan-fixed-tp") == 3 and html.count("config-rtp-flip-fan-enabled") == 3
    assert "Fan-flip fixed TP (Nov 16, DECISION_LOG 563)" in html and "o.flip_fan_tp_pct" in html
    import models
    assert "flip_fan_tp_pct" in models.Order.__table__.columns
    assert "('flip_fan_tp_pct', 'FLOAT')" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert open(os.path.join(ROOT, "main.py"), encoding="utf-8").read().count('"flip_fan_tp_pct": getattr(o, \'flip_fan_tp_pct\', None)') == 2
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "flip_fan_tp_pct=_ff_tp," in eng and "'flip_fan_tp_pct': _ff_tp," in eng
    assert "'flip_fan_tp_pct': getattr(order, 'flip_fan_tp_pct', None)," in eng                  # cache rebuild (restart-proof)
    assert "or _fbs or _exh or _ff_tp is not None:" in eng   # 🪫 (DECISION_LOG 578) + EXHAUSTION                                                  # the resting limit is armed at the open
    assert "and _ff_tp_rt is None:   # 🎯 (563) a FAN flip on its fixed TP runs no ladder" in eng
    assert "if _ff_tp_rt is not None:   # 🎯 (DECISION_LOG 563) a FAN flip on its fixed TP: no strpk trail" in eng


def _run(monkeypatch, price, tp, peak=0.0, rtp=False, atr=0.4):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config.trading_config.thresholds, "resting_tp_flip_fan_enabled", rtp, raising=False)
    closed = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        import models
        opened = dt.datetime.utcnow() - dt.timedelta(minutes=5)
        async with SL() as s:
            o = models.Order(pair=PAIR, direction="SHORT", status="OPEN", is_paper=True, entry_strategy=FAN, confidence="STRONG_BUY",
                             entry_price=1.0, quantity=1000.0, investment=100.0, leverage=10, notional_value=1000.0, entry_fee=0.45,
                             opened_at=opened, flip_fan_tp_pct=tp, entry_atr_pct=atr)
            s.add(o); await s.commit(); oid = o.id
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; e.is_running = True
        e._record_filter_block = lambda *a, **k: None

        async def cp(db, order, px, reason):
            closed.append(reason); order.status = "CLOSED"; return order
        e.close_position = cp
        TE._open_orders_cache[PAIR] = [dict(id=oid, direction="SHORT", entry_price=1.0, quantity=1000.0, entry_fee=0.45, stop_loss=-0.70,
                                            entry_strategy=FAN, opened_at=opened, peak_pnl=peak, trough_pnl=0.0, confidence="STRONG_BUY",
                                            flip_fan_tp_pct=tp, entry_atr_pct=atr, current_tp_level=1)]
        try:
            await e.check_realtime_stop_loss(PAIR, price)
        finally:
            TE._open_orders_cache.pop(PAIR, None)
            await eng.dispose()
    asyncio.run(go())
    return closed


def test_software_tp_books_at_the_touch(monkeypatch):
    assert _run(monkeypatch, 0.993, 0.5) == ["FLIP_FAN_TP"]                       # ≈ +0.6 net ≥ +0.5, no resting limit alive
    assert _run(monkeypatch, 0.997, 0.5) == []                                    # ≈ +0.2 net: below the TP, above the stop → hold


def test_stop_unchanged(monkeypatch):
    out = _run(monkeypatch, 1.012, 0.5)                                           # ≈ −1.3 net, beyond the ATR-widened stop (−0.70)
    assert len(out) == 1 and "STOP_LOSS" in out[0]


def test_tp_row_skips_ladder_and_trail_but_trail_row_keeps_them(monkeypatch):
    # peak +1.20 then back to ≈ +0.40: the short HARD_TP ladder floor (1.0 − 0.25 = 0.75) / the strpk trail would close a trail row
    assert _run(monkeypatch, 0.9951, 0.5, peak=1.20) == []                         # TP row: still below its +0.5 TP — held (no ladder / trail)
    out = _run(monkeypatch, 0.9951, None, peak=1.20)
    assert len(out) == 1 and ("HARD_TP_LADDER" in out[0] or "RUNNER_TRAIL" in out[0])   # a pre-563 row keeps today's exit


def test_tp_row_never_gets_the_tier_trail_back():
    """(563 review) the tier trailing stop stays suppressed on a stamped FAN TP row whatever flip_fan_runner_strpk says."""
    import pathlib
    eng = (pathlib.Path(__file__).resolve().parents[1] / "services/trading_engine.py").read_text()
    i = eng.index("if _ff_tp_rt is not None:   # 🎯 (563 review) a FAN flip on its fixed TP never gets the tier trailing stop back")
    j = eng.index("if trailing_stop_would_be_active and order_info.get('tp_trailing_enabled', False) and not _handoff_suppress:", i)
    assert "_handoff_suppress = True" in eng[i:j] and j - i < 400
