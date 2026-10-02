"""🩹 Recovery hold (DECISION_LOG 172) — pure rules + the wiring that must not drift (whitelists, urgent lists, D11/D12)."""
import json
import os
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

import models
import services.recovery_hold as RH
import services.trading_engine as TE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TH = SimpleNamespace(recovery_hold_enabled=True, recovery_hold_rsi_min=60.0, recovery_hold_rsi_max=66.0, recovery_hold_room_pct=0.5,
                     recovery_hold_time_min=30.0, recovery_hold_max_min=240.0, recovery_hold_release_pct=0.40)
ok = lambda **k: RH.rh_trigger_ok(TH, **{**dict(reason="STOP_LOSS L1", strategy="MOMENTUM", direction="LONG", already_triggered=False,
                                                 fl_flagged=False, pnl=-0.71, stop_level=-0.70, peak_pnl=0.05, rsi_entry=60.4, rsi_now=63.8,
                                                 rsi_is_fresh=True), **k})


def test_closed_rsi_is_the_research_ruler_and_drops_the_forming_bar():
    rng = np.random.default_rng(3); c = 100 + np.cumsum(rng.normal(0, 0.3, 120)); t0 = 1_790_000_000_000
    bars = [[t0 + i * 300_000, 0, 0, 0, float(x), 0] for i, x in enumerate(c)]
    now = bars[-1][0] + 120_000                                            # the last bar is still forming
    d = pd.Series(c[:-1]).diff(); u = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    want = float((100 - 100 / (1 + u / v)).iloc[-1])
    got, ts = RH.closed_rsi(bars, now)
    assert abs(got - want) < 1e-9 and ts == bars[-2][0]
    got2, ts2 = RH.closed_rsi(bars, bars[-1][0] + 300_000)                 # once it closes it is used
    assert ts2 == bars[-1][0] and got2 != got
    assert RH.closed_rsi(bars[:20], now) == (None, None) and RH.closed_rsi(None, now) == (None, None)


def test_freshness_window():
    t = 1_790_000_000_000
    assert RH.rsi_fresh(t, t + 300_000) and RH.rsi_fresh(t, t + 599_000) and RH.rsi_fresh(t, t + 12 * 60_000)
    assert not RH.rsi_fresh(t, t + 13 * 60_000) and not RH.rsi_fresh(None, t) and not RH.rsi_fresh(t, t - 1)


def test_trigger_needs_every_condition():
    assert ok()                                                            # the ZRO case: 60.4 → 63.8, inside 60–66
    assert ok(reason="STOP_LOSS_WIDE L1") and ok(rsi_entry=64.2, rsi_now=64.2) and ok(rsi_now=60.0, rsi_entry=58.0) and ok(rsi_now=66.0)
    assert not ok(rsi_now=59.9, rsi_entry=58.0)                            # below the band
    assert not ok(rsi_now=66.1)                                            # above the band (stretched)
    assert not ok(rsi_entry=64.0, rsi_now=63.9)                            # BTC weakened since entry
    assert not ok(strategy="BULLRUN_LONG") and not ok(strategy="SPIKE_FADE") and not ok(strategy="MANUAL") and not ok(direction="SHORT")
    assert not ok(reason="BREAKEVEN_EXIT_L1") and not ok(reason="RUNNER_TRAIL") and not ok(reason=None)
    assert not ok(already_triggered=True) and not ok(fl_flagged=True)      # once per trade; never on an FL-flagged trade
    assert not ok(rsi_entry=None) and not ok(rsi_now=None) and not ok(rsi_is_fresh=False) and not ok(stop_level=None) and not ok(pnl=None)
    assert not ok(pnl=-1.21) and not ok(pnl=-1.195) and ok(pnl=-1.18)      # at / through the hard stop (−0.70 − 0.5, same 0.01 band as the exit) → plain stop
    assert ok(floor_pct=-2.2) and not ok(floor_pct=-1.2) and not ok(stop_level=-1.8, pnl=-1.81, floor_pct=-2.2)   # never into the exchange backstop
    assert not ok(peak_pnl=0.40)                                           # already past the release level
    assert not RH.rh_trigger_ok(SimpleNamespace(**{**TH.__dict__, "recovery_hold_enabled": False}), "STOP_LOSS L1", "MOMENTUM", "LONG",
                                False, False, -0.71, -0.70, 0.0, 60.4, 63.8, True)


def test_hold_exits_in_order_and_fail_closed():
    x = lambda **k: RH.rh_exit(TH, **{**dict(pnl=-0.9, minutes_held=5, hard_stop=-1.20, rsi_entry=60.4, rsi_now=63.8, rsi_is_fresh=True), **k})
    assert x() is None                                                     # keep holding
    assert x(pnl=-1.20) == "RH_HARD_STOP" and x(pnl=-1.195) == "RH_HARD_STOP" and x(pnl=-1.18) is None
    assert x(hard_stop=None) == "RH_HARD_STOP" and x(pnl=None) == "RH_HARD_STOP"
    assert x(rsi_now=59.9) == "RH_PREMISE_EXIT" and x(rsi_now=60.3) == "RH_PREMISE_EXIT"      # below the band · below entry
    assert x(rsi_is_fresh=False) == "RH_PREMISE_EXIT" and x(rsi_now=None) == "RH_PREMISE_EXIT" and x(rsi_entry=None) == "RH_PREMISE_EXIT"
    assert x(rsi_now=70.0) is None                                         # the band maximum gates the TRIGGER only
    assert x(minutes_held=30) == "RH_TIME_EXIT" and x(minutes_held=29.9) is None
    assert x(minutes_held=45, pnl=0.10) is None and x(minutes_held=240, pnl=0.10) == "RH_TIME_EXIT"
    assert x(pnl=-1.3, rsi_now=50, minutes_held=99) == "RH_HARD_STOP"      # the hard stop wins
    # a process that has not read BTC yet (restart / deploy) must not dump an open hold: premise skipped, hard stop + time still apply
    assert x(rsi_now=None, rsi_is_fresh=False, rsi_ever_read=False) is None
    assert x(rsi_now=None, rsi_is_fresh=False, rsi_ever_read=False, pnl=-1.25) == "RH_HARD_STOP"
    assert x(rsi_now=None, rsi_is_fresh=False, rsi_ever_read=False, minutes_held=31) == "RH_TIME_EXIT"
    zero = SimpleNamespace(**{**TH.__dict__, "recovery_hold_time_min": 0.0, "recovery_hold_max_min": 0.0})
    assert RH.rh_exit(zero, -0.9, 1, -1.2, 60.4, 63.8, True) is None       # 0 in the UI never means "exit at once"


def test_release_ends_the_hold():
    t = datetime.utcnow()
    assert RH.rh_in_hold(TH, t, 0.0) and RH.rh_in_hold(TH, t, 0.39) and RH.rh_in_hold(TH, t, None)
    assert not RH.rh_in_hold(TH, t, 0.40) and not RH.rh_in_hold(TH, None, 0.0)


def test_reason_naming():
    assert RH.rh_prefixed("RUNNER_TRAIL", True) == "RH_RUNNER_TRAIL" and RH.rh_prefixed("STOP_LOSS L1", False) == "STOP_LOSS L1"
    for r in ("RH_HARD_STOP", "MANUAL", "MANUAL_TP", "BACKSTOP_STOP"):
        assert RH.rh_prefixed(r, True) == r
    assert RH.rh_strip("RH_RUNNER_TRAIL") == "RUNNER_TRAIL" and RH.rh_strip("RH_STOP_LOSS L1") == "STOP_LOSS L1"
    for r in RH.RH_STOP_CLASS:
        assert RH.rh_strip(r) == r                                         # the hold's own closes keep their name
    assert TE._strip_reason_prefixes("RH_RUNNER_TRAIL_LATE") == "RUNNER_TRAIL_LATE" and TE._strip_reason_prefixes("FLIP_STOP_LOSS L1") == "STOP_LOSS L1"
    assert RH.rh_strip("FL_RH_TRAILING_STOP L1") == "FL_TRAILING_STOP L1" and TE._strip_reason_prefixes("FL_RH_TRAILING_STOP L1") == "TRAILING_STOP L1"
    assert RH.rh_strip("FL_RH_HARD_STOP") == "FL_RH_HARD_STOP"            # FL_ is added after RH_ in the close path; the hold's own closes keep their name


def test_kill_bar():
    win, hard = (0.3, -0.7, "RH_RUNNER_TRAIL"), (-1.2, -0.7, "RH_HARD_STOP")
    assert RH.rh_tripwire([win, hard, hard]) == (None, False)
    assert RH.rh_tripwire([win, hard, hard, hard]) == ("3 hard stops in a row", True)
    assert RH.rh_tripwire([win, hard, hard, (-1.2, -0.7, "FL_RH_HARD_STOP")])[0] == "3 hard stops in a row"
    assert RH.rh_tripwire([hard, hard, win] * 3) == (None, False)          # 9 holds, never 3 in a row
    why, judged = RH.rh_tripwire([hard, hard, win] * 3 + [win])            # 10 holds: 6 × −0.5 + 4 × +1.0 = +1.0
    assert why is None and judged
    why, judged = RH.rh_tripwire([hard, hard, (-0.6, -0.7, "RH_TIME_EXIT")] * 3 + [hard])
    assert judged and why.startswith("first 10 holds -3.20")                # 7 × −0.5 + 3 × +0.1, never 3 hard stops in a row
    why, judged = RH.rh_tripwire([hard, (-0.6, -0.7, "RH_TIME_EXIT")] * 5)
    assert judged and why.startswith("first 10 holds -2.00")


def test_pnl_helper_matches_the_exit_arithmetic(monkeypatch):
    import config
    monkeypatch.setattr(config.trading_config, "taker_fee", 0.0005, raising=False)
    p = TE._rh_pnl_pct("LONG", 100.0, 99.0, 10.0, 0.5)                     # −10 raw − 0.5 entry − 0.495 exit over 1000
    assert abs(p - (-1.0995)) < 1e-9 and TE._rh_pnl_pct("LONG", 0, 1, 1, 0) == 0.0


def test_wiring_cannot_drift():
    src = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert src.count("not _reason_base.startswith(RH_STOP_CLASS) and not (_reason_base.startswith(\"BREAKEVEN_EXIT\")") == 2   # live + restart whitelists
    assert src.count("_reason_base = _rh_strip(") == 4                                                                         # before and after the FLIP_/FL_/BR_ strips, both whitelists
    assert src.count("reason.startswith(RH_STOP_CLASS) or any(_rh_strip(reason).startswith(p) for p in (") == 2              # live + paper urgent lists
    assert "reason = _rh_prefixed(reason, getattr(order, 'rh_triggered_at', None) is not None)" in src                         # the close funnel
    assert src.count("rh_trigger_ok(") == 2 and src.count("rh_exit(") == 2                                                    # monitor + realtime, same rules
    assert src.count("rsi_ever_read=_current_btc_rsi_closed_bar_ts is not None") == 2 and src.count("floor_pct=_rh_backstop_floor(getattr(self, 'is_paper_mode', True))") == 2
    # the realtime hold decision sits BEFORE every other realtime exit, the monitor's before every exit but MAX_HOLD
    rt = src[src.index("async def check_realtime_stop_loss"):]
    assert rt.index("_rh_reason_rt = rh_exit(") < min(rt.index(k) for k in ("PATTERN_FIXED_TP L1", "EMA13_CROSS_EXIT", "FAST_EXIT", '_mn_lbl + "_TP"', "STOP_LOSS_WIDE L{tp_level}"))
    mon = src[src.index("async def update_open_positions"):src.index("async def scan_and_trade")]
    assert mon.index("MAX_HOLD_TIME") < mon.index("_rh_reason_m = rh_exit(") < min(mon.index(k) for k in ("REGIME_CHANGE L", "SIGNAL_LOST L", "check_exit_conditions("))
    # the 1 Hz cache rebuild and the open-time cache entry carry the hold's keys
    assert "new_info['rh_triggered_at'] = old_info.get('rh_triggered_at')" in src and "'entry_btc_rsi_closed': order.entry_btc_rsi_closed," in src
    assert src.count("await self._rh_kill_bar_check(db)") == 3                                                                # close funnel · exit-retry queue · startup
    ind = open(os.path.join(ROOT, "services", "indicators.py")).read()
    assert '"stop_level": effective_stop_loss' in ind
    for c in ("entry_btc_rsi_closed", "rh_triggered_at", "rh_trigger_pnl", "rh_btc_rsi", "rh_stop_level_pct", "rh_hard_stop_pct"):
        assert c in models.Order.__table__.columns and c in open(os.path.join(ROOT, "database.py")).read()


def test_config_ui_and_reports_parity():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    import config
    fields = {"recovery_hold_enabled": "config-rh-enabled", "recovery_hold_rsi_min": "config-rh-rsi-min", "recovery_hold_rsi_max": "config-rh-rsi-max",
              "recovery_hold_room_pct": "config-rh-room", "recovery_hold_time_min": "config-rh-time", "recovery_hold_max_min": "config-rh-max",
              "recovery_hold_release_pct": "config-rh-release"}
    for k, idn in fields.items():
        assert k in cfg and hasattr(config.SignalThresholds(), k)
        assert html.count(f'id="{idn}"') == 1 and html.count(f"'{idn}'") >= 2 and html.count(k) >= 2      # input + load + save
    assert "recovery_hold_kill_verdict" in cfg and 'id="rh-verdict"' in html
    assert html.count("## 🩹 Recovery Hold") == 2 and html.count('id="recovery-hold-body"') == 1          # both exports + the UI table
    assert not re.search(r"recovery_hold_kill_verdict\s*:", html)                                          # the page never writes the verdict back


def test_table_math():
    import main
    t = datetime(2026, 10, 2, 2, 3)
    o = lambda pnl, trig, reason, n=10_000.0: SimpleNamespace(rh_triggered_at=t, closed_at=t + timedelta(minutes=13), pnl_percentage=pnl, rh_trigger_pnl=trig,
                                                              close_reason=reason, notional_value=n, entry_price=1.0, quantity=n, pair="ZROUSDT", entry_btc_rsi_closed=60.4,
                                                              rh_btc_rsi=63.8, rh_stop_level_pct=-1.1, rh_hard_stop_pct=-1.6)
    r = main._compute_recovery_hold([o(0.34, -1.10, "RH_RUNNER_TRAIL", 28_476.0), o(-1.62, -1.10, "RH_HARD_STOP"),
                                     SimpleNamespace(rh_triggered_at=None, closed_at=t)])
    assert r["n"] == 2 and r["better"] == 1 and r["worse"] == 1 and abs(r["delta_pct"] - 0.92) < 1e-9
    assert abs(r["delta_usd"] - (1.44 / 100 * 28_476 - 0.52 / 100 * 10_000)) < 0.01 and r["trades"][0]["held_min"] == 13.0
    assert main._compute_recovery_hold([])["n"] == 0


def test_realtime_path_behaviour(monkeypatch):
    """Behavioural: drive check_realtime_stop_loss on a MOMENTUM LONG cache entry — the stop becomes a hold only under the
    condition, a held trade ignores the plain stop and closes only on the hold's exits, and a just-restarted process keeps it."""
    import asyncio
    import time
    T = TE; eng = T.TradingEngine.__new__(T.TradingEngine); eng.is_paper_mode = True
    closed = []; row = SimpleNamespace(rh_triggered_at=None, rh_trigger_pnl=None, rh_btc_rsi=None, rh_stop_level_pct=None, rh_hard_stop_pct=None)

    class _Res:
        def scalar_one_or_none(self): return row

    class _DB:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *a, **k): return _Res()

    async def _close(db, order, price, reason): closed.append(reason); return True
    async def _commit(db): return None
    monkeypatch.setattr(T, "AsyncSessionLocal", lambda: _DB())
    monkeypatch.setattr(T, "locked_commit", _commit)
    monkeypatch.setattr(eng, "close_position", _close, raising=False)
    th = T.config.trading_config.thresholds
    for k, v in dict(recovery_hold_enabled=True, recovery_hold_rsi_min=60.0, recovery_hold_rsi_max=66.0, recovery_hold_room_pct=0.5,
                     recovery_hold_time_min=30.0, recovery_hold_max_min=240.0, recovery_hold_release_pct=0.40).items():
        monkeypatch.setattr(th, k, v, raising=False)
    fee = float(getattr(T.config.trading_config, 'taker_fee', T.config.trading_config.trading_fee))

    def entry(**k):
        return {**{'id': 7, 'direction': 'LONG', 'entry_price': 100.0, 'quantity': 10.0, 'entry_fee': 1000.0 * fee, 'confidence': 'STRONG_BUY',
                   'opened_at': T.datetime.utcnow(), 'entry_strategy': 'MOMENTUM', 'stop_loss': -0.7, 'peak_pnl': 0.0, 'trough_pnl': 0.0, 'leverage': 20.0,
                   'notional_value': 1000.0, 'investment': 50.0, 'entry_btc_rsi_closed': 60.4, 'be_levels_enabled': False}, **k}

    def rsi(now, fresh=True, ever=True):
        monkeypatch.setattr(T, "_current_btc_rsi_closed", now, raising=False)
        monkeypatch.setattr(T, "_current_btc_rsi_closed_bar_ts", (int(time.time() * 1000) - (300_000 if fresh else 3_600_000)) if ever else None, raising=False)

    async def drive(price, e):
        closed.clear(); T._open_orders_cache["ZZZUSDT"] = [e]
        await eng.check_realtime_stop_loss("ZZZUSDT", price)
        return list(closed), e

    run = lambda price, e: asyncio.run(drive(price, e))
    rsi(63.8)
    c, e = run(99.2, entry())                                              # ≈ −0.9 % after fees: through the −0.70 stop, above the −1.20 hard stop
    assert c == [] and e.get('rh_triggered_at') is not None and abs(e['rh_hard_stop_pct'] - (-1.2)) < 0.02 and row.rh_triggered_at is not None
    c, e = run(99.2, e); assert c == []                                    # held: the plain stop no longer fires
    c, e = run(98.6, e); assert c == ["RH_HARD_STOP"]                      # through the hard stop
    held = lambda: entry(rh_triggered_at=T.datetime.utcnow(), rh_hard_stop_pct=-1.2)
    rsi(59.0); assert run(99.2, held())[0] == ["RH_PREMISE_EXIT"]          # BTC weakened
    rsi(63.8, fresh=False); assert run(99.2, held())[0] == ["RH_PREMISE_EXIT"]   # reading went stale → fail closed
    rsi(None, ever=False); assert run(99.2, held())[0] == []               # just restarted: no reading yet → keep holding …
    assert run(98.6, held())[0] == ["RH_HARD_STOP"]                        # … but the hard stop still protects
    rsi(63.8)
    old = entry(rh_triggered_at=T.datetime.utcnow() - timedelta(minutes=31), rh_hard_stop_pct=-1.2)
    assert run(99.2, old)[0] == ["RH_TIME_EXIT"]
    # no hold when the condition is not met — the stop closes exactly as before
    for setup in (lambda: rsi(59.0), lambda: rsi(67.0), lambda: rsi(60.0), lambda: rsi(63.8, fresh=False), lambda: rsi(None, ever=False)):
        setup(); row.rh_triggered_at = None
        c, e = run(99.2, entry(entry_btc_rsi_closed=60.4))
        assert len(c) == 1 and c[0].startswith("STOP_LOSS") and e.get('rh_triggered_at') is None, c
    rsi(63.8); row.rh_triggered_at = None
    assert run(99.2, entry(entry_btc_rsi_closed=None))[0][0].startswith("STOP_LOSS")          # no entry stamp → plain stop
    monkeypatch.setattr(th, "recovery_hold_enabled", False, raising=False)
    assert run(99.2, entry())[0][0].startswith("STOP_LOSS")                                    # switched off → plain stop
    assert run(99.2, held())[0] == []                                                          # an open hold keeps running after the switch-off
    T._open_orders_cache.pop("ZZZUSDT", None)
