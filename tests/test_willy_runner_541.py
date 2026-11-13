"""🏃 DECISION_LOG 541 — WILLY RUNNER: FRENZY_WILLY on coins with 1h EMA20 < EMA200 trails past its target at max(peak − k × ATR, floor, stop),
no resting TP on those rows, decided at the open and stored on the row; revert-gate scout."""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from services.frenzy import (frenzy_willy_exit_for, frenzy_willy_runner_config_errors, frenzy_willy_runner_k_for,  # noqa: E402
                             frenzy_willy_runner_line)


def _th(**kw):
    import config
    d = dict(config.trading_config.thresholds.model_dump())
    d.update(frenzy_willy_runner_enabled=True, frenzy_willy_runner_atr_k=2.0, frenzy_willy_runner_floor_pct=0.10, frenzy_willy_tp_pct=1.25,
             frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0, frenzy_willy_late_tp_pct=0.25)
    d.update(kw)
    return NS(**d)


def _src(p):
    return open(os.path.join(ROOT, p)).read()


def test_decided_at_open_only_falling_willy():
    th = _th()
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY", -3.0, 1.8) == 2.0
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY", 0.0, 1.8) is None          # rising (gap ≥ 0): today's exit
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY_HT", -3.0, 1.8) is None      # never HT
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY", None, 1.8) is None         # unknown gap → today's exit
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY", -3.0, None) is None        # unknown ATR → today's exit
    assert frenzy_willy_runner_k_for(_th(frenzy_willy_runner_enabled=False), "FRENZY_WILLY", -3.0, 1.8) is None


def test_runner_exit_path():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 0.5}
    assert frenzy_willy_exit_for(1.0, 1.0, th, 5, row=row)[0] is False
    assert frenzy_willy_exit_for(-4.0, 0.5, th, 5, row=row)[:2] == (True, "STOP_LOSS")
    c, why, line = frenzy_willy_exit_for(1.3, 1.3, th, 5, row=row)                 # the target touched → ARMS, never sells
    assert c is False and why == "WILLY_RUNNER_TRAIL" and abs(line - 0.30) < 1e-9   # max(1.3 − 1.0, 0.10)
    assert frenzy_willy_exit_for(2.1, 3.0, th, 10, row=row)[0] is False             # line 2.0
    assert frenzy_willy_exit_for(1.9, 3.0, th, 10, row=row)[:3] == (True, "WILLY_RUNNER_TRAIL", 2.0)
    assert frenzy_willy_exit_for(3.0, 3.0, th, 95, row=row)[:2] == (True, "MAX_HOLD_TIME")   # the cap stays once armed


def test_arms_where_todays_limit_would_fill():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 0.5}
    assert frenzy_willy_exit_for(1.23, 1.23, th, 5, row=row)[1] == "WILLY_RUNNER_TRAIL"   # taker-net 1.23 = the maker limit at +1.25 filled
    assert frenzy_willy_exit_for(1.20, 1.20, th, 5, row=row)[1] != "WILLY_RUNNER_TRAIL"


def test_scout_compares_ex_funding():
    import scout_willy_runner_watch as W
    assert abs(W._ex_funding({"pnl_percentage": 1.0, "funding_fee_usd": -1.0, "notional_value": 1000.0, "funding_in_pnl": True}) - 1.1) < 1e-9
    assert W._ex_funding({"pnl_percentage": 1.0, "funding_fee_usd": -1.0, "notional_value": 1000.0, "funding_in_pnl": None}) == 1.0


def test_floor_keeps_a_target_hit_trade_out_of_the_red():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 2.0}
    assert frenzy_willy_exit_for(0.2, 1.6, th, 10, row=row)[0] is False             # line = max(1.6 − 4, 0.10) = 0.10
    assert frenzy_willy_exit_for(0.05, 1.6, th, 10, row=row)[:3] == (True, "WILLY_RUNNER_TRAIL", 0.10)
    assert frenzy_willy_runner_line(1.6, 1.25, 2.0, 2.0, 0.10, -4.0) == 0.10


def test_late_tp_before_arming_only():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 0.25}
    assert frenzy_willy_exit_for(0.3, 1.0, th, 31, row=row)[:2] == (True, "WILLY_LATE_TP")   # never armed → today's late TP
    c, why, line = frenzy_willy_exit_for(0.8, 1.5, th, 31, row=row)                 # armed earlier: no late TP, the trail (line 1.0)
    assert (c, why, line) == (True, "WILLY_RUNNER_TRAIL", 1.0)
    assert frenzy_willy_exit_for(1.2, 1.5, th, 31, row=row)[0] is False


def test_non_runner_rows_unchanged():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": None, "entry_atr_pct": 0.5}
    assert frenzy_willy_exit_for(1.3, 1.3, th, 5, row=row)[:2] == (True, "FRENZY_TP")
    run = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 0.5}
    assert frenzy_willy_exit_for(1.3, 1.3, th, 5, resting_tp=True, row=run) == frenzy_willy_exit_for(1.3, 1.3, th, 5, resting_tp=True, row=row)


def test_no_resting_tp_on_runner_rows():
    from services.resting_tp import rtp_row_target
    th = _th(resting_tp_willy_enabled=True)
    base = dict(entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=1.5, exit_override_at=None, direction="LONG", manual_exit_mode=None)
    assert rtp_row_target(th, NS(**base, frenzy_willy_runner_k=None), 0.0) is not None
    assert rtp_row_target(th, NS(**base, frenzy_willy_runner_k=2.0), 0.0) is None


def test_config_errors():
    assert frenzy_willy_runner_config_errors(_th()) == []
    assert frenzy_willy_runner_config_errors(_th(frenzy_willy_runner_atr_k=0)) != []
    assert frenzy_willy_runner_config_errors(_th(frenzy_willy_runner_floor_pct=1.5)) != []
    assert frenzy_willy_runner_config_errors(_th(frenzy_willy_runner_enabled="yes")) != []


def test_scout_gate():
    import scout_willy_runner_watch as W
    assert W.gate([-3.0, -2.5], [True, True])[0] == "REVERT"                       # early stop: Σ < −5
    keep = W.gate([0.5] * 10 + [0.0] * 3, [True] * 10 + [False] * 3)
    assert keep[0] == "KEEP"
    rev = W.gate([0.5] * 9 + [-4.6], [True] * 10)
    assert rev[0] == "REVERT"
    assert W.gate([0.5] * 10, [True] * 10, held=True)[0] == "OBSERVING"            # an earlier fill still pending
    assert W.gate([0.1] * 5, [True] * 5)[0] == "OBSERVING"
    assert W.gate([-0.3] * 30, [False] * 30)[0] == "OBSERVING"                     # stops with cost noise never fire the gate (review)
    assert W.gate([0.5] * 10 + [-9.0], [True] * 10 + [True])[0] == "KEEP"           # FINAL = the first 10 runner fills


def test_missing_atr_falls_back_to_today():
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": None}
    assert frenzy_willy_exit_for(1.3, 1.3, th, 5, row=row)[:2] == (True, "FRENZY_TP")


def test_wiring():
    import json
    s = _src("services/trading_engine.py")
    assert "_wrun_k = frenzy_willy_runner_k_for(config.trading_config.thresholds, _fz_es, _z_pair_gap, entry_atr_pct)" in s
    assert "frenzy_willy_runner_k=_wrun_k," in s and "'frenzy_willy_runner_k': _wrun_k," in s
    assert "'frenzy_willy_runner_k': getattr(order, 'frenzy_willy_runner_k', None)," in s
    assert _src("main.py").count('"frenzy_willy_runner_k": getattr(o, \'frenzy_willy_runner_k\', None),') == 2
    assert "frenzy_willy_runner_config_errors as _wrun_cfg_errors" in _src("main.py")
    assert "frenzy_willy_runner_k = Column(Float" in _src("models.py") and "('frenzy_willy_runner_k', 'FLOAT')" in _src("database.py")
    h = _src("templates/index.html")
    for i in ("config-fz-willy-runner'", "config-fz-willy-runner-k", "config-fz-willy-runner-floor"):
        assert i in h
    assert "frenzy_willy_runner_enabled: document.getElementById('config-fz-willy-runner')?.checked" in h
    assert "🏃 WILLY runner (DECISION_LOG 541)" in h
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert j["frenzy_willy_runner_enabled"] is True and j["frenzy_willy_runner_atr_k"] == 2.0 and j["frenzy_willy_runner_floor_pct"] == 0.1
    assert "    frenzy_willy_runner_enabled: bool = False\n" in _src("config.py")                 # code default OFF (the JSON arms it)
    assert "scout_willy_runner_watch.py" in _src("scripts/observe_registry.py") and "import scout_willy_runner_watch" in _src("scripts/opportunity_scout.py")


def test_engine_rule_matches_study_walker_on_a_path():
    """the tick rule and the study's leg() agree on a simple path: up to the target, on to +4, back down through the trail line."""
    import study_willy_ht_exit_1112 as S
    th = _th()
    row = {"frenzy_willy_tp_target_pct": 1.25, "frenzy_willy_runner_k": 2.0, "entry_atr_pct": 0.5}
    pnl = list(np.linspace(0, 1.25, 6)) + list(np.linspace(1.3, 4.0, 10)) + list(np.linspace(3.9, 2.5, 10))
    pk, out = -1e9, None
    for i, p in enumerate(pnl):
        pk = max(pk, p)
        c, why, line = frenzy_willy_exit_for(p, pk, th, i * 0.5, row=row)
        if c:
            out = (why, line)
            break
    assert out == ("WILLY_RUNNER_TRAIL", 3.0)
    t = np.arange(len(pnl)) * 30_000
    P = dict(te=0, t=t, hi=np.array(pnl), lo=np.array(pnl), cl=np.array(pnl), op=np.array(pnl), bar=np.zeros(len(pnl), bool), end_ms=10**12)
    v, w = S.leg(P, 1.25, run_k=2.0, atr=0.5, be_floor=0.10)
    assert w == "TRAIL" and abs((v + S.SLIP) - 3.0) < 0.2
