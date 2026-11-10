"""🎯 Nov-10 (DECISION_LOG 519 → REVERTED 521): Bull-Run REARM-door fixed take-profit (`bullrun_rearm_tp_pct`, now 0 = OFF).

Invariants: the mechanism fires only on door='REARM' at pnl ≥ the value (net %); GREEN / None / SURGE (trail override) / no-lock rows
never; 0 = off; below the TP the exit is unchanged; D11 parity (config.py default = JSON = 0 · UI input + load / save + export line).
The revert-gate scout was retired with the revert (DL 521).
"""
import json
import os
from types import SimpleNamespace


from services.trading_engine import _bullrun_exit_for

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _th(tp=1.0):
    return SimpleNamespace(bullrun_base_sl_pct=-0.7, sl_atr_multiplier=1.5, sl_atr_widen_floor_pct=-1.2, bullrun_be_arm_pct=1.0,
                           bullrun_be_lock_pct=0.2, bullrun_trail_atr_mult=2.0, bullrun_rearm_trail_atr_mult=1.0, bullrun_ladder="",
                           bullrun_rearm_tp_pct=tp)


def _x(monkeypatch, th, *a, **kw):
    monkeypatch.setattr("services.trading_engine.config", SimpleNamespace(trading_config=SimpleNamespace(thresholds=th)), raising=False)
    return _bullrun_exit_for(*a, **kw)


def test_rearm_tp_fires_at_the_value(monkeypatch):
    assert _x(monkeypatch, _th(), 1.0, 1.0, 0.5, "REARM") == (True, "REARM_TP", 1.0)
    assert _x(monkeypatch, _th(), 1.3, 1.3, 0.5, "rearm")[1] == "REARM_TP"
    close, why, _ = _x(monkeypatch, _th(), 0.99, 0.99, 0.5, "REARM")
    assert not close and why is None                                      # below the TP: today's exit, nothing fires


def test_never_on_green_surge_manual_or_off(monkeypatch):
    for door in ("GREEN", None, ""):
        assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, door)[1] != "REARM_TP"
    assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, "REARM", trail_mult_override=1.0)[1] != "REARM_TP"   # SURGE_LONG
    assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, "REARM", no_lock=True)[1] != "REARM_TP"
    assert _x(monkeypatch, _th(0.0), 1.5, 1.5, 0.5, "REARM")[1] != "REARM_TP"                          # 0 = off
    assert _x(monkeypatch, SimpleNamespace(**{k: v for k, v in vars(_th()).items() if k != "bullrun_rearm_tp_pct"}),
              1.5, 1.5, 0.5, "REARM")[1] != "REARM_TP"                                                     # absent = off


def test_stop_side_unchanged(monkeypatch):
    assert _x(monkeypatch, _th(), -0.8, 0.3, 0.5, "REARM") == (True, "STOP_LOSS", -0.75)


def test_d11_parity():
    import config
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["bullrun_rearm_tp_pct"] == config.SignalThresholds().bullrun_rearm_tp_pct == 0.0   # REVERTED, DL 521
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-br-rearm-tp-pct"') == 1
    assert "['config-br-rearm-tp-pct', 'bullrun_rearm_tp_pct', 0]" in html                            # load
    assert "bullrun_rearm_tp_pct: safeFloat(document.getElementById('config-br-rearm-tp-pct')" in html  # save
    assert "REARM TP ${" in html                                                                         # exported config line


def test_tp_fires_when_already_armed(monkeypatch):
    assert _x(monkeypatch, _th(), 1.2, 1.5, 0.5, "REARM") == (True, "REARM_TP", 1.0)


def test_scout_retired():
    assert not os.path.exists(os.path.join(ROOT, "scripts", "scout_bullrun_rearm_tp.py"))
    assert "scout_bullrun_rearm_tp" not in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
