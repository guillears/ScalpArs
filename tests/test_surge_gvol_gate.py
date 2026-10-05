"""🌊 Oct-4 (DECISION_LOG 202): SURGE_LONG market-volume gate — pure rule (services.surge.surge_gvol_gate)."""
from types import SimpleNamespace

from services.surge import surge_gvol_gate


def th(g):
    return SimpleNamespace(surge_long_gvol_min=g)


def test_long_passes_at_and_above_the_gate():
    assert surge_gvol_gate(1.0, th(1.0), "LONG") == (True, None)
    assert surge_gvol_gate(2.4, th(1.0), "LONG") == (True, None)


def test_long_refused_below_the_gate():
    assert surge_gvol_gate(0.99, th(1.0), "LONG") == (False, "SURGE_GVOL_LOW")


def test_unreadable_is_fail_closed():
    assert surge_gvol_gate(None, th(1.0), "LONG") == (False, "SURGE_GVOL_UNREAD")
    assert surge_gvol_gate(float("nan"), th(1.0), "LONG") == (False, "SURGE_GVOL_UNREAD")
    assert surge_gvol_gate("x", th(1.0), "LONG") == (False, "SURGE_GVOL_UNREAD")


def test_gate_off_and_short_never_gated():
    assert surge_gvol_gate(None, th(0.0), "LONG") == (True, None)
    assert surge_gvol_gate(0.1, th(1.0), "SHORT") == (True, None)
    assert surge_gvol_gate(0.1, SimpleNamespace(), "LONG") == (True, None)   # missing key = off


def test_shipped_default_is_one():
    from config import SignalThresholds
    assert SignalThresholds().surge_long_gvol_min == 1.0


def test_no_automatic_kill_by_default():
    """🛑 Oct-5 operator: kill bars judge + record, never switch a sleeve off unless auto_kill_enabled."""
    import json, os
    from config import SignalThresholds
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert SignalThresholds().auto_kill_enabled is False
    live = json.load(open(os.path.join(root, "trading_config.json")))
    assert (live.get("thresholds", live)).get("auto_kill_enabled") is False
    eng = open(os.path.join(root, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("getattr(th, 'auto_kill_enabled', False)") >= 3       # SURGE verdict + SURGE log + recovery hold
    html = open(os.path.join(root, "templates", "index.html"), encoding="utf-8").read()
    assert html.count("config-auto-kill-enabled") == 3                     # input + load + save
