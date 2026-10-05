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


def test_surge_long_no_lock_exit_matches_the_tested_variant():
    """🎯 Oct-5 (DECISION_LOG 203): the live no-lock exit == scripts/surge_bearrun_review.py BR_NO_BE_LOCK
    (line = max(sl, peak − 1×ATR) once peak ≥ +1 %, else the initial stop), and the lock path is unchanged when off."""
    import services.trading_engine as TE
    atr = 2.0
    sl = min(-0.70, max(-1.5 * atr, -1.2))                     # the initial stop both paths share (-1.2)
    for pk in (0.0, 0.5, 0.99, 1.0, 1.5, 3.0, 6.0, 12.0):
        want = max(sl, pk - atr) if pk >= 1.0 else sl
        _, _, line = TE._bullrun_exit_for(float("inf"), pk, atr, trail_mult_override=1.0, no_lock=True)
        assert abs(line - want) < 1e-9, (pk, line, want)
    # a +1.5 % peak giving back to −0.4 closes with the lock (floor +0.2) but not without it
    assert TE._bullrun_exit_for(-0.4, 1.5, atr, trail_mult_override=1.0)[0] is True
    assert TE._bullrun_exit_for(-0.4, 1.5, atr, trail_mult_override=1.0, no_lock=True)[0] is False
    assert TE._bullrun_exit_for(0.5, 3.0, atr, trail_mult_override=1.0, no_lock=True)[1] == "TRAILING_STOP"   # line +1.0
    assert TE._bullrun_exit_for(-1.3, 1.0, 3.0, trail_mult_override=1.0, no_lock=True)[1] == "STOP_LOSS"      # armed, gave it all back (line = sl)
    assert TE._bullrun_exit_for(-0.5, 1.6, atr, trail_mult_override=1.0, no_lock=True)[:2] == (True, "TRAILING_STOP")   # armed, line −0.4
    c, why, line = TE._bullrun_exit_for(4.4, 5.0, atr)            # BULLRUN_LONG keeps its lock + ladder (no_lock default False)
    assert c is True and why == "LADDER_FLOOR" and line >= 4.5
    import config as C
    th = C.trading_config.thresholds
    old = getattr(th, "surge_long_exit_no_lock", True)
    try:
        th.surge_long_exit_no_lock = True
        assert TE._surge_no_lock("SURGE_LONG") is True and TE._surge_no_lock("BULLRUN_LONG") is False
        th.surge_long_exit_no_lock = False
        assert TE._surge_no_lock("SURGE_LONG") is False
    finally:
        th.surge_long_exit_no_lock = old
    # NB: the replica check above uses today's JSON stop settings (bullrun_base_sl_pct −0.7 · sl_atr_multiplier 1.5 · floor −1.2) —
    # a settings change fails this test by design (re-check the replica equivalence), not a code bug.
