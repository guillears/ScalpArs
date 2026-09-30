"""Sep-14 VTHO bug: the realtime EMA13-cross exit closed a SPIKE_FADE short 36 ms after open because a
fade is entered ABOVE EMA13 by construction (the exit's short-side condition is true on tick 1).
The exit's sleeve scope is now a pure predicate; this pins it. Falsifiable: dropping either spike
species from the exclusion, or re-admitting flips, fails."""
from services.trading_engine import _ema13_cross_exit_applies as applies


def test_spike_fade_and_bounce_are_excluded():
    assert applies("SPIKE_FADE") is False      # short entered above EMA13 → cross true at t+0
    assert applies("SPIKE_BOUNCE") is False    # long entered below EMA13 → mirror case


def test_flips_stay_excluded_as_before():
    assert applies("FLIP:FAN_RATIO_GATE") is False
    assert applies("FLIP:PAIR_RSI_OB") is False


def test_momentum_and_other_sleeves_still_eligible():
    # SPIKE_CHASE stays eligible on purpose: a chase LONG is entered ABOVE EMA13 (cross false at
    # t+0) and exits via the option-D spike branch, not the fixed fade/bounce stack.
    for es in ("MOMENTUM", "", None, "SPIKE_CHASE", "BULL_LONG", "BOUNCE_LONG", "BULLRUN_LONG"):
        assert applies(es) is True, es


def test_fail_safe_on_garbage():
    assert applies(object()) is True


def test_surge_sleeves_are_excluded():
    # Sep-30 SOON (1 s) / MOVR (0.5 s): SURGE shorts open with no pair-EMA condition, 20 min after a dump → often above EMA13
    assert applies("SURGE_SHORT") is False
    assert applies("SURGE_LONG") is False
